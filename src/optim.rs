//! This module provides optimization utilities for spike trains memorization.
use grb::prelude::*;
use itertools::Itertools;
use log::{debug, error, info, trace, warn};
use rand::Rng;

use super::connection::{Connection, Input};
use super::error::SNNError;
use crate::{FIRING_THRESHOLD, REFRACTORY_PERIOD};
use super::spike_train::{InputSpike, Spike};

/// The tolerance for a constraint to be considered as satisfied
pub const CONSTRAINT_TOLERANCE: f64 = 1e-6;
/// The number of iterations for the computation of the jitter stability eigenvalue.
pub const NUM_ITER_EIGENVALUE: usize = 10;

#[derive(Clone, Copy)]
pub enum Objective {
    /// Flat objective function, i.e., uniform prior.
    None,
    /// L2 norm, i.e., Gaussian prior.
    L2Norm,
    /// L1 norm, i.e., Laplace prior.
    L1Norm,
}

pub fn init_gurobi(name: &str, log_file: &str) -> Result<grb::Model, SNNError> {
    let mut env = Env::empty().map_err(|e| SNNError::GurobiError(e.to_string()))?;
    env.set(param::OutputFlag, 0)
        .map_err(|e| SNNError::GurobiError(e.to_string()))?;
    env.set(param::UpdateMode, 1)
        .map_err(|e| SNNError::GurobiError(e.to_string()))?;
    env.set(param::LogFile, log_file.to_string())
        .map_err(|e| SNNError::GurobiError(e.to_string()))?;
    let env: Env = env
        .start()
        .map_err(|e| SNNError::GurobiError(e.to_string()))?;
    Model::with_env(name, env).map_err(|e| SNNError::GurobiError(e.to_string()))
}

pub fn init_weights(
    model: &mut grb::Model,
    num_weights: usize,
    lim_weights: (f64, f64),
) -> Result<Vec<grb::Var>, SNNError> {
    let (min_weight, max_weight) = lim_weights;
    if !(min_weight <= max_weight) {
        return Err(SNNError::InvalidWeight);
    }

    let weights = match (min_weight.is_infinite(), max_weight.is_infinite()) {
        (false, false) => (0..num_weights)
            .map(|_| add_ctsvar!(model, bounds: min_weight..max_weight).unwrap())
            .collect::<Vec<Var>>(),
        (true, false) => (0..num_weights)
            .map(|_| add_ctsvar!(model, bounds: ..max_weight).unwrap())
            .collect::<Vec<Var>>(),
        (false, true) => (0..num_weights)
            .map(|_| add_ctsvar!(model, bounds: min_weight..).unwrap())
            .collect::<Vec<Var>>(),
        (true, true) => (0..num_weights)
            .map(|_| add_ctsvar!(model, bounds: ..).unwrap())
            .collect::<Vec<Var>>(),
    };
    Ok(weights)
}

pub fn init_objective(
    model: &mut grb::Model,
    weights: &Vec<grb::Var>,
    objective: Objective,
) -> Result<(), SNNError> {
    match objective {
        Objective::None => (),
        Objective::L2Norm => {
            let mut obj_expr = grb::expr::QuadExpr::new();
            for &var in weights.iter() {
                obj_expr.add_qterm(1.0, var, var);
            }
            model
                .set_objective(obj_expr, ModelSense::Minimize)
                .map_err(|e| SNNError::GurobiError(e.to_string()))?;
        }
        Objective::L1Norm => {
            let mut obj_expr = grb::expr::LinExpr::new();
            for (i, &var) in weights.iter().enumerate() {
                let slack = add_ctsvar!(model).unwrap();
                obj_expr.add_term(1.0, slack);
                model
                    .add_constr(format!("min_slack_{}", i).as_str(), c!(var >= -slack))
                    .map_err(|e| SNNError::GurobiError(e.to_string()))?;
                model
                    .add_constr(format!("max_slack_{}", i).as_str(), c!(var <= slack))
                    .map_err(|e| SNNError::GurobiError(e.to_string()))?;
            }
            model
                .set_objective(obj_expr, ModelSense::Minimize)
                .map_err(|e| SNNError::GurobiError(e.to_string()))?;
        }
    };
    Ok(())
}

/// Add threshold-crossing constraint at every firing time.
pub fn add_firing_time_constraints(
    // &self,
    model: &mut Model,
    weights: &Vec<Var>,
    firing_times: &Vec<f64>,
    input_spikes: &Vec<InputSpike>,
    period: f64,
) -> Result<(), SNNError> {
    for &ft in firing_times {
        let mut expr = grb::expr::LinExpr::new();
        for input_spike in input_spikes.iter() {
            let dt = (ft - input_spike.time()).rem_euclid(period);
            expr.add_term(dt * (1_f64 - dt).exp(), weights[input_spike.input_id()]);
        }
        // for input_spike in self.input_spikes.iter() {
        //     let dt = (ft - input_spike.time).rem_euclid(period);
        //     expr.add_term(dt * (1_f64 - dt).exp(), weights[input_spike.input_id]);
        // }
        model
            .add_constr(
                format!("firing_time_{}", ft).as_str(),
                c!(expr == FIRING_THRESHOLD),
            )
            .map_err(|e| SNNError::GurobiError(e.to_string()))?;

        println!("New firing time constraints added at t={}", ft);
    }

    Ok(())
}

/// Returns the neuron potential at the given time, based on all its input spikes and their periodic extension.
/// The result only make sense if the contribution of a spike is negligible after the prescribed period (see POTENTIAL_TOLERANCE).
pub fn periodic_potential(
    t: f64,
    inputs: &Vec<Input>,
    input_spikes: &Vec<InputSpike>,
    period: f64,
) -> f64 {
    input_spikes.iter().fold(0.0, |acc, input_spike| {
        let dt = (t - input_spike.time()).rem_euclid(period);
        acc + dt * (1_f64 - dt).exp() * inputs[input_spike.input_id()].weight()
    })
}

/// Returns the maximum potential and the associated time in the prescribed interval.
/// The two following assumptions are made:
/// 1. The input spikes repeat themselfs with the provided period
/// 2. The contribution of an input spike on the neuron potential fades quickly away; after the provided period, its effect is negligible (see POTENTIAL_TOLERANCE).
/// The function returns None if the interval of interest is empty, i.e., start > end, or too long, i.e., end - start > period.
pub fn max_periodic_potential(
    inputs: &Vec<Input>,
    input_spikes: &Vec<InputSpike>,
    start: f64,
    end: f64,
    period: f64,
) -> Option<(f64, f64)> {
    if start > end {
        warn!("The provided interval is empty [{}, {}]", start, end);
        return None;
    }
    if end - start > period {
        warn!("The provided interval is too long [{}, {}]", start, end);
        return None;
    }

    if input_spikes.is_empty() {
        return Some((start, 0_f64));
    }

    // Init the global maximum and the associated time with the greatest of the two endpoints
    let (mut tmax, mut zmax) = (
        start,
        periodic_potential(start, inputs, input_spikes, period),
    );
    let tmp_zmax = periodic_potential(start, inputs, input_spikes, period);
    if tmp_zmax > zmax {
        (tmax, zmax) = (end, tmp_zmax);
    }

    if input_spikes.len() == 1 {
        let weight = inputs[input_spikes[0].input_id()].weight();
        let time =
            input_spikes[0].time() - ((input_spikes[0].time() - end) / period).ceil() * period;
        let tmp_tmax = match weight > 0.0 {
            true => time + 1.0,
            false => time,
        };
        if tmp_tmax < end && tmp_tmax > start {
            let tmp_zmax = periodic_potential(tmp_tmax, inputs, input_spikes, period);
            if tmp_zmax > zmax {
                (tmax, zmax) = (tmp_tmax, tmp_zmax);
            }
        }
        return Some((tmax, zmax));
    }

    (tmax, zmax) = input_spikes
        .iter()
        .circular_tuple_windows()
        .map(|(input_spike, next_input_spike)| {
            let time = input_spike.time() - ((input_spike.time() - end) / period).ceil() * period;
            let next_time = next_input_spike.time()
                - ((next_input_spike.time() - time) / period).floor() * period;
            let weight = inputs[input_spike.input_id()].weight();
            (weight, time, next_time)
        })
        .filter(|(_, _, next_time)| next_time >= &start)
        .map(|(weight, time, next_time)| {
            let t = match weight > 0.0 {
                true => {
                    let (a, b) = input_spikes
                        .iter()
                        .map(|tmp_input_spike| {
                            let tmp_weight = inputs[tmp_input_spike.input_id()].weight();
                            let tmp_time = tmp_input_spike.time()
                                - ((tmp_input_spike.time() - time) / period).ceil() * period;
                            (tmp_weight, tmp_time)
                        })
                        .fold((0.0, 0.0), |(acc_a, acc_b), (tmp_weight, tmp_time)| {
                            (
                                acc_a + tmp_weight * (tmp_time - time).exp(),
                                acc_b + tmp_weight * tmp_time * (tmp_time - time).exp(),
                            )
                        });
                    1.0 + b / a
                }
                false => time,
            };
            if t < next_time && t > start && t < end {
                (t, periodic_potential(t, inputs, input_spikes, period))
            } else {
                (f64::NAN, f64::NEG_INFINITY)
            }
        })
        .fold((tmax, zmax), |(acc_t, acc_z), (t, z)| {
            if z > acc_z {
                (t, z)
            } else {
                (acc_t, acc_z)
            }
        });

    Some((tmax, zmax))
}

/// Add maximum potential constraints at the provided times.
pub fn add_max_potential_constraint(
    model: &mut Model,
    weights: &Vec<Var>,
    input_spikes: &Vec<InputSpike>,
    times: &[f64],
    period: f64,
    max_level: f64,
) -> Result<(), SNNError> {
    for &t in times {
        let mut expr = grb::expr::LinExpr::new();
        for input_spike in input_spikes.iter() {
            let dt = (t - input_spike.time()).rem_euclid(period);
            expr.add_term(dt * (1_f64 - dt).exp(), weights[input_spike.input_id()]);
        }

        model
            .add_constr(
                format!("max_potential_{}", t).as_str(),
                c!(expr <= max_level),
            )
            .map_err(|e| SNNError::GurobiError(e.to_string()))?;

        debug!("New max potential constraint added for t={}", t);
    }

    Ok(())
}

/// Returns the neuron potential derivative at the given time, based on all its input spikes and their periodic extension.
/// The result only make sense if the contribution of a spike is negligible after the prescribed period (see POTENTIAL_TOLERANCE).
fn periodic_potential_derivative(
    t: f64,
    inputs: &Vec<Input>,
    input_spikes: &Vec<InputSpike>,
    period: f64,
) -> f64 {
    input_spikes.iter().fold(0.0, |acc, input_spike| {
        let dt = (t - input_spike.time()).rem_euclid(period);
        acc + (1_f64 - dt) * (1_f64 - dt).exp() * inputs[input_spike.input_id()].weight()
    })
}

/// Returns the maximum potential and the associated time in the prescribed interval.
/// The two following assumptions are made:
/// 1. The input spikes repeat themselfs with the provided period
/// 2. The contribution of an input spike on the neuron potential derivative fades quickly away; after the provided period, its effect is negligible (see POTENTIAL_TOLERANCE).
/// The function returns None if the interval of interest is empty, i.e., start > end, or too long, i.e., end - start > period.
pub fn min_periodic_potential_derivative(
    inputs: &Vec<Input>,
    input_spikes: &Vec<InputSpike>,
    start: f64,
    end: f64,
    period: f64,
) -> Option<(f64, f64)> {
    if start > end {
        warn!("The provided interval is empty [{}, {}]", start, end);
        return None;
    }
    if end - start > period {
        warn!("The provided interval is too long [{}, {}]", start, end);
        return None;
    }

    if input_spikes.is_empty() {
        return Some((start, 0_f64));
    }

    // Init the global minimum and the associated time with the lowest of the two endpoints
    let (mut tmin, mut zpmin) = (
        start,
        periodic_potential_derivative(start, inputs, input_spikes, period),
    );
    let tmp_zpmin = periodic_potential_derivative(end, inputs, input_spikes, period);
    if tmp_zpmin < zpmin {
        (tmin, zpmin) = (end, tmp_zpmin);
    }

    if input_spikes.len() == 1 {
        let weight = inputs[input_spikes[0].input_id()].weight();
        let time =
            input_spikes[0].time() - ((input_spikes[0].time() - end) / period).ceil() * period;
        let tmp_tmin = match weight > 0.0 {
            true => time + 2.0,
            false => time,
        };
        if tmp_tmin < end && tmp_tmin > start {
            let tmp_zpmin = periodic_potential_derivative(tmp_tmin, inputs, input_spikes, period);
            if tmp_zpmin < zpmin {
                (tmin, zpmin) = (tmp_tmin, tmp_zpmin);
            }
        }
        return Some((tmin, zpmin));
    }

    (tmin, zpmin) = input_spikes
        .iter()
        .circular_tuple_windows()
        .map(|(spike, next_spike)| {
            let time = spike.time() - ((spike.time() - end) / period).ceil() * period;
            let next_time =
                next_spike.time() - ((next_spike.time() - time) / period).floor() * period;
            let weight = inputs[spike.input_id()].weight();
            (weight, time, next_time)
        })
        .filter(|(_, _, next_time)| next_time >= &start)
        .map(|(weight, time, next_time)| {
            let t = match weight > 0.0 {
                true => {
                    let (a, b) = input_spikes
                        .iter()
                        .map(|tmp_spike| {
                            let tmp_weight = inputs[tmp_spike.input_id()].weight();
                            let tmp_time = tmp_spike.time()
                                - ((tmp_spike.time() - time) / period).ceil() * period;
                            (tmp_weight, tmp_time)
                        })
                        .fold((0.0, 0.0), |(acc_a, acc_b), (tmp_weight, tmp_time)| {
                            (
                                acc_a + tmp_weight * (tmp_time - time).exp(),
                                acc_b + tmp_weight * tmp_time * (tmp_time - time).exp(),
                            )
                        });
                    2.0 + b / a
                }
                false => time,
            };
            if t < next_time && t > start && t < end {
                (
                    t,
                    periodic_potential_derivative(t, inputs, input_spikes, period),
                )
            } else {
                (f64::NAN, f64::INFINITY)
            }
        })
        .fold((tmin, zpmin), |(acc_t, acc_z), (t, z)| {
            if z < acc_z {
                (t, z)
            } else {
                (acc_t, acc_z)
            }
        });

    Some((tmin, zpmin))
}

/// Add minimum potential derivative constraints at the provided times.
pub fn add_min_potential_derivative_constraints(
    model: &mut Model,
    weights: &Vec<Var>,
    input_spikes: &Vec<InputSpike>,
    times: &[f64],
    period: f64,
    min_slope: f64,
) -> Result<(), SNNError> {
    for &t in times {
        let mut expr = grb::expr::LinExpr::new();
        for input_spike in input_spikes.iter() {
            let dt = (t - input_spike.time()).rem_euclid(period);
            expr.add_term(
                (1_f64 - dt) * (1_f64 - dt).exp(),
                weights[input_spike.input_id()],
            );
        }

        model
            .add_constr(
                format!("min_potential_derivative_{}", t).as_str(),
                c!(expr >= min_slope),
            )
            .map_err(|e| SNNError::GurobiError(e.to_string()))?;

        debug!("New min potential derivative constraint added for t={}", t);
    }

    Ok(())
}

pub fn refine_constraints(
    model: &mut Model,
    weights: &Vec<Var>,
    firing_times: &Vec<f64>,
    inputs: &Vec<Input>,
    input_spikes: &Vec<InputSpike>,
    period: f64,
    max_level: f64,
    min_slope: f64,
    half_width: f64,
) -> Result<bool, SNNError> {
    let mut is_valid = true;

    // 1. Check for maximum level constraint
    if firing_times.is_empty() {
        if let Some((tmax, zmax)) =
            max_periodic_potential(inputs, input_spikes, 0.0, period, period)
        {
            if zmax > max_level + CONSTRAINT_TOLERANCE {
                add_max_potential_constraint(
                    model,
                    &weights,
                    input_spikes,
                    &[tmax],
                    period,
                    max_level,
                )?;
                is_valid = false;
            };
        }
    } else if firing_times.len() == 1 {
        let (ft, next_ft) = (firing_times[0], firing_times[0] + period);

        if let Some((tmax, zmax)) = max_periodic_potential(
            inputs,
            input_spikes,
            ft + REFRACTORY_PERIOD,
            next_ft - half_width,
            period,
        ) {
            if zmax > max_level + CONSTRAINT_TOLERANCE {
                add_max_potential_constraint(
                    model,
                    &weights,
                    input_spikes,
                    &[tmax],
                    period,
                    max_level,
                )?;
                is_valid = false;
            };
        };
    } else {
        for (&ft, &next_ft) in firing_times.iter().circular_tuple_windows() {
            let next_ft = next_ft - ((next_ft - ft) / period).floor() * period;
            if let Some((tmax, zmax)) = max_periodic_potential(
                inputs,
                input_spikes,
                ft + REFRACTORY_PERIOD,
                next_ft - half_width,
                period,
            ) {
                if zmax > max_level + CONSTRAINT_TOLERANCE {
                    add_max_potential_constraint(
                        model,
                        weights,
                        input_spikes,
                        &[tmax],
                        period,
                        max_level,
                    )?;
                    is_valid = false;
                };
            }
        }
    }

    // 2. Check for minimum slope constraint
    for &ft in firing_times.iter() {
        if let Some((tmin, zpmin)) = min_periodic_potential_derivative(
            inputs,
            input_spikes,
            ft - half_width,
            ft + half_width,
            period,
        ) {
            if zpmin < min_slope - CONSTRAINT_TOLERANCE {
                add_min_potential_derivative_constraints(
                    model,
                    weights,
                    input_spikes,
                    &[tmin],
                    period,
                    min_slope,
                )?;
                is_valid = false;
            };
        }
    }

    model
        .update()
        .map_err(|e| SNNError::GurobiError(e.to_string()))?;

    Ok(is_valid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_max_periodic_potential() {
        let inputs: Vec<Input> = vec![];
        let input_spikes: Vec<InputSpike> = vec![];

        assert_eq!(
            max_periodic_potential(&inputs, &input_spikes, 100.0, 0.0, 100.0),
            None
        );
        assert_eq!(
            max_periodic_potential(&inputs, &input_spikes, 0.0, 500.0, 100.0),
            None
        );

        // Without any input spike
        let inputs: Vec<Input> = vec![
            Input::new(0, 0, 1.0, 0.0),
            Input::new(1, 1, 1.0, 0.0),
            Input::new(2, 2, 1.0, 0.0),
            Input::new(3, 3, -1.0, 0.0),
        ];
        let input_spikes: Vec<InputSpike> = vec![];

        let (tmax, zmax) =
            max_periodic_potential(&inputs, &input_spikes, 0.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 0.0);
        assert_relative_eq!(zmax, 0.0);

        // With a single input spike
        let inputs: Vec<Input> = vec![
            Input::new(0, 0, 1.0, 0.0),
            Input::new(1, 1, 1.0, 0.0),
            Input::new(2, 2, 1.0, 0.0),
            Input::new(3, 3, -1.0, 0.0),
        ];
        let input_spikes: Vec<InputSpike> = vec![InputSpike::new(0, 1.0)];

        let (tmax, zmax) =
            max_periodic_potential(&inputs, &input_spikes, 0.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 2.0);
        assert_relative_eq!(zmax, 1.0);

        let (tmax, zmax) =
            max_periodic_potential(&inputs, &input_spikes, 2.5, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 2.5);
        assert_relative_eq!(zmax, 0.909795, epsilon = 1e-6);

        let (tmax, zmax) =
            max_periodic_potential(&inputs, &input_spikes, 70.0, 80.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 70.0);
        assert_relative_eq!(zmax, 0.0, epsilon = 1e-6);

        // With multiple input spikes
        let inputs: Vec<Input> = vec![
            Input::new(0, 0, 1.0, 0.0),
            Input::new(1, 1, 1.0, 0.0),
            Input::new(2, 2, 1.0, 0.0),
            Input::new(3, 3, -1.0, 0.0),
        ];
        let input_spikes: Vec<InputSpike> = vec![
            InputSpike::new(0, 1.0),
            InputSpike::new(1, 2.5),
            InputSpike::new(2, 4.0),
            InputSpike::new(3, 3.5),
        ];

        let (tmax, zmax) =
            max_periodic_potential(&inputs, &input_spikes, 0.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 3.226362, epsilon = 1e-6);
        assert_relative_eq!(zmax, 1.608097, epsilon = 1e-6);

        let (tmax, zmax) = max_periodic_potential(&inputs, &input_spikes, 2.0, 4.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 3.226362, epsilon = 1e-6);
        assert_relative_eq!(zmax, 1.608097, epsilon = 1e-6);

        let (tmax, zmax) =
            max_periodic_potential(&inputs, &input_spikes, 3.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 3.226362, epsilon = 1e-6);
        assert_relative_eq!(zmax, 1.608097, epsilon = 1e-6);

        let (tmax, zmax) =
            max_periodic_potential(&inputs, &input_spikes, 5.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 5.0);
        assert_relative_eq!(zmax, 0.847178, epsilon = 1e-6);

        let (tmax, zmax) =
            max_periodic_potential(&inputs, &input_spikes, 500.0, 550.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 503.226362, epsilon = 1e-6);
        assert_relative_eq!(zmax, 1.608097, epsilon = 1e-6);
    }

    #[test]
    fn test_min_periodic_potential_derivative() {
        let inputs: Vec<Input> = vec![];
        let input_spikes: Vec<InputSpike> = vec![];

        assert_eq!(
            max_periodic_potential(&inputs, &input_spikes, 100.0, 0.0, 100.0),
            None
        );
        assert_eq!(
            max_periodic_potential(&inputs, &input_spikes, 0.0, 500.0, 100.0),
            None
        );

        // Without any input spike
        let inputs: Vec<Input> = vec![
            Input::new(0, 0, 1.0, 0.0),
            Input::new(1, 1, 1.0, 0.0),
            Input::new(2, 2, 1.0, 0.0),
            Input::new(3, 3, -1.0, 0.0),
        ];
        let input_spikes: Vec<InputSpike> = vec![];

        let (tmax, zmax) =
            max_periodic_potential(&inputs, &input_spikes, 0.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 0.0);
        assert_relative_eq!(zmax, 0.0);

        // With a single input spike
        let inputs: Vec<Input> = vec![
            Input::new(0, 0, 1.0, 0.0),
            Input::new(1, 1, 1.0, 0.0),
            Input::new(2, 2, 1.0, 0.0),
            Input::new(3, 3, -1.0, 0.0),
        ];
        let input_spikes: Vec<InputSpike> = vec![InputSpike::new(0, 1.0)];

        let (tmin, zmin) =
            min_periodic_potential_derivative(&inputs, &input_spikes, 0.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmin, 3.0);
        assert_relative_eq!(zmin, -0.367879, epsilon = 1e-6);

        let (tmin, zmin) =
            min_periodic_potential_derivative(&inputs, &input_spikes, 3.5, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmin, 3.5);
        assert_relative_eq!(zmin, -0.334695, epsilon = 1e-6);

        let (tmin, zmin) =
            min_periodic_potential_derivative(&inputs, &input_spikes, 70.0, 80.0, 100.0).unwrap();
        assert_relative_eq!(tmin, 70.0);
        assert_relative_eq!(zmin, 0.0, epsilon = 1e-6);

        // With multiple input spikes
        let inputs: Vec<Input> = vec![
            Input::new(0, 0, 1.0, 0.0),
            Input::new(1, 1, 1.0, 0.0),
            Input::new(2, 2, 1.0, 0.0),
            Input::new(3, 3, -1.0, 0.0),
        ];
        let input_spikes: Vec<InputSpike> = vec![
            InputSpike::new(0, 1.0),
            InputSpike::new(1, 2.5),
            InputSpike::new(2, 4.0),
            InputSpike::new(3, 3.5),
        ];

        let (tmin, zmin) =
            min_periodic_potential_derivative(&inputs, &input_spikes, 0.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmin, 3.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.052977, epsilon = 1e-6);

        let (tmin, zmin) =
            min_periodic_potential_derivative(&inputs, &input_spikes, 2.0, 4.0, 100.0).unwrap();
        assert_relative_eq!(tmin, 3.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.052977, epsilon = 1e-6);

        let (tmin, zmin) =
            min_periodic_potential_derivative(&inputs, &input_spikes, 3.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmin, 3.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.052977, epsilon = 1e-6);

        let (tmin, zmin) =
            min_periodic_potential_derivative(&inputs, &input_spikes, 5.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmin, 5.728699, epsilon = 1e-6);
        assert_relative_eq!(zmin, -0.321555, epsilon = 1e-6);

        let (tmin, zmin) =
            min_periodic_potential_derivative(&inputs, &input_spikes, 500.0, 550.0, 100.0).unwrap();
        assert_relative_eq!(tmin, 503.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.052977, epsilon = 1e-6);
    }
}
