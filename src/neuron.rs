//! This module provides `Neuron` and `Input` structures which are the core components of neural networks.

use core::f64;
use embed_doc_image::embed_doc_image;
use grb::prelude::*;
use itertools::Itertools;
use lambert_w::lambert_w0;
use log::{debug, error, info, trace, warn};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::error::SNNError;
use super::optim::*;
use super::spike_train::SpikeTrain;
use super::utils::{mean, median};

/// The minimum time between spikes. Can be seen as the default unit of time of a neuron.
pub const REFRACTORY_PERIOD: f64 = 1.0;
/// The nominal threshold for a neuron to fire.
pub const FIRING_THRESHOLD: f64 = 1.0;
/// The tolerance for a potential value to be considered negligible (relative to the number of inputs).
const POTENTIAL_TOLERANCE: f64 = 1e-9;

/// Represents an input to a neuron.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Input {
    // The ID of the input.
    id: usize,
    // The ID of the source neuron.
    source_id: usize,
    // The connection weight.
    weight: f64,
    // The connection delay.
    delay: f64,
}

impl Input {
    /// Returns the ID of neuron at the origin of the input.
    pub fn source_id(&self) -> usize {
        self.source_id
    }

    /// Returns the weight of the input.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Returns the delay of the input.
    pub fn delay(&self) -> f64 {
        self.delay
    }
}

/// Represents an input spike to a neuron, i.e., a time and a weight.
/// This structure is used for simulation purposes.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
struct InputSpike {
    // The input related to the spike.
    input_id: usize,
    // The time at which the spike was received.
    time: f64,
}
/// Represents a spiking neuron.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Neuron {
    // The neuron ID.
    id: usize,
    // The firing threshold of the neuron.
    threshold: f64,
    // A collection of (sorted) firing times of the neuron.
    firing_times: Vec<f64>,
    // A collection of inputs to the neuron (sorted by 1. source_id and 2. delay?)
    inputs: Vec<Input>,
    // A collection of input spikes to the neuron, sorted by time of arrival.
    input_spikes: Vec<InputSpike>,
}

impl Neuron {
    /// Create a new empty neuron with nominal threshold (see FIRING_THRESHOLD).
    pub fn new(id: usize) -> Self {
        Neuron {
            id,
            threshold: FIRING_THRESHOLD,
            firing_times: vec![],
            inputs: vec![],
            input_spikes: vec![],
        }
    }

    /// Returns the neuron potential at the given time, based on all its input spikes.
    /// When the potential exceeds the threshold, the neuron fires.
    ///
    /// ![A Foobaring][neuron]
    ///
    #[embed_doc_image("neuron", "images/neuron.svg")]
    pub fn potential(&self, t: f64) -> f64 {
        self.input_spikes
            .iter()
            .filter(|input_spike| input_spike.time < t)
            .fold(0.0, |acc, input_spike| {
                let dt = t - input_spike.time;
                acc + dt * (1_f64 - dt).exp() * self.inputs[input_spike.input_id].weight
            })
    }

    /// Returns the neuron potential at the given time, based on all its input spikes and their periodic extension.
    /// The result only make sense if the contribution of a spike is negligible after the prescribed period (see POTENTIAL_TOLERANCE).
    pub fn periodic_potential(&self, t: f64, period: f64) -> f64 {
        self.input_spikes.iter().fold(0.0, |acc, input_spike| {
            let dt = (t - input_spike.time).rem_euclid(period);
            acc + dt * (1_f64 - dt).exp() * self.inputs[input_spike.input_id].weight
        })
    }

    /// Returns the neuron potential derivative at the given time, based on all its input spikes and their periodic extension.
    /// The result only make sense if the contribution of a spike is negligible after the prescribed period (see POTENTIAL_TOLERANCE).
    pub fn periodic_potential_derivative(&self, t: f64, period: f64) -> f64 {
        self.input_spikes.iter().fold(0.0, |acc, input_spike| {
            let dt = (t - input_spike.time).rem_euclid(period);
            acc + (1_f64 - dt) * (1_f64 - dt).exp() * self.inputs[input_spike.input_id].weight
        })
    }

    /// Returns the neuron ID.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the neuron firing threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Set the neuron firing threshold.
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    /// Returns a slice of firing times of the neuron.
    pub fn firing_times(&self) -> &[f64] {
        &self.firing_times[..]
    }

    /// Returns a slice of inputs of the neuron.
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    /// Returns the number of firing times of the neuron.
    pub fn num_firing_times(&self) -> usize {
        self.firing_times.len()
    }

    /// Returns the number of inputs of the neuron.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Returns the total number of input spikes of the neuron.
    pub fn num_input_spikes(&self) -> usize {
        self.input_spikes.len()
    }

    /// Returns the number of inputs of the neuron from a given source ID.
    /// Reminder: inputs are sorted by source_id and delay.
    pub fn num_inputs_from(&self, source_id: usize) -> usize {
        self.inputs
            .iter()
            .filter(|input| input.source_id == source_id)
            .count()
    }

    // Returns the minimum delay path from a source neuron (specified by its ID) to the neuron.
    // The function returns None if there is no path from the source neuron to the neuron.
    pub fn min_delay_path(&self, source_id: usize) -> Option<f64> {
        if source_id == self.id {
            return Some(0.0);
        }
        self.inputs
            .iter()
            .filter(|input| input.source_id == source_id)
            .map(|input| input.delay)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Clear all neuron firing times.
    pub fn clear_firing_times(&mut self) {
        self.firing_times = vec![];
    }

    /// Add a firing time to the neuron.
    /// The function returns an error if the refractory period is violated.
    pub fn add_firing_time(&mut self, t: f64) -> Result<(), SNNError> {
        if let Some(&last) = self.firing_times.last() {
            if t < last + REFRACTORY_PERIOD {
                return Err(SNNError::RefractoryPeriodViolation { t1: last, t2: t });
            }
        }

        self.firing_times.push(t);
        Ok(())
    }

    /// Extend the neuron's firing times with new ones.
    /// If necessary, the provided firing times are sorted before being added.
    /// The function returns an error if the refractory period is violated.
    pub fn extend_firing_times(&mut self, firing_times: &[f64]) -> Result<(), SNNError> {
        for t in firing_times {
            if !t.is_finite() {
                return Err(SNNError::InvalidFiringTimes);
            }
        }

        let mut firing_times = firing_times.to_vec();
        firing_times.sort_by(|t1, t2| {
            t1.partial_cmp(t2).unwrap_or_else(|| {
                panic!("Comparison failed: NaN values should have been caught earlier")
            })
        });

        for ts in firing_times.windows(2) {
            if ts[1] - ts[0] < REFRACTORY_PERIOD {
                return Err(SNNError::RefractoryPeriodViolation {
                    t1: ts[0],
                    t2: ts[1],
                });
            }
        }

        if let (Some(&first), Some(&last)) = (firing_times.first(), self.firing_times.last()) {
            if first <= last + REFRACTORY_PERIOD {
                return Err(SNNError::RefractoryPeriodViolation {
                    t1: last,
                    t2: first,
                });
            }
        }

        self.firing_times.extend(firing_times);
        Ok(())
    }

    /// Make the neuron fire at the specified time and update the threshold.
    pub fn fires(&mut self, t: f64, threshold_noise: f64) -> Result<(), SNNError> {
        self.add_firing_time(t)?;
        self.threshold = FIRING_THRESHOLD + threshold_noise;
        Ok(())
    }

    /// Clear all neuron inputs.
    pub fn clear_inputs(&mut self) {
        self.inputs = vec![];
        self.input_spikes = vec![];
    }

    /// Add an input to the neuron, with the specified source ID, weight, and delay.
    pub fn add_input(&mut self, source_id: usize, weight: f64, delay: f64) {
        self.inputs.push(Input {
            id: self.inputs.len(),
            source_id,
            weight,
            delay,
        });
    }

    /// Clear all neuron input spikes.
    pub fn reset_input_spikes(&mut self) {
        self.input_spikes = vec![];
    }

    /// Add input spikes to the neuron from a given source ID firing at the specified time.
    pub fn add_input_spikes_for_source(&mut self, source_id: usize, firing_times: &[f64]) {
        let new_spikes: Vec<InputSpike> = self
            .inputs
            .iter()
            .filter(|input| input.source_id == source_id)
            .flat_map(|input| {
                firing_times.iter().map(|ft| InputSpike {
                    time: ft + input.delay,
                    input_id: input.id,
                })
            })
            .collect();

        self.merge_input_spikes(new_spikes);
    }

    /// Merge new input spikes with the existing ones and sort them by time of arrival.
    fn merge_input_spikes(&mut self, mut new_spikes: Vec<InputSpike>) {
        self.input_spikes.append(&mut new_spikes);
        self.input_spikes
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Returns the next firing time of the neuron, if any.
    pub fn next_spike(&self, mut start: f64) -> Option<f64> {
        if let Some(last) = self.firing_times().last() {
            start = last + REFRACTORY_PERIOD;
        }

        if self.potential(start) >= self.threshold {
            return Some(start);
        }

        let mut firing_time = f64::NAN;
        let pos = self
            .input_spikes
            .binary_search_by(|input_spike| {
                if input_spike.time >= start {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            })
            .unwrap_or_else(|pos| pos);

        for (i, input_spike) in self.input_spikes[pos..].iter().enumerate() {
            if firing_time <= input_spike.time {
                break;
            }

            let (a, b) =
                self.input_spikes[..=pos + i]
                    .iter()
                    .fold((0.0, 0.0), |(mut a, mut b), item| {
                        a += self.inputs[item.input_id].weight
                            * (item.time - input_spike.time).exp();
                        b += self.inputs[item.input_id].weight
                            * item.time
                            * (item.time - input_spike.time).exp();
                        (a, b)
                    });

            firing_time = match a == 0.0 {
                true => input_spike.time + 1.0 + (-b / self.threshold()).ln(),
                false => {
                    b / a
                        - lambert_w0(-self.threshold() / a * (b / a - 1.0 - input_spike.time).exp())
                }
            };

            if firing_time < input_spike.time {
                firing_time = f64::NAN;
            }
        }

        match firing_time.is_finite() {
            true => Some(firing_time),
            false => None,
        }
    }

    /// Returns the maximum potential and the associated time in the prescribed interval.
    /// The two following assumptions are made:
    /// 1. The input spikes repeat themselfs with the provided period
    /// 2. The contribution of an input spike on the neuron potential fades quickly away; after the provided period, its effect is negligible (see POTENTIAL_TOLERANCE).
    /// The function returns None if the interval of interest is empty, i.e., start > end, or too long, i.e., end - start > period.
    fn max_periodic_potential(&self, start: f64, end: f64, period: f64) -> Option<(f64, f64)> {
        if start > end {
            warn!("The provided interval is empty [{}, {}]", start, end);
            return None;
        }
        if end - start > period {
            warn!("The provided interval is too long [{}, {}]", start, end);
            return None;
        }

        if self.input_spikes.is_empty() {
            return Some((start, 0_f64));
        }

        // Init the global maximum and the associated time with the greatest of the two endpoints
        let (mut tmax, mut zmax) = (start, self.periodic_potential(start, period));
        let tmp_zmax = self.periodic_potential(end, period);
        if tmp_zmax > zmax {
            (tmax, zmax) = (end, tmp_zmax);
        }

        if self.num_input_spikes() == 1 {
            let weight = self.inputs[self.input_spikes[0].input_id].weight;
            let time = self.input_spikes[0].time
                - ((self.input_spikes[0].time - end) / period).ceil() * period;
            let tmp_tmax = match weight > 0.0 {
                true => time + 1.0,
                false => time,
            };
            if tmp_tmax < end && tmp_tmax > start {
                let tmp_zmax = self.periodic_potential(tmp_tmax, period);
                if tmp_zmax > zmax {
                    (tmax, zmax) = (tmp_tmax, tmp_zmax);
                }
            }
            return Some((tmax, zmax));
        }

        (tmax, zmax) = self
            .input_spikes
            .iter()
            .circular_tuple_windows()
            .map(|(spike, next_spike)| {
                let time = spike.time - ((spike.time - end) / period).ceil() * period;
                let next_time =
                    next_spike.time - ((next_spike.time - time) / period).floor() * period;
                let weight = self.inputs[spike.input_id].weight;
                (weight, time, next_time)
            })
            .filter(|(_, _, next_time)| next_time >= &start)
            .map(|(weight, time, next_time)| {
                let t = match weight > 0.0 {
                    true => {
                        let (a, b) = self
                            .input_spikes
                            .iter()
                            .map(|tmp_spike| {
                                let tmp_weight = self.inputs[tmp_spike.input_id].weight;
                                let tmp_time = tmp_spike.time
                                    - ((tmp_spike.time - time) / period).ceil() * period;
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
                    (t, self.periodic_potential(t, period))
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

    /// Returns the maximum potential and the associated time in the prescribed interval.
    /// The two following assumptions are made:
    /// 1. The input spikes repeat themselfs with the provided period
    /// 2. The contribution of an input spike on the neuron potential derivative fades quickly away; after the provided period, its effect is negligible (see POTENTIAL_TOLERANCE).
    /// The function returns None if the interval of interest is empty, i.e., start > end, or too long, i.e., end - start > period.
    fn min_periodic_potential_derivative(
        &self,
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

        if self.input_spikes.is_empty() {
            return Some((start, 0_f64));
        }

        // Init the global minimum and the associated time with the lowest of the two endpoints
        let (mut tmin, mut zpmin) = (start, self.periodic_potential_derivative(start, period));
        let tmp_zpmin = self.periodic_potential_derivative(end, period);
        if tmp_zpmin < zpmin {
            (tmin, zpmin) = (end, tmp_zpmin);
        }

        if self.num_input_spikes() == 1 {
            let weight = self.inputs[self.input_spikes[0].input_id].weight;
            let time = self.input_spikes[0].time
                - ((self.input_spikes[0].time - end) / period).ceil() * period;
            let tmp_tmin = match weight > 0.0 {
                true => time + 2.0,
                false => time,
            };
            if tmp_tmin < end && tmp_tmin > start {
                let tmp_zpmin = self.periodic_potential_derivative(tmp_tmin, period);
                if tmp_zpmin < zpmin {
                    (tmin, zpmin) = (tmp_tmin, tmp_zpmin);
                }
            }
            return Some((tmin, zpmin));
        }

        (tmin, zpmin) = self
            .input_spikes
            .iter()
            .circular_tuple_windows()
            .map(|(spike, next_spike)| {
                let time = spike.time - ((spike.time - end) / period).ceil() * period;
                let next_time =
                    next_spike.time - ((next_spike.time - time) / period).floor() * period;
                let weight = self.inputs[spike.input_id].weight;
                (weight, time, next_time)
            })
            .filter(|(_, _, next_time)| next_time >= &start)
            .map(|(weight, time, next_time)| {
                let t = match weight > 0.0 {
                    true => {
                        let (a, b) = self
                            .input_spikes
                            .iter()
                            .map(|tmp_spike| {
                                let tmp_weight = self.inputs[tmp_spike.input_id].weight;
                                let tmp_time = tmp_spike.time
                                    - ((tmp_spike.time - time) / period).ceil() * period;
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
                    (t, self.periodic_potential_derivative(t, period))
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

    /// Add threshold-crossing constraint at every firing time.
    fn add_firing_time_constraints(
        &self,
        model: &mut Model,
        weights: &Vec<Var>,
        period: f64,
    ) -> Result<(), SNNError> {
        for &ft in self.firing_times() {
            let mut expr = grb::expr::LinExpr::new();
            for input_spike in self.input_spikes.iter() {
                let dt = (ft - input_spike.time).rem_euclid(period);
                expr.add_term(dt * (1_f64 - dt).exp(), weights[input_spike.input_id]);
            }
            model
                .add_constr(
                    format!("firing_time_{}", ft).as_str(),
                    c!(expr == self.threshold()),
                )
                .map_err(|e| SNNError::GurobiError(e.to_string()))?;

            debug!("New firing time constraints added for t={}", ft);
        }

        Ok(())
    }

    /// Add maximum potential constraints at the provided times.
    fn add_max_potential_constraint(
        &self,
        model: &mut Model,
        weights: &Vec<Var>,
        times: &[f64],
        period: f64,
        max_level: f64,
    ) -> Result<(), SNNError> {
        for &t in times {
            let mut expr = grb::expr::LinExpr::new();
            for input_spike in self.input_spikes.iter() {
                let dt = (t - input_spike.time).rem_euclid(period);
                expr.add_term(dt * (1_f64 - dt).exp(), weights[input_spike.input_id]);
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

    /// Add minimum potential derivative constraints at the provided times.
    fn add_min_potential_derivative_constraints(
        &self,
        model: &mut Model,
        weights: &Vec<Var>,
        times: &[f64],
        period: f64,
        min_slope: f64,
    ) -> Result<(), SNNError> {
        for &t in times {
            let mut expr = grb::expr::LinExpr::new();
            for input_spike in self.input_spikes.iter() {
                let dt = (t - input_spike.time).rem_euclid(period);
                expr.add_term(
                    (1_f64 - dt) * (1_f64 - dt).exp(),
                    weights[input_spike.input_id],
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

    /// Compute the weights of the inputs that reproduce the neuron's firing times from the inputs.
    /// Memorization amounts to minimizing a convex objective function subject to linear constraints.
    /// No cost / uniform prior is mostly used for feasibility check.
    /// L2 cost / Gaussian prior yields low magnitude solutions.
    /// L1 cost / Laplace prior yields sparse solutions (c.f. [here](https://math.stackexchange.com/questions/1639716/how-can-l-1-norm-minimization-with-linear-equality-constraints-basis-pu))
    pub fn memorize_periodic_spike_trains(
        &mut self,
        spike_trains: &[SpikeTrain],
        period: f64,
        lim_weights: (f64, f64),
        max_level: f64,
        min_slope: f64,
        half_width: f64,
        objective: Objective,
    ) -> Result<(), SNNError> {
        if (self.num_inputs() as f64) * period * (1_f64 - period).exp() > POTENTIAL_TOLERANCE {
            return Err(SNNError::InvalidPeriod);
        }

        // Initialize the neuron firing times
        match spike_trains
            .iter()
            .find(|spike_train| spike_train.id() == self.id)
        {
            Some(spike_train) => self.firing_times = spike_train.firing_times().to_vec(),
            None => self.firing_times = vec![],
        };

        // Initialize input spikes for the neuron
        self.reset_input_spikes();
        for spike_train in spike_trains {
            self.add_input_spikes_for_source(spike_train.id(), spike_train.firing_times());
        }

        // Initialize the Gurobi environment and model for memorization
        let mut model = init_gurobi(format!("neuron_{}", self.id).as_str(), "gurobi.log")?;

        // Setup the decision variables, i.e., the weights
        let weights = init_vars(&mut model, self.num_inputs(), lim_weights)?;

        // Setup the objective function
        init_objective(&mut model, &weights, objective)?;

        // Setup the firing time constraints (and some additional maximum level and minimum slope constraints?)
        self.add_firing_time_constraints(&mut model, &weights, period)?;

        let mut is_valid = false;
        while !is_valid {
            // For fixed constraints, determine the optimal weights
            debug!("1. Optimize weights...");
            model
                .optimize()
                .map_err(|e| SNNError::GurobiError(e.to_string()))?;

            // If the model is infeasible, return an error
            let status = model
                .status()
                .map_err(|e| SNNError::GurobiError(e.to_string()))?;

            debug!("Status: {:?}", status);
            if Status::Optimal != status {
                return Err(SNNError::InfeasibleMemorization);
            }

            // Set the input weights to the optimal values
            for input in self.inputs.iter_mut() {
                input.weight = model
                    .get_obj_attr(grb::attribute::VarDoubleAttr::X, &weights[input.id])
                    .map_err(|e| SNNError::GurobiError(e.to_string()))?;
            }

            // Check if the current solution satifies all template constraints
            debug!("2. Refine constraints...");
            is_valid = true;

            // 1. Check for maximum level constraint
            if self.firing_times.is_empty() {
                if let Some((tmax, zmax)) = self.max_periodic_potential(0.0, period, period) {
                    if zmax > max_level + CONSTRAINT_TOLERANCE {
                        self.add_max_potential_constraint(
                            &mut model,
                            &weights,
                            &[tmax],
                            period,
                            max_level,
                        )?;
                        is_valid = false;
                    };
                }
            } else if self.num_firing_times() == 1 {
                let (ft, next_ft) = (self.firing_times[0], self.firing_times[0] + period);

                if let Some((tmax, zmax)) = self.max_periodic_potential(
                    ft + REFRACTORY_PERIOD,
                    next_ft - half_width,
                    period,
                ) {
                    if zmax > max_level + CONSTRAINT_TOLERANCE {
                        self.add_max_potential_constraint(
                            &mut model,
                            &weights,
                            &[tmax],
                            period,
                            max_level,
                        )?;
                        is_valid = false;
                    };
                };
            } else {
                for (&ft, &next_ft) in self.firing_times.iter().circular_tuple_windows() {
                    let next_ft = next_ft - ((next_ft - ft) / period).floor() * period;
                    if let Some((tmax, zmax)) = self.max_periodic_potential(
                        ft + REFRACTORY_PERIOD,
                        next_ft - half_width,
                        period,
                    ) {
                        if zmax > max_level + CONSTRAINT_TOLERANCE {
                            self.add_max_potential_constraint(
                                &mut model,
                                &weights,
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
            for &ft in self.firing_times.iter() {
                if let Some((tmin, zpmin)) =
                    self.min_periodic_potential_derivative(ft - half_width, ft + half_width, period)
                {
                    if zpmin < min_slope - CONSTRAINT_TOLERANCE {
                        self.add_min_potential_derivative_constraints(
                            &mut model,
                            &weights,
                            &[tmin],
                            period,
                            min_slope,
                        )?;
                        is_valid = false;
                    };
                }
            }
        }

        // Get the total number of constraints and the objective value
        let num_constrs = model
            .get_attr(grb::attribute::ModelIntAttr::NumConstrs)
            .map_err(|e| SNNError::GurobiError(e.to_string()))?;
        let obj_val = model
            .get_attr(grb::attribute::ModelDoubleAttr::ObjVal)
            .map_err(|e| SNNError::GurobiError(e.to_string()))?;

        info!(
            "Memorization done! \n\tObjective value: {} \n\tNumber of constraints: {}",
            obj_val, num_constrs
        );
        debug!(
            "Mean weight: {:?} | Median weight: {:?}",
            mean(self.inputs.iter().map(|input| input.weight)),
            median(self.inputs.iter().map(|input| input.weight))
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use core::panic;

    use super::*;

    #[test]
    fn test_neuron_constructor() {
        let neuron = Neuron::new(0);
        assert_eq!(
            neuron,
            Neuron {
                id: 0,
                threshold: FIRING_THRESHOLD,
                firing_times: vec![],
                inputs: vec![],
                input_spikes: vec![]
            }
        );
    }

    #[test]
    fn test_neuron_add_input() {
        let mut neuron = Neuron::new(0);

        neuron.add_input(0, 0.5, 0.5);
        neuron.add_input(1, 1.0, 1.0);
        neuron.add_input(0, -0.5, 2.0);
        neuron.add_input(3, 0.75, 1.5);
        neuron.add_input(1, 1.0, 0.25);

        assert_eq!(
            neuron.inputs(),
            &[
                Input {
                    id: 0,
                    source_id: 0,
                    weight: 0.5,
                    delay: 0.5
                },
                Input {
                    id: 1,
                    source_id: 1,
                    weight: 1.0,
                    delay: 1.0
                },
                Input {
                    id: 2,
                    source_id: 0,
                    weight: -0.5,
                    delay: 2.0
                },
                Input {
                    id: 3,
                    source_id: 3,
                    weight: 0.75,
                    delay: 1.5
                },
                Input {
                    id: 4,
                    source_id: 1,
                    weight: 1.0,
                    delay: 0.25
                },
            ]
        );
    }

    #[test]
    fn test_neuron_min_delay_path() {
        let mut neuron = Neuron::new(0);

        neuron.add_input(0, 0.5, 0.5);
        neuron.add_input(1, 1.0, 1.0);
        neuron.add_input(0, -0.5, 2.0);
        neuron.add_input(3, 0.75, 1.5);
        neuron.add_input(1, 1.0, 0.25);

        assert_eq!(neuron.min_delay_path(0), Some(0.0));
        assert_eq!(neuron.min_delay_path(1), Some(0.25));
        assert_eq!(neuron.min_delay_path(2), None);
        assert_eq!(neuron.min_delay_path(3), Some(1.5));
    }

    #[test]
    fn test_add_input_spikes_for_source() {
        let mut neuron = Neuron::new(0);

        neuron.add_input(0, 0.5, 1.0); // id: 0
        neuron.add_input(1, 0.75, 2.0); // id: 1
        neuron.add_input(0, -0.25, 1.5); // id: 2

        assert_eq!(neuron.num_inputs(), 3);

        neuron.add_input_spikes_for_source(0, &[1.0, 5.0]); // arrive at 2.0 (id:0), 6.0 (id:0), 2.5 (id:2), and 6.5 (id:2)
        neuron.add_input_spikes_for_source(1, &[0.5]); // arrive at 2.5 (id:1)

        assert_eq!(
            neuron.input_spikes,
            vec![
                InputSpike {
                    input_id: 0,
                    time: 2.0
                },
                InputSpike {
                    input_id: 2,
                    time: 2.5
                },
                InputSpike {
                    input_id: 1,
                    time: 2.5
                },
                InputSpike {
                    input_id: 0,
                    time: 6.0
                },
                InputSpike {
                    input_id: 2,
                    time: 6.5
                }
            ]
        );
    }

    #[test]
    fn test_neuron_extend_firing_times() {
        let mut neuron = Neuron::new(42);
        assert_eq!(neuron.extend_firing_times(&[0.0, 3.0, 7.0]), Ok(()));
        assert_eq!(neuron.firing_times, [0.0, 3.0, 7.0]);
        assert_eq!(
            neuron.extend_firing_times(&[6.0]),
            Err(SNNError::RefractoryPeriodViolation { t1: 7.0, t2: 6.0 })
        );
        assert_eq!(neuron.firing_times, [0.0, 3.0, 7.0]);
        assert_eq!(neuron.extend_firing_times(&[10.0, 12.0]), Ok(()));
        assert_eq!(neuron.firing_times, [0.0, 3.0, 7.0, 10.0, 12.0]);
    }

    #[test]
    fn add_neuron_add_firing_time() {
        let mut neuron = Neuron::new(42);
        assert_eq!(neuron.add_firing_time(0.0), Ok(()));
        assert_eq!(neuron.firing_times, [0.0]);
        assert_eq!(neuron.add_firing_time(7.0), Ok(()));
        assert_eq!(neuron.firing_times, [0.0, 7.0]);
        assert_eq!(
            neuron.add_firing_time(5.0),
            Err(SNNError::RefractoryPeriodViolation { t1: 7.0, t2: 5.0 })
        );
        assert_eq!(neuron.firing_times, [0.0, 7.0]);
    }

    #[test]
    fn test_neuron_fires() {
        let mut neuron = Neuron::new(42);
        assert_eq!(neuron.fires(0.0, 0.0), Ok(()));
        assert_eq!(neuron.firing_times, [0.0]);
        assert_eq!(neuron.threshold, FIRING_THRESHOLD);
        assert_eq!(neuron.fires(7.0, 0.25), Ok(()));
        assert_eq!(neuron.firing_times, [0.0, 7.0]);
        assert_eq!(neuron.threshold, FIRING_THRESHOLD + 0.25);
        assert_eq!(
            neuron.fires(5.0, 0.0),
            Err(SNNError::RefractoryPeriodViolation { t1: 7.0, t2: 5.0 })
        );
        assert_eq!(neuron.firing_times, [0.0, 7.0]);
        assert_eq!(neuron.threshold, FIRING_THRESHOLD + 0.25);
    }

    #[test]
    fn test_neuron_max_periodic_potential() {
        let mut neuron = Neuron::new(42);
        assert_eq!(neuron.max_periodic_potential(100.0, 0.0, 100.0), None);
        assert_eq!(neuron.max_periodic_potential(0.0, 200.0, 100.0), None);

        neuron.add_input(0, 1.0, 0.0);
        neuron.add_input(1, 1.0, 0.0);
        neuron.add_input(2, 1.0, 0.0);
        neuron.add_input(3, -1.0, 0.0);

        // Without any input spike
        let (tmax, zmax) = neuron.max_periodic_potential(0.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 0.0);
        assert_relative_eq!(zmax, 0.0);

        // With a single input spike
        neuron.add_input_spikes_for_source(0, &[1.0]);
        let (tmax, zmax) = neuron.max_periodic_potential(0.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 2.0);
        assert_relative_eq!(zmax, 1.0);
        let (tmax, zmax) = neuron.max_periodic_potential(2.5, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 2.5);
        assert_relative_eq!(zmax, 0.909795, epsilon = 1e-6);
        let (tmax, zmax) = neuron.max_periodic_potential(70.0, 80.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 70.0);
        assert_relative_eq!(zmax, 0.0, epsilon = 1e-6);

        // With multiple input spikes
        neuron.add_input_spikes_for_source(1, &[2.5]);
        neuron.add_input_spikes_for_source(2, &[4.0]);
        neuron.add_input_spikes_for_source(3, &[3.5]);
        let (tmax, zmax) = neuron.max_periodic_potential(0.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 3.226362, epsilon = 1e-6);
        assert_relative_eq!(zmax, 1.608097, epsilon = 1e-6);
        let (tmax, zmax) = neuron.max_periodic_potential(2.0, 4.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 3.226362, epsilon = 1e-6);
        assert_relative_eq!(zmax, 1.608097, epsilon = 1e-6);
        let (tmax, zmax) = neuron.max_periodic_potential(3.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 3.226362, epsilon = 1e-6);
        assert_relative_eq!(zmax, 1.608097, epsilon = 1e-6);
        let (tmax, zmax) = neuron.max_periodic_potential(5.0, 10.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 5.0);
        assert_relative_eq!(zmax, 0.847178, epsilon = 1e-6);
        let (tmax, zmax) = neuron.max_periodic_potential(500.0, 550.0, 100.0).unwrap();
        assert_relative_eq!(tmax, 503.226362, epsilon = 1e-6);
        assert_relative_eq!(zmax, 1.608097, epsilon = 1e-6);
    }

    #[test]
    fn test_neuron_min_periodic_potential_derivative() {
        let mut neuron = Neuron::new(42);
        assert_eq!(
            neuron.min_periodic_potential_derivative(100.0, 0.0, 100.0),
            None
        );
        assert_eq!(
            neuron.min_periodic_potential_derivative(0.0, 200.0, 100.0),
            None
        );

        neuron.add_input(0, 1.0, 0.0);
        neuron.add_input(1, 1.0, 0.0);
        neuron.add_input(2, 1.0, 0.0);
        neuron.add_input(3, -1.0, 0.0);

        // Without any input spike
        let (tmin, zmin) = neuron
            .min_periodic_potential_derivative(0.0, 10.0, 100.0)
            .unwrap();
        assert_relative_eq!(tmin, 0.0);
        assert_relative_eq!(zmin, 0.0);

        // With a single input spike
        neuron.add_input_spikes_for_source(0, &[1.0]);
        let (tmin, zmin) = neuron
            .min_periodic_potential_derivative(0.0, 10.0, 100.0)
            .unwrap();
        assert_relative_eq!(tmin, 3.0);
        assert_relative_eq!(zmin, -0.367879, epsilon = 1e-6);
        let (tmin, zmin) = neuron
            .min_periodic_potential_derivative(3.5, 10.0, 100.0)
            .unwrap();
        assert_relative_eq!(tmin, 3.5);
        assert_relative_eq!(zmin, -0.334695, epsilon = 1e-6);
        let (tmin, zmin) = neuron
            .min_periodic_potential_derivative(70.0, 80.0, 100.0)
            .unwrap();
        assert_relative_eq!(tmin, 70.0);
        assert_relative_eq!(zmin, 0.0, epsilon = 1e-6);

        // With multiple input spikes
        neuron.add_input_spikes_for_source(1, &[2.5]);
        neuron.add_input_spikes_for_source(2, &[4.0]);
        neuron.add_input_spikes_for_source(3, &[3.5]);
        let (tmin, zmin) = neuron
            .min_periodic_potential_derivative(0.0, 10.0, 100.0)
            .unwrap();
        assert_relative_eq!(tmin, 3.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.052977, epsilon = 1e-6);
        let (tmin, zmin) = neuron
            .min_periodic_potential_derivative(2.0, 4.0, 100.0)
            .unwrap();
        assert_relative_eq!(tmin, 3.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.052977, epsilon = 1e-6);
        let (tmin, zmin) = neuron
            .min_periodic_potential_derivative(3.0, 10.0, 100.0)
            .unwrap();
        assert_relative_eq!(tmin, 3.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.052977, epsilon = 1e-6);
        let (tmin, zmin) = neuron
            .min_periodic_potential_derivative(5.0, 10.0, 100.0)
            .unwrap();
        assert_relative_eq!(tmin, 5.728699, epsilon = 1e-6);
        assert_relative_eq!(zmin, -0.321555, epsilon = 1e-6);
        let (tmin, zmin) = neuron
            .min_periodic_potential_derivative(500.0, 550.0, 100.0)
            .unwrap();
        assert_relative_eq!(tmin, 503.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.052977, epsilon = 1e-6);
    }

    #[test]
    fn test_neuron_potential() {
        let neuron = Neuron::new(42);
        assert_eq!(neuron.potential(0.0), 0.0);
        assert_eq!(neuron.potential(1.0), 0.0);
        assert_eq!(neuron.potential(2.0), 0.0);

        let mut neuron = Neuron::new(42);
        neuron.add_input(0, 1.0, 1.0);
        neuron.add_input_spikes_for_source(0, &[0.0]);
        assert_eq!(neuron.potential(0.0), 0.0);
        assert_eq!(neuron.potential(1.0), 0.0);
        assert_eq!(neuron.potential(2.0), 1.0);

        let mut neuron = Neuron::new(42);
        neuron.add_input(0, 1.0, 1.0);
        neuron.add_input(1, -1.0, 1.0);
        neuron.add_input_spikes_for_source(0, &[0.0, 1.0]);
        neuron.add_input_spikes_for_source(1, &[0.0]);
        assert_eq!(neuron.potential(0.0), 0.0);
        assert_eq!(neuron.potential(1.0), 0.0);
        assert_eq!(neuron.potential(2.0), 0.0);
        assert_eq!(neuron.potential(3.0), 1.0);
    }

    #[test]
    fn test_neuron_next_spike() {
        // 1 input / 1 input spike producing a spike
        let mut neuron = Neuron::new(42);
        neuron.add_input(0, 1.0, 1.0);
        neuron.add_input_spikes_for_source(0, &[0.0]);
        assert_eq!(neuron.next_spike(0.0), Some(2.0));

        // 2 inputs / 2 input spikes canceling each other
        let mut neuron = Neuron::new(42);
        neuron.add_input(0, 1.0, 1.0);
        neuron.add_input(1, -1.0, 0.5);
        neuron.add_input_spikes_for_source(0, &[0.0]);
        neuron.add_input_spikes_for_source(1, &[0.5]);
        assert_eq!(neuron.next_spike(0.0), None);

        // 2 inputs / 4 input spikes producing a spike
        let mut neuron = Neuron::new(42);
        neuron.add_input(0, 1.0, 1.0);
        neuron.add_input(1, -0.25, 0.5);
        neuron.add_input_spikes_for_source(0, &[0.0, 2.0, 3.0]);
        neuron.add_input_spikes_for_source(1, &[1.0]);
        assert_eq!(neuron.next_spike(0.0), Some(3.2757576038986502));

        // 1 input / 1 input spike producing a spike after refractory period
        let mut neuron = Neuron::new(42);
        neuron.add_firing_time(2.0).unwrap();
        neuron.add_input(0, 10.0, 1.0);
        neuron.add_input_spikes_for_source(0, &[0.0]);
        assert_eq!(neuron.next_spike(0.0), Some(3.0));

        // 1 input with zero-weight / many input spikes producing no spike
        let mut neuron = Neuron::new(42);
        neuron.add_input(0, 0.0, 1.0);
        neuron.add_input_spikes_for_source(0, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(neuron.next_spike(0.0), None);

        // no input / no input spike producing a spike because of zero firing threshold
        let mut neuron = Neuron::new(42);
        neuron.set_threshold(0.0);
        assert_eq!(neuron.next_spike(0.0), Some(0.0));

        // 1 input with large weight / many input spikes producing no spike because of extreme firing threshold
        let mut neuron = Neuron::new(42);
        neuron.set_threshold(1_000_000_f64);
        neuron.add_input(0, 100.0, 1.0);
        neuron.add_input_spikes_for_source(0, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(neuron.next_spike(0.0), None);
    }

    #[test]
    fn test_memorize_empty_periodic_spike_train() {
        let period = 100.0;
        let lim_weights = (-1.0, 1.0);
        let max_level = 0.0;
        let min_slope = 0.0;
        let half_width = 0.5;
        let objective = Objective::None;

        let spike_trains = vec![
            SpikeTrain::build(1, &[1.0, 3.0, 25.0]).unwrap(),
            SpikeTrain::build(2, &[5.0, 77.0, 89.0]).unwrap(),
        ];

        let mut neuron = Neuron::new(0);
        neuron.add_input(0, 1.0, 1.0);
        neuron.add_input(1, 1.0, 2.0);
        neuron.add_input(2, 1.0, 0.5);
        neuron.add_input(1, 1.0, 5.0);

        assert_eq!(
            neuron.memorize_periodic_spike_trains(
                &spike_trains,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                objective
            ),
            Ok(())
        );

        assert_eq!(neuron.inputs[0].weight, 0.0);
        assert_eq!(neuron.inputs[1].weight, 0.0);
        assert_eq!(neuron.inputs[2].weight, 0.0);
        assert_eq!(neuron.inputs[3].weight, 0.0);
    }

    #[test]
    fn test_memorize_single_spike_periodic_spike_train() {
        let period = 100.0;
        let lim_weights = (-5.0, 5.0);
        let max_level = 0.5;
        let min_slope = 0.0;
        let half_width = 0.25;

        let spike_trains = vec![
            SpikeTrain::build(0, &[1.55]).unwrap(),
            SpikeTrain::build(1, &[1.0]).unwrap(),
            SpikeTrain::build(2, &[1.5]).unwrap(),
            SpikeTrain::build(3, &[2.0]).unwrap(),
            SpikeTrain::build(4, &[3.5]).unwrap(),
        ];

        let mut neuron = Neuron::new(0);
        neuron.add_input(0, 1.0, 0.0);
        neuron.add_input(1, 1.0, 0.0);
        neuron.add_input(2, 1.0, 0.0);
        neuron.add_input(3, 1.0, 0.0);
        neuron.add_input(4, 1.0, 0.0);

        neuron
            .memorize_periodic_spike_trains(
                &spike_trains,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                Objective::None,
            )
            .unwrap();
        println!("inputs: {:?}", neuron.inputs);
        assert_relative_eq!(neuron.inputs[0].weight, -2.0920029, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[1].weight, 0.82764217, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[2].weight, 2.2129265, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[3].weight, -5.0, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[4].weight, -5.0, epsilon = 1e-6);

        neuron
            .memorize_periodic_spike_trains(
                &spike_trains,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                Objective::L2Norm,
            )
            .unwrap();
        println!("inputs: {:?}", neuron.inputs);
        assert_relative_eq!(neuron.inputs[0].weight, -1.405015, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[1].weight, 0.827642, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[2].weight, 2.212927, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[3].weight, -1.211926, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[4].weight, 0.0, epsilon = 1e-6);

        neuron
            .memorize_periodic_spike_trains(
                &spike_trains,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                Objective::L1Norm,
            )
            .unwrap();
        println!("inputs: {:?}", neuron.inputs);
        assert_relative_eq!(neuron.inputs[0].weight, -1.405015, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[1].weight, 0.827642, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[2].weight, 2.212927, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[3].weight, -1.211926, epsilon = 1e-6);
        assert_relative_eq!(neuron.inputs[4].weight, 0.0, epsilon = 1e-6);
    }
}
