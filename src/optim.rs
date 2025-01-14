//! This module provides optimization utilities for spike trains memorization.
use grb::prelude::*;
use itertools::Itertools;
use log::debug;

use super::error::SNNError;
use super::signal::{
    max_periodic_potential, min_periodic_potential_derivative
};
use super::spike_train::InSpike;
use crate::{FIRING_THRESHOLD, REFRACTORY_PERIOD};

/// The tolerance for a constraint to be considered as satisfied (see Gurobi documentation)
/// See [FeasibilityTol](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#parameterfeasibilitytol)
pub const FEASIBILITY_TOL: f64 = 1e-9;

#[derive(Clone, Copy)]
pub enum Objective {
    /// Flat objective function, i.e., uniform prior.
    None,
    /// L2 norm, i.e., Gaussian prior.
    L2Norm,
    /// L1 norm, i.e., Laplace prior.
    L1Norm,
}

impl Objective {
    pub fn from_str(s: &str) -> Result<Self, SNNError> {
        match s {
            "none" => Ok(Objective::None),
            "l2norm" => Ok(Objective::L2Norm),
            "l1norm" => Ok(Objective::L1Norm),
            _ => Err(SNNError::InvalidParameters("Invalid objective".to_string())),
        }
    }
}

pub fn init_gurobi(name: &str, log_file: &str) -> Result<grb::Model, SNNError> {
    let mut env = Env::empty().map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    env.set(param::OutputFlag, 0)
        .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    env.set(param::UpdateMode, 1)
        .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    env.set(param::LogFile, log_file.to_string())
        .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    env.set(param::FeasibilityTol, FEASIBILITY_TOL)
        .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    let env: Env = env
        .start()
        .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    Model::with_env(name, env).map_err(|e| SNNError::OptimizationError(e.to_string()))
}

pub fn init_weights(
    model: &mut grb::Model,
    num_weights: usize,
    lim_weights: (f64, f64),
) -> Result<Vec<grb::Var>, SNNError> {
    let (min_weight, max_weight) = lim_weights;
    if !(min_weight <= max_weight) {
        return Err(SNNError::InvalidParameters(
            "Invalid weight limits".to_string(),
        ));
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
                .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
        }
        Objective::L1Norm => {
            let mut obj_expr = grb::expr::LinExpr::new();
            for (i, &var) in weights.iter().enumerate() {
                let slack = add_ctsvar!(model).unwrap();
                obj_expr.add_term(1.0, slack);
                model
                    .add_constr(format!("min_slack_{}", i).as_str(), c!(var >= -slack))
                    .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
                model
                    .add_constr(format!("max_slack_{}", i).as_str(), c!(var <= slack))
                    .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
            }
            model
                .set_objective(obj_expr, ModelSense::Minimize)
                .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
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
    inspikes: &Vec<InSpike>,
    period: f64,
) -> Result<(), SNNError> {
    for &time in firing_times {
        let mut expr = grb::expr::LinExpr::new();
        for inspike in inspikes.iter() {
            expr.add_term(inspike.periodic_kernel(time, period), weights[inspike.input_id()]);
        }
        model
            .add_constr(
                format!("firing_time_{}", time).as_str(),
                c!(expr == FIRING_THRESHOLD),
            )
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

        debug!("New firing time constraints added at time={}", time);
    }

    Ok(())
}

/// Add maximum potential constraints at the provided times.
pub fn add_max_potential_constraint(
    model: &mut Model,
    weights: &Vec<Var>,
    inspikes: &Vec<InSpike>,
    times: &[f64],
    period: f64,
    max_level: f64,
) -> Result<(), SNNError> {
    for &time in times {
        let mut expr = grb::expr::LinExpr::new();
        for inspike in inspikes.iter() {
            expr.add_term(inspike.periodic_kernel(time, period), weights[inspike.input_id()]);
        }

        model
            .add_constr(
                format!("max_potential_{}", time).as_str(),
                c!(expr <= max_level),
            )
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

        debug!("New max potential constraint added at time={}", time);
    }

    Ok(())
}

/// Add minimum potential derivative constraints at the provided times.
pub fn add_min_potential_derivative_constraints(
    model: &mut Model,
    weights: &Vec<Var>,
    inspikes: &Vec<InSpike>,
    times: &[f64],
    period: f64,
    min_slope: f64,
) -> Result<(), SNNError> {
    for &time in times {
        let mut expr = grb::expr::LinExpr::new();
        for inspike in inspikes.iter() {
            expr.add_term(inspike.periodic_kernel_derivative(time, period), weights[inspike.input_id()]);
        }

        model
            .add_constr(
                format!("min_potential_derivative_{}", time).as_str(),
                c!(expr >= min_slope),
            )
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

        debug!(
            "New min potential derivative constraint added at time={}",
            time
        );
    }

    Ok(())
}

pub fn refine_constraints(
    model: &mut Model,
    weights: &Vec<Var>,
    firing_times: &Vec<f64>,
    inspikes: &Vec<InSpike>,
    period: f64,
    max_level: f64,
    min_slope: f64,
    half_width: f64,
) -> Result<bool, SNNError> {
    let mut is_valid = true;

    // 1. Check for maximum level constraint
    if firing_times.is_empty() {
        if let Some((tmax, zmax)) = max_periodic_potential(inspikes, 0.0, period, period) {
            if zmax > max_level + FEASIBILITY_TOL {
                add_max_potential_constraint(
                    model,
                    &weights,
                    inspikes,
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
            inspikes,
            ft + REFRACTORY_PERIOD,
            next_ft - half_width,
            period,
        ) {
            if zmax > max_level + FEASIBILITY_TOL {
                add_max_potential_constraint(
                    model,
                    &weights,
                    inspikes,
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
                inspikes,
                ft + REFRACTORY_PERIOD,
                next_ft - half_width,
                period,
            ) {
                if zmax > max_level + FEASIBILITY_TOL {
                    add_max_potential_constraint(
                        model,
                        weights,
                        inspikes,
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
        if let Some((tmin, zpmin)) =
            min_periodic_potential_derivative(inspikes, ft - half_width, ft + half_width, period)
        {
            if zpmin < min_slope - FEASIBILITY_TOL {
                add_min_potential_derivative_constraints(
                    model,
                    weights,
                    inspikes,
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
        .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

    Ok(is_valid)
}
