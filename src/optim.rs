//! This module provides optimization utilities for spike trains memorization.
use grb::prelude::*;

use crate::error::SNNError;

/// The tolerance for a constraint to be considered as satisfied
pub const CONSTRAINT_TOLERANCE: f64 = 1e-6;

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

pub fn init_vars(
    model: &mut grb::Model,
    num_vars: usize,
    lim_vars: (f64, f64),
) -> Result<Vec<grb::Var>, SNNError> {
    let (min_var, max_var) = lim_vars;
    if !(min_var <= max_var) {
        return Err(SNNError::InvalidWeight);
    }

    let vars = match (min_var.is_infinite(), max_var.is_infinite()) {
        (false, false) => (0..num_vars)
            .map(|_| add_ctsvar!(model, bounds: min_var..max_var).unwrap())
            .collect::<Vec<Var>>(),
        (true, false) => (0..num_vars)
            .map(|_| add_ctsvar!(model, bounds: ..max_var).unwrap())
            .collect::<Vec<Var>>(),
        (false, true) => (0..num_vars)
            .map(|_| add_ctsvar!(model, bounds: min_var..).unwrap())
            .collect::<Vec<Var>>(),
        (true, true) => (0..num_vars)
            .map(|_| add_ctsvar!(model, bounds: ..).unwrap())
            .collect::<Vec<Var>>(),
    };
    Ok(vars)
}

pub fn init_objective(
    model: &mut grb::Model,
    vars: &Vec<grb::Var>,
    objective: Objective,
) -> Result<(), SNNError> {
    match objective {
        Objective::None => (),
        Objective::L2Norm => {
            let mut obj_expr = grb::expr::QuadExpr::new();
            for &var in vars.iter() {
                obj_expr.add_qterm(1.0, var, var);
            }
            model
                .set_objective(obj_expr, ModelSense::Minimize)
                .map_err(|e| SNNError::GurobiError(e.to_string()))?;
        }
        Objective::L1Norm => {
            let mut obj_expr = grb::expr::LinExpr::new();
            for (i, &var) in vars.iter().enumerate() {
                let slack = add_ctsvar!(model).unwrap();
                obj_expr.add_term(1.0, slack);
                model
                    .add_constrs(vec![
                        (&format!("min_slack_{}", i).as_str(), c!(var >= -slack)),
                        (&format!("max_slack_{}", i).as_str(), c!(var <= slack)),
                    ])
                    .map_err(|e| SNNError::GurobiError(e.to_string()))?;
            }
            model
                .set_objective(obj_expr, ModelSense::Minimize)
                .map_err(|e| SNNError::GurobiError(e.to_string()))?;
        }
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
}
