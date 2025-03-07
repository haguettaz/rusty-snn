//! Optimization-related utilities. For further details, check the [Gurobi documentation](https://docs.gurobi.com/).
use grb::prelude::*;
use itertools::Itertools;

use crate::core::utils::{TimeInterval, TimeIntervalUnion};
use crate::core::REFRACTORY_PERIOD;
use crate::error::SNNError;

/// The tolerance for a constraint to be considered as satisfied by the Gurobi solver, c.f. [FeasibilityTol](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#parameterfeasibilitytol)
pub const GRB_FEASIBILITY_TOL: f64 = 1e-6;
/// The tolerance for a constraint to be considered as satisfied during the refinement process.
pub const REF_FEASIBILITY_TOL: f64 = 1e-5;
/// The maximum number of new constraints to be added to the model during the refinement.
pub const MAX_NEW_CSTRS: usize = 1;
/// The maximum number of iterations for the memorization process.
pub const MAX_ITER: usize = 1_000;

/// The objective function for the optimization problem.
#[derive(Clone, Copy)]
pub enum Objective {
    /// No regularization.
    None,
    /// 0-norm promoting sparse weights (minimize the number of non-zero weights).
    L0,
    /// 1-norm (Laplace) promoting sparse weights.
    L1,
    /// 2-norm (Gauss) promoting low-magnitude weights.
    L2,
    /// infinity-norm promoting low-magnitude weights (minimize the largest weight magnitude).
    LInfinity,
}

impl Objective {
    /// Returns the objective function from a string.
    pub fn from_str(s: &str) -> Result<Self, SNNError> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Objective::None),
            "l0" => Ok(Objective::L0),
            "l1" => Ok(Objective::L1),
            "l2" => Ok(Objective::L2),
            "linfinity" => Ok(Objective::LInfinity),
            _ => Err(SNNError::InvalidParameter("Invalid objective".to_string())),
        }
    }
}

/// The decomposition of the timeline into relevant intervals.
#[derive(PartialEq, Debug, Clone)]
pub struct TimeTemplate {
    // The times at which the neuron fires.
    pub ftimes: Vec<f64>,
    // The times at which the neuron is silent.
    pub silence_regions: TimeIntervalUnion,
    // The times at which the neuron is active, i.e., is about to fire.
    pub active_regions: TimeIntervalUnion,
    // The time interval of the template.
    pub interval: TimeInterval,
}

impl TimeTemplate {
    /// Creates a new time template from a sorted list of firing times.
    ///
    /// This method constructs a time template containing:
    /// - A list of firing times within the period
    /// - Silence regions where no firing occurs
    /// - Active regions around each firing time
    pub fn new_from(ftimes: &Vec<f64>, half_width: f64, interval: TimeInterval) -> Self {
        match interval {
            TimeInterval::Empty => TimeTemplate {
                ftimes: vec![],
                silence_regions: TimeIntervalUnion::Empty,
                active_regions: TimeIntervalUnion::Empty,
                interval: TimeInterval::Empty,
            },
            TimeInterval::Closed { start, end } => {
                if ftimes.is_empty() {
                    TimeTemplate {
                        ftimes: vec![],
                        silence_regions: TimeIntervalUnion::new_from(vec![interval.clone()]),
                        active_regions: TimeIntervalUnion::Empty,
                        interval: interval.clone(),
                    }
                } else {
                    let mut silence_regions: Vec<TimeInterval> = vec![];

                    let new_interval =
                        TimeInterval::new(start, ftimes.first().unwrap() - half_width);
                    let intersect = interval.intersect(new_interval);
                    if !intersect.is_empty() {
                        silence_regions.push(intersect);
                    }

                    silence_regions.extend(
                        ftimes
                            .iter()
                            .tuple_windows()
                            .filter_map(|(time, next_time)| {
                                let new_interval = TimeInterval::new(
                                    *time + REFRACTORY_PERIOD,
                                    *next_time - half_width,
                                );
                                match interval.intersect(new_interval) {
                                    TimeInterval::Empty => None,
                                    intersect => Some(intersect),
                                }
                            })
                            .collect::<Vec<TimeInterval>>(),
                    );

                    let new_interval =
                        TimeInterval::new(ftimes.last().unwrap() + REFRACTORY_PERIOD, end);
                    let intersect = interval.intersect(new_interval);
                    if !intersect.is_empty() {
                        silence_regions.push(intersect);
                    }

                    let active_regions = ftimes
                        .iter()
                        .filter_map(|time| {
                            let new_interval =
                                TimeInterval::new(time - half_width, time + half_width);
                            match interval.intersect(new_interval) {
                                TimeInterval::Empty => None,
                                intersect => Some(intersect),
                            }
                        })
                        .collect::<Vec<TimeInterval>>();

                    let mut ftimes = ftimes.clone();
                    ftimes.retain(|time| interval.contains(*time));

                    TimeTemplate {
                        ftimes,
                        silence_regions: TimeIntervalUnion::new_from(silence_regions),
                        active_regions: TimeIntervalUnion::new_from(active_regions),
                        interval,
                    }
                }
            }
        }
    }

    /// Creates a new cyclic time template from a sorted list of firing times.
    ///
    /// This method constructs a time template that repeats with the given period, containing:
    /// - A list of firing times within the period
    /// - Silence regions where no firing occurs
    /// - Active regions around each firing time
    pub fn new_cyclic_from(ftimes: &[f64], half_width: f64, period: f64) -> Self {
        if ftimes.is_empty() {
            TimeTemplate {
                ftimes: vec![],
                silence_regions: TimeIntervalUnion::new_from(vec![TimeInterval::new(0.0, period)]),
                active_regions: TimeIntervalUnion::Empty,
                interval: TimeInterval::new(0.0, period),
            }
        } else {
            let before_first = ftimes[0] - half_width;
            let interval = TimeInterval::new(before_first, before_first + period);

            let mut silence_regions = ftimes
                .iter()
                .tuple_windows()
                .filter_map(|(time, next_time)| {
                    let new_interval =
                        TimeInterval::new(*time + REFRACTORY_PERIOD, *next_time - half_width);
                    match interval.intersect(new_interval) {
                        TimeInterval::Empty => None,
                        intersect => Some(intersect),
                    }
                })
                .collect::<Vec<TimeInterval>>();

            let new_interval = TimeInterval::new(
                ftimes.last().unwrap() + REFRACTORY_PERIOD,
                before_first + period,
            );
            let intersect = interval.intersect(new_interval);
            if !intersect.is_empty() {
                silence_regions.push(intersect);
            }

            let active_regions = ftimes
                .iter()
                .map(|time| TimeInterval::new(time - half_width, time + half_width))
                .collect::<Vec<TimeInterval>>();

            TimeTemplate {
                ftimes: ftimes.to_vec(),
                silence_regions: TimeIntervalUnion::new_from(silence_regions),
                active_regions: TimeIntervalUnion::new_from(active_regions),
                interval,
            }
        }
    }
}

pub fn set_grb_model(name: &str, log_file: &str) -> Result<grb::Model, SNNError> {
    let mut env = Env::empty().map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    env.set(param::OutputFlag, 0)
        .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    env.set(param::UpdateMode, 1)
        .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    env.set(param::LogFile, log_file.to_string())
        .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    env.set(param::FeasibilityTol, GRB_FEASIBILITY_TOL)
        .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    let env: Env = env
        .start()
        .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
    Model::with_env(name, env).map_err(|e| SNNError::OptimizationError(e.to_string()))
}

pub fn set_grb_vars(
    model: &mut grb::Model,
    num_weights: usize,
    lim_weights: (f64, f64),
) -> Result<Vec<grb::Var>, SNNError> {
    let (min_weight, max_weight) = lim_weights;
    if !(min_weight <= max_weight) {
        return Err(SNNError::InvalidParameter(
            "Invalid weight limits".to_string(),
        ));
    }

    let weights = match (min_weight.is_infinite(), max_weight.is_infinite()) {
        (false, false) => {
            let mut vars = Vec::with_capacity(num_weights);
            for _ in 0..num_weights {
                let var = add_ctsvar!(model, bounds: min_weight..max_weight)
                    .map_err(|e| SNNError::InvalidOperation(e.to_string()))?;
                vars.push(var);
            }
            vars
        }
        (true, false) => {
            let mut vars = Vec::with_capacity(num_weights);
            for _ in 0..num_weights {
                let var = add_ctsvar!(model, bounds: ..max_weight)
                    .map_err(|e| SNNError::InvalidOperation(e.to_string()))?;
                vars.push(var);
            }
            vars
        }
        (false, true) => {
            let mut vars = Vec::with_capacity(num_weights);
            for _ in 0..num_weights {
                let var = add_ctsvar!(model, bounds: min_weight..)
                    .map_err(|e| SNNError::InvalidOperation(e.to_string()))?;
                vars.push(var);
            }
            vars
        }
        (true, true) => {
            let mut vars = Vec::with_capacity(num_weights);
            for _ in 0..num_weights {
                let var = add_ctsvar!(model, bounds: ..)
                    .map_err(|e| SNNError::InvalidOperation(e.to_string()))?;
                vars.push(var);
            }
            vars
        }
    };

    Ok(weights)
}

pub fn set_grb_objective(
    model: &mut grb::Model,
    weights: &Vec<grb::Var>,
    objective: Objective,
) -> Result<(), SNNError> {
    match objective {
        Objective::None => (),
        Objective::L0 => Err(SNNError::NotImplemented(
            "L0 regularization is not implemented yet".to_string(),
        ))?,

        Objective::L1 => {
            let mut obj_expr = grb::expr::LinExpr::new();
            for (i, &weight) in weights.iter().enumerate() {
                let slack = add_ctsvar!(model).unwrap();
                obj_expr.add_term(1.0, slack);

                model
                    .add_constr(format!("min_slack_{}", i).as_str(), c!(weight >= -slack))
                    .map_err(|e| SNNError::InvalidOperation(e.to_string()))?;

                model
                    .add_constr(format!("max_slack_{}", i).as_str(), c!(weight <= slack))
                    .map_err(|e| SNNError::InvalidOperation(e.to_string()))?;
            }
            model
                .set_objective(obj_expr, Minimize)
                .map_err(|e| SNNError::InvalidOperation(e.to_string()))?;
        }
        Objective::L2 => {
            let mut obj_expr = grb::expr::QuadExpr::new();
            for &weight in weights.iter() {
                obj_expr.add_qterm(1.0, weight, weight);
            }
            model
                .set_objective(obj_expr, Minimize)
                .map_err(|e| SNNError::InvalidOperation(e.to_string()))?;
        }
        Objective::LInfinity => {
            let mut obj_expr = grb::expr::LinExpr::new();
            let lim_weight = add_ctsvar!(model).unwrap();
            obj_expr.add_term(1.0, lim_weight);
            model
                .set_objective(obj_expr, Minimize)
                .map_err(|e| SNNError::InvalidOperation(e.to_string()))?;

            for (i, &weight) in weights.iter().enumerate() {
                model
                    .add_constr(format!("min_lim_weight_{}", i).as_str(), c!(weight >= -lim_weight))
                    .map_err(|e| SNNError::InvalidOperation(e.to_string()))?;
                model
                    .add_constr(format!("max_lim_weight_{}", i).as_str(), c!(weight <= lim_weight))
                    .map_err(|e| SNNError::InvalidOperation(e.to_string()))?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test objective from string
    #[test]
    fn test_objective_from_str() {
        assert!(matches!(Objective::from_str("none"), Ok(Objective::None)));
        assert!(matches!(Objective::from_str("l0"), Ok(Objective::L0)));
        assert!(matches!(Objective::from_str("l1"), Ok(Objective::L1)));
        assert!(matches!(Objective::from_str("l2"), Ok(Objective::L2)));
        assert!(matches!(
            Objective::from_str("linfinity"),
            Ok(Objective::LInfinity)
        ));
        assert!(matches!(
            Objective::from_str("invalid"),
            Err(SNNError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_time_template_new_from() {
        let time_template = TimeTemplate::new_from(&vec![], 0.25, TimeInterval::new(0.0, 8.0));
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![TimeInterval::new(0.0, 8.0)])
        );
        assert_eq!(time_template.active_regions, TimeIntervalUnion::Empty);

        let time_template = TimeTemplate::new_from(&vec![0.0], 0.25, TimeInterval::new(0.0, 8.0));
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![TimeInterval::new(1.0, 8.0)])
        );
        assert_eq!(
            time_template.active_regions,
            TimeIntervalUnion::new_from(vec![TimeInterval::new(0.0, 0.25)])
        );

        let time_template = TimeTemplate::new_from(&vec![1.0], 0.25, TimeInterval::new(0.0, 8.0));
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(0.0, 0.75),
                TimeInterval::new(2.0, 8.0)
            ])
        );
        assert_eq!(
            time_template.active_regions,
            TimeIntervalUnion::new_from(vec![TimeInterval::new(0.75, 1.25)])
        );

        let time_template =
            TimeTemplate::new_from(&vec![0.0, 2.0, 6.0], 0.25, TimeInterval::new(0.0, 8.0));
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(1.0, 1.75),
                TimeInterval::new(3.0, 5.75),
                TimeInterval::new(7.0, 8.0)
            ])
        );
        assert_eq!(
            time_template.active_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(0.0, 0.25),
                TimeInterval::new(1.75, 2.25),
                TimeInterval::new(5.75, 6.25)
            ])
        );

        let time_template =
            TimeTemplate::new_from(&vec![1.0, 3.0, 7.0], 0.25, TimeInterval::new(0.0, 8.0));
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(0.0, 0.75),
                TimeInterval::new(2.0, 2.75),
                TimeInterval::new(4.0, 6.75),
            ])
        );
        assert_eq!(
            time_template.active_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(0.75, 1.25),
                TimeInterval::new(2.75, 3.25),
                TimeInterval::new(6.75, 7.25)
            ])
        );

        let time_template =
            TimeTemplate::new_from(&vec![0.0, 3.0, 7.0], 0.25, TimeInterval::new(0.0, 8.0));
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(1.0, 2.75),
                TimeInterval::new(4.0, 6.75)
            ])
        );
        assert_eq!(
            time_template.active_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(0.0, 0.25),
                TimeInterval::new(2.75, 3.25),
                TimeInterval::new(6.75, 7.25)
            ])
        );
    }

    #[test]
    fn test_time_template_new_cyclic_from() {
        let time_template = TimeTemplate::new_cyclic_from(&vec![], 0.25, 8.0);
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![TimeInterval::new(0.0, 8.0)])
        );
        assert_eq!(time_template.active_regions, TimeIntervalUnion::Empty);

        let time_template = TimeTemplate::new_cyclic_from(&vec![0.0], 0.25, 8.0);
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![TimeInterval::new(1.0, 7.75)])
        );
        assert_eq!(
            time_template.active_regions,
            TimeIntervalUnion::new_from(vec![TimeInterval::new(-0.25, 0.25)])
        );

        let time_template = TimeTemplate::new_cyclic_from(&vec![1.0], 0.25, 8.0);
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![TimeInterval::new(2.0, 8.75)])
        );
        assert_eq!(
            time_template.active_regions,
            TimeIntervalUnion::new_from(vec![TimeInterval::new(0.75, 1.25)])
        );

        let time_template = TimeTemplate::new_cyclic_from(&vec![0.0, 2.0, 6.0], 0.25, 8.0);
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(1.0, 1.75),
                TimeInterval::new(3.0, 5.75),
                TimeInterval::new(7.0, 7.75)
            ])
        );
        assert_eq!(
            time_template.active_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(-0.25, 0.25),
                TimeInterval::new(1.75, 2.25),
                TimeInterval::new(5.75, 6.25)
            ])
        );

        let time_template = TimeTemplate::new_cyclic_from(&vec![1.0, 3.0, 7.0], 0.25, 8.0);
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(2.0, 2.75),
                TimeInterval::new(4.0, 6.75),
                TimeInterval::new(8.0, 8.75)
            ])
        );
        assert_eq!(
            time_template.active_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(0.75, 1.25),
                TimeInterval::new(2.75, 3.25),
                TimeInterval::new(6.75, 7.25)
            ])
        );

        let time_template = TimeTemplate::new_cyclic_from(&vec![0.0, 3.0, 7.0], 0.25, 8.0);
        assert_eq!(
            time_template.silence_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(1.0, 2.75),
                TimeInterval::new(4.0, 6.75)
            ])
        );
        assert_eq!(
            time_template.active_regions,
            TimeIntervalUnion::new_from(vec![
                TimeInterval::new(-0.25, 0.25),
                TimeInterval::new(2.75, 3.25),
                TimeInterval::new(6.75, 7.25)
            ])
        );
    }

    // #[test]
    // fn test_time_template_build_periodic_from() {
    //     let time_template = TimeTemplate::build_periodic_from(vec![], 0.25, 8.0).unwrap();
    //     assert_eq!(time_template.silence_regions, vec![(0.0, 8.0)]);
    //     assert_eq!(time_template.active_regions, vec![]);

    //     let time_template = TimeTemplate::build_periodic_from(vec![0.0], 0.25, 8.0).unwrap();
    //     assert_eq!(time_template.silence_regions, vec![(1.0, 7.75)]);
    //     assert_eq!(time_template.active_regions, vec![(7.75, 0.25)]);

    //     let time_template = TimeTemplate::build_periodic_from(vec![1.0], 0.25, 8.0).unwrap();
    //     assert_eq!(time_template.silence_regions, vec![(2.0, 0.75)]);
    //     assert_eq!(time_template.active_regions, vec![(0.75, 1.25)]);

    //     let time_template =
    //         TimeTemplate::build_periodic_from(vec![0.0, 2.0, 6.0], 0.25, 8.0).unwrap();
    //     assert_eq!(
    //         time_template.silence_regions,
    //         vec![(1.0, 1.75), (3.0, 5.75), (7.0, 7.75)]
    //     );
    //     assert_eq!(
    //         time_template.active_regions,
    //         vec![(7.75, 0.25), (1.75, 2.25), (5.75, 6.25)]
    //     );

    //     let time_template =
    //         TimeTemplate::build_periodic_from(vec![1.0, 3.0, 7.0], 0.25, 8.0).unwrap();
    //     assert_eq!(
    //         time_template.silence_regions,
    //         vec![(2.0, 2.75), (4.0, 6.75), (0.0, 0.75)]
    //     );
    //     assert_eq!(
    //         time_template.active_regions,
    //         vec![(0.75, 1.25), (2.75, 3.25), (6.75, 7.25)]
    //     );

    //     let time_template =
    //         TimeTemplate::build_periodic_from(vec![0.0, 3.0, 7.0], 0.25, 8.0).unwrap();
    //     assert_eq!(time_template.silence_regions, vec![(1.0, 2.75), (4.0, 6.75)]);
    //     assert_eq!(
    //         time_template.active_regions,
    //         vec![(7.75, 0.25), (2.75, 3.25), (6.75, 7.25)]
    //     );

    //     let time_template =
    //         TimeTemplate::build_periodic_from(vec![0.0, 1.0, 2.0], 0.25, 8.0).unwrap();
    //     assert_eq!(time_template.silence_regions, vec![(3.0, 7.75)]);
    //     assert_eq!(
    //         time_template.active_regions,
    //         vec![(7.75, 0.25), (0.75, 1.25), (1.75, 2.25)]
    //     );
    // }
}
