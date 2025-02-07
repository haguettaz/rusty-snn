//! Optimization-related utilities. For further details, check the [Gurobi documentation](https://docs.gurobi.com/).
use grb::prelude::*;
use itertools::Itertools;

use crate::core::utils::{TimeInterval, TimeIntervalUnion};
use crate::error::SNNError;
use crate::core::REFRACTORY_PERIOD;

/// The tolerance for a constraint to be considered as satisfied, c.f. [FeasibilityTol](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#parameterfeasibilitytol)
pub const FEASIBILITY_TOL: f64 = 1e-9;
/// The maximum number of new constraints to be added to the model during the refinement.
pub const MAX_NEW_CSTRS: usize = 1;
/// The maximum number of iterations for the memorization process.
pub const MAX_ITER: usize = 10_000;

/// The objective function for the optimization problem.
#[derive(Clone, Copy)]
pub enum Objective {
    /// Uniform prior.
    None,
    /// Gaussian prior, i.e., L2 norm, for low magnitude weights.
    L2,
    /// Laplace prior, i.e., L1 norm, for sparse weights.
    L1,
}

impl Objective {
    /// Returns the objective function from a string.
    pub fn from_str(s: &str) -> Result<Self, SNNError> {
        match s {
            "none" => Ok(Objective::None),
            "l2" => Ok(Objective::L2),
            "l1" => Ok(Objective::L1),
            _ => Err(SNNError::InvalidParameters("Invalid objective".to_string())),
        }
    }
}

/// The decomposition of the timeline into relevant intervals.
#[derive(PartialEq, Debug, Clone)]
pub struct TimeTemplate {
    // The times at which the neuron fires.
    pub firing_times: Vec<f64>,
    // The times at which the neuron is silent.
    pub silence_regions: TimeIntervalUnion,
    // The times at which the neuron is active, i.e., is about to fire.
    pub active_regions: TimeIntervalUnion,
    // The time interval of the template.
    pub interval: TimeInterval,
}

impl TimeTemplate {
    pub fn new_from(firing_times: &Vec<f64>, half_width: f64, interval: TimeInterval) -> Self {
        match interval {
            TimeInterval::Empty => TimeTemplate {
                firing_times: vec![],
                silence_regions: TimeIntervalUnion::Empty,
                active_regions: TimeIntervalUnion::Empty,
                interval: TimeInterval::Empty,
            },
            TimeInterval::Closed { start, end } => {
                if firing_times.is_empty() {
                    TimeTemplate {
                        firing_times: vec![],
                        silence_regions: TimeIntervalUnion::new_from(vec![interval.clone()]),
                        active_regions: TimeIntervalUnion::Empty,
                        interval: interval.clone(),
                    }
                } else {
                    let mut silence_regions: Vec<TimeInterval> = vec![];

                    let new_interval =
                        TimeInterval::new(start, firing_times.first().unwrap() - half_width);
                    let intersect = interval.intersect(new_interval);
                    if !intersect.is_empty() {
                        silence_regions.push(intersect);
                    }

                    silence_regions.extend(
                        firing_times
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
                        TimeInterval::new(firing_times.last().unwrap() + REFRACTORY_PERIOD, end);
                    let intersect = interval.intersect(new_interval);
                    if !intersect.is_empty() {
                        silence_regions.push(intersect);
                    }

                    let active_regions = firing_times
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

                    let mut firing_times = firing_times.clone();
                    firing_times.retain(|time| interval.contains(*time));

                    TimeTemplate {
                        firing_times,
                        silence_regions: TimeIntervalUnion::new_from(silence_regions),
                        active_regions: TimeIntervalUnion::new_from(active_regions),
                        interval,
                    }
                }
            }
        }
    }

    /// Create a new time template from a list of sorted firing times.
    pub fn new_cyclic_from(firing_times: &Vec<f64>, half_width: f64, period: f64) -> Self {
        if firing_times.is_empty() {
            TimeTemplate {
                firing_times: vec![],
                silence_regions: TimeIntervalUnion::new_from(vec![TimeInterval::new(0.0, period)]),
                active_regions: TimeIntervalUnion::Empty,
                interval: TimeInterval::new(0.0, period),
            }
        } else {
            let before_first = firing_times[0] - half_width;
            let interval = TimeInterval::new(before_first, before_first + period);

            let mut silence_regions = firing_times
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
                firing_times.last().unwrap() + REFRACTORY_PERIOD,
                before_first + period,
            );
            let intersect = interval.intersect(new_interval);
            if !intersect.is_empty() {
                silence_regions.push(intersect);
            }

            let active_regions = firing_times
                .iter()
                .map(|time| TimeInterval::new(time - half_width, time + half_width))
                .collect::<Vec<TimeInterval>>();

            let firing_times = firing_times.clone();
            TimeTemplate {
                firing_times,
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
    env.set(param::FeasibilityTol, FEASIBILITY_TOL)
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

pub fn set_grb_objective(
    model: &mut grb::Model,
    weights: &Vec<grb::Var>,
    objective: Objective,
) -> Result<(), SNNError> {
    match objective {
        Objective::None => (),
        Objective::L2 => {
            let mut obj_expr = grb::expr::QuadExpr::new();
            for &var in weights.iter() {
                obj_expr.add_qterm(1.0, var, var);
            }
            model
                .set_objective(obj_expr, ModelSense::Minimize)
                .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
        }
        Objective::L1 => {
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

#[cfg(test)]
mod tests {
    use super::*;

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
