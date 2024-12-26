//! This module provides the `Neuron` structure which composes the `Network` structure.

use core::f64;
use embed_doc_image::embed_doc_image;
use grb::prelude::*;
use itertools::Itertools;
use lambert_w::lambert_w0;
use log::{debug, error, info, trace, warn};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::max;

use super::connection::{Connection, Input};
use super::error::SNNError;
use super::optim::*;
use super::spike_train::{InSpike, InputSpike, Spike};
use super::utils::{mean, median};
use crate::{FIRING_THRESHOLD, POTENTIAL_TOLERANCE, REFRACTORY_PERIOD};

// /// Represents an input to a neuron.
// #[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
// pub struct Input {
//     // The ID of the input.
//     id: usize,
//     // The ID of the source neuron.
//     source_id: usize,
//     // The connection weight.
//     weight: f64,
//     // The connection delay.
//     delay: f64,
// }

// impl Input {
//     /// Returns the ID of neuron at the origin of the input.
//     pub fn source_id(&self) -> usize {
//         self.source_id
//     }

//     /// Returns the weight of the input.
//     pub fn weight(&self) -> f64 {
//         self.weight
//     }

//     /// Returns the delay of the input.
//     pub fn delay(&self) -> f64 {
//         self.delay
//     }
// }

// /// Represents an input spike to a neuron, i.e., a time and a weight.
// /// This structure is used for simulation purposes.
// #[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
// pub struct InSpike {
//     // The time at which the spike is received.
//     pub time: f64,
//     // The weight of the spike.
//     pub weight: f64,
// }

// /// Represents an input spike to a neuron, i.e., a time and a weight.
// /// This structure is used for simulation purposes.
// #[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
// pub struct Spike {
//     // The time at which the spike is emitted.
//     pub time: f64,
//     // The ID of the source neuron.
//     pub source_id: usize,
// }

// impl InputSpike {
//     /// Evaluate the contribution of the input spike at the given time.
//     pub fn eval(&self, t: f64) -> f64 {
//         let dt = t - self.time;
//         dt * (1_f64 - dt).exp() * self.weight
//     }
// }

/// Represents a spiking neuron.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Neuron {
    // The neuron ID.
    id: usize,
    // The firing threshold of the neuron.
    threshold: f64,
    // A collection of (sorted) firing times of the neuron.
    firing_times: Vec<f64>,
    // // A collection of inputs to the neuron (sorted by 1. source_id and 2. delay?)
    // inputs: Vec<Input>,
    // A collection of input spikes to the neuron, sorted by time of arrival.
    inspikes: Vec<InSpike>,
}

impl Neuron {
    /// Create a new empty neuron with nominal threshold (see FIRING_THRESHOLD).
    pub fn new(id: usize) -> Self {
        Neuron {
            id,
            threshold: FIRING_THRESHOLD,
            firing_times: vec![],
            inspikes: vec![],
        }
    }

    /// Returns the neuron potential at the given time, based on all its input spikes.
    /// When the potential exceeds the threshold, the neuron fires.
    ///
    /// ![A Foobaring][neuron]
    ///
    #[embed_doc_image("neuron", "images/neuron.svg")]
    pub fn potential(&self, t: f64) -> f64 {
        self.inspikes
            .iter()
            .filter(|input_spike| input_spike.time() < t)
            .fold(0.0, |acc, input_spike| {
                let dt = t - input_spike.time();
                acc + dt * (1_f64 - dt).exp() * input_spike.weight()
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

    // /// Returns a slice of inputs of the neuron.
    // pub fn inputs(&self) -> &[Input] {
    //     &self.inputs
    // }

    /// Returns the number of firing times of the neuron.
    pub fn num_firing_times(&self) -> usize {
        self.firing_times.len()
    }

    // /// Returns the number of inputs of the neuron.
    // pub fn num_inputs(&self) -> usize {
    //     self.inputs.len()
    // }

    /// Returns a slice of input spikes of the neuron.
    pub fn inspikes(&self) -> &[InSpike] {
        &self.inspikes
    }

    /// Returns the total number of input spikes of the neuron.
    pub fn num_inspikes(&self) -> usize {
        self.inspikes.len()
    }

    // /// Returns the number of inputs of the neuron from a given source ID.
    // /// Reminder: inputs are sorted by source_id and delay.
    // pub fn num_inputs_from(&self, source_id: usize) -> usize {
    //     self.inputs
    //         .iter()
    //         .filter(|input| input.source_id == source_id)
    //         .count()
    // }

    // // Returns the minimum delay path from a source neuron (specified by its ID) to the neuron.
    // // The function returns None if there is no path from the source neuron to the neuron.
    // pub fn min_delay_path(&self, source_id: usize) -> Option<f64> {
    //     if source_id == self.id {
    //         return Some(0.0);
    //     }
    //     self.inputs
    //         .iter()
    //         .filter(|input| input.source_id == source_id)
    //         .map(|input| input.delay)
    //         .min_by(|a, b| a.partial_cmp(b).unwrap())
    // }

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

    // /// Clear all neuron inputs.
    // pub fn clear_inputs(&mut self) {
    //     self.inputs = vec![];
    //     self.inspikes = vec![];
    // }

    // /// Add an input to the neuron, with the specified source ID, weight, and delay.
    // pub fn add_input(&mut self, source_id: usize, weight: f64, delay: f64) {
    //     self.inputs.push(Input {
    //         id: self.inputs.len(),
    //         source_id,
    //         weight,
    //         delay,
    //     });
    // }

    /// Clear all neuron input spikes.
    pub fn reset_inspikes(&mut self) {
        self.inspikes = vec![];
    }

    // /// Add input spikes to the neuron from a given source ID firing at the specified time.
    // pub fn add_inspikes_for_source(&mut self, source_id: usize, firing_times: &[f64]) {
    //     let new_spikes: Vec<InputSpike> = self
    //         .inputs
    //         .iter()
    //         .filter(|input| input.source_id == source_id)
    //         .flat_map(|input| {
    //             firing_times.iter().map(|ft| InputSpike {
    //                 time: ft + input.delay,
    //                 input_id: input.id,
    //             })
    //         })
    //         .collect();

    //     self.merge_inspikes(new_spikes);
    // }

    /// Merge new input spikes with the existing ones and sort them by time of arrival.
    pub fn add_inspikes(&mut self, new_inspike: &mut Vec<InSpike>) {
        self.inspikes.append(new_inspike);
        self.inspikes.sort_by(|inspike_1, inspike_2| {
            inspike_1.time().partial_cmp(&inspike_2.time()).unwrap()
        });
    }

    /// Returns the next firing time of the neuron, if any.
    pub fn next_spike(&self, mut start: f64) -> Option<Spike> {
        if let Some(last) = self.firing_times().last() {
            start = last + REFRACTORY_PERIOD;
        }

        if self.potential(start) >= self.threshold {
            return Some(Spike::new(self.id, start));
        }

        let mut firing_time = f64::NAN;
        let pos = self
            .inspikes
            .binary_search_by(|input_spike| {
                if input_spike.time() >= start {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            })
            .unwrap_or_else(|pos| pos);

        for (i, input_spike) in self.inspikes[pos..].iter().enumerate() {
            if firing_time <= input_spike.time() {
                break;
            }

            let (a, b) =
                self.inspikes[..=pos + i]
                    .iter()
                    .fold((0.0, 0.0), |(mut a, mut b), item| {
                        a += item.weight() * (item.time() - input_spike.time()).exp();
                        b += item.weight() * item.time() * (item.time() - input_spike.time()).exp();
                        (a, b)
                    });

            firing_time = match a == 0.0 {
                true => input_spike.time() + 1.0 + (-b / self.threshold()).ln(),
                false => {
                    b / a
                        - lambert_w0(
                            -self.threshold() / a * (b / a - 1.0 - input_spike.time()).exp(),
                        )
                }
            };

            if firing_time < input_spike.time() {
                firing_time = f64::NAN;
            }
        }

        match firing_time.is_finite() {
            true => Some(Spike::new(self.id, firing_time)),
            false => None,
        }
    }

    /// Compute the weights of the inputs that reproduce the neuron's firing times from the inputs.
    /// Memorization amounts to minimizing a convex objective function subject to linear constraints.
    /// No cost / uniform prior is mostly used for feasibility check.
    /// L2 cost / Gaussian prior yields low magnitude solutions.
    /// L1 cost / Laplace prior yields sparse solutions (c.f. [here](https://math.stackexchange.com/questions/1639716/how-can-l-1-norm-minimization-with-linear-equality-constraints-basis-pu))
    pub fn memorize_periodic_spike_trains(
        &self,
        spike_train: &Vec<Spike>, // for multiple spike trains, use Vec<Vec<Spike>> instead
        connections: &mut Vec<Vec<Connection>>,
        period: f64,
        lim_weights: (f64, f64),
        max_level: f64,
        min_slope: f64,
        half_width: f64,
        objective: Objective,
    ) -> Result<(), SNNError> {
        // Initialize the neuron firing times
        let firing_times = spike_train
            .iter()
            .filter(|spike| spike.source_id() == self.id)
            .map(|spike| spike.time())
            .collect::<Vec<f64>>();

        let num_inputs = connections
            .iter()
            .flat_map(|inputs_from| inputs_from.iter())
            .count();
        if (num_inputs as f64) * period * (1_f64 - period).exp() > POTENTIAL_TOLERANCE {
            return Err(SNNError::InvalidPeriod);
        }
        let mut inputs = connections
            .iter()
            .flat_map(|inputs_from| inputs_from.iter())
            .enumerate()
            .map(|(id, input)| Input::new(id, input.source_id(), input.weight(), input.delay()))
            .collect::<Vec<Input>>();

        // Initialize input spikes from the provided spike train
        let mut input_spikes: Vec<InputSpike> = spike_train
            .iter()
            .flat_map(|spike| {
                inputs
                    .iter()
                    .filter(|input| input.source_id() == spike.source_id())
                    .map(|input| InputSpike::new(input.id(), spike.time() + input.delay()))
            })
            .collect();
        input_spikes.sort_by(|input_spike_1, input_spike_2| {
            input_spike_1
                .time()
                .partial_cmp(&input_spike_2.time())
                .unwrap()
        });

        // Initialize the Gurobi environment and model for memorization
        let mut model = init_gurobi(format!("neuron_{}", self.id).as_str(), "gurobi.log")?;

        // Setup the decision variables, i.e., the weights
        let weights = init_weights(&mut model, num_inputs, lim_weights)?;

        // Setup the objective function
        init_objective(&mut model, &weights, objective)?;

        // Setup the firing time constraints
        add_firing_time_constraints(&mut model, &weights, &firing_times, &input_spikes, period)?;

        let mut is_valid = false;
        while !is_valid {
            // For fixed constraints, determine the optimal weights
            println!("1. Optimize weights...");
            model
                .optimize()
                .map_err(|e| SNNError::GurobiError(e.to_string()))?;

            // If the model is infeasible, return an error
            let status = model
                .status()
                .map_err(|e| SNNError::GurobiError(e.to_string()))?;

            println!("Status: {:?}", status);
            if Status::Optimal != status {
                return Err(SNNError::InfeasibleMemorization);
            }

            // Set the input weights to the optimal values
            for input in inputs.iter_mut() {
                let weight = model
                    .get_obj_attr(grb::attribute::VarDoubleAttr::X, &weights[input.id()])
                    .map_err(|e| SNNError::GurobiError(e.to_string()))?;
                input.update_weight(weight);
            }

            // Check if the current solution satifies all template constraints
            debug!("2. Refine constraints...");
            is_valid = refine_constraints(
                &mut model,
                &weights,
                &firing_times,
                &inputs,
                &input_spikes,
                period,
                max_level,
                min_slope,
                half_width,
            )?;
        }

        // Update the connections with the optimal weights
        for (input, connection) in inputs.iter().zip(
            connections
                .iter_mut()
                .flat_map(|connection_between| connection_between.iter_mut()),
        ) {
            connection.update_weight(input.weight());
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
            mean(inputs.iter().map(|input| input.weight())),
            median(inputs.iter().map(|input| input.weight()))
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
                inspikes: vec![]
            }
        );
    }

    // #[test]
    // fn test_neuron_add_input() {
    //     let mut neuron = Neuron::new(0);

    //     neuron.add_input(0, 0.5, 0.5);
    //     neuron.add_input(1, 1.0, 1.0);
    //     neuron.add_input(0, -0.5, 2.0);
    //     neuron.add_input(3, 0.75, 1.5);
    //     neuron.add_input(1, 1.0, 0.25);

    //     assert_eq!(
    //         neuron.inputs(),
    //         &[
    //             Input {
    //                 id: 0,
    //                 source_id: 0,
    //                 weight: 0.5,
    //                 delay: 0.5
    //             },
    //             Input {
    //                 id: 1,
    //                 source_id: 1,
    //                 weight: 1.0,
    //                 delay: 1.0
    //             },
    //             Input {
    //                 id: 2,
    //                 source_id: 0,
    //                 weight: -0.5,
    //                 delay: 2.0
    //             },
    //             Input {
    //                 id: 3,
    //                 source_id: 3,
    //                 weight: 0.75,
    //                 delay: 1.5
    //             },
    //             Input {
    //                 id: 4,
    //                 source_id: 1,
    //                 weight: 1.0,
    //                 delay: 0.25
    //             },
    //         ]
    //     );
    // }

    // #[test]
    // fn test_neuron_min_delay_path() {
    //     let mut neuron = Neuron::new(0);

    //     neuron.add_input(0, 0.5, 0.5);
    //     neuron.add_input(1, 1.0, 1.0);
    //     neuron.add_input(0, -0.5, 2.0);
    //     neuron.add_input(3, 0.75, 1.5);
    //     neuron.add_input(1, 1.0, 0.25);

    //     assert_eq!(neuron.min_delay_path(0), Some(0.0));
    //     assert_eq!(neuron.min_delay_path(1), Some(0.25));
    //     assert_eq!(neuron.min_delay_path(2), None);
    //     assert_eq!(neuron.min_delay_path(3), Some(1.5));
    // }

    #[test]
    fn test_add_inspikes() {
        let mut neuron = Neuron::new(0);

        let mut new_inspikes = vec![
            InSpike::new(0.5, 0.0),
            InSpike::new(1.0, 1.0),
            InSpike::new(-0.5, 2.0),
            InSpike::new(0.75, 1.5),
            InSpike::new(1.0, 0.25),
        ];
        neuron.add_inspikes(&mut new_inspikes);

        let mut new_inspikes = vec![InSpike::new(-3.0, 7.5), InSpike::new(5.0, 0.25)];
        neuron.add_inspikes(&mut new_inspikes);

        let expected_inspikes = vec![
            InSpike::new(0.5, 0.0),
            InSpike::new(1.0, 0.25),
            InSpike::new(5.0, 0.25),
            InSpike::new(1.0, 1.0),
            InSpike::new(0.75, 1.5),
            InSpike::new(-0.5, 2.0),
            InSpike::new(-3.0, 7.5),
        ];
        assert_eq!(neuron.inspikes(), expected_inspikes);
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
    fn test_neuron_potential() {
        let mut neuron = Neuron::new(42);

        assert_eq!(neuron.potential(0.0), 0.0);
        assert_eq!(neuron.potential(1.0), 0.0);
        assert_eq!(neuron.potential(2.0), 0.0);

        let mut new_inspikes = vec![
            InSpike::new(1.0, 1.0),
            InSpike::new(1.0, 2.0),
            InSpike::new(-1.0, 1.0),
        ];
        neuron.add_inspikes(&mut new_inspikes);

        assert_eq!(neuron.potential(0.0), 0.0);
        assert_eq!(neuron.potential(1.0), 0.0);
        assert_eq!(neuron.potential(2.0), 0.0);
        assert_eq!(neuron.potential(3.0), 1.0);
    }

    #[test]
    fn test_neuron_next_spike() {
        // 1 inspike producing a spike
        let mut neuron = Neuron::new(42);
        neuron.add_inspikes(&mut vec![InSpike::new(1.0, 1.0)]);
        assert_eq!(neuron.next_spike(0.0), Some(Spike::new(42, 2.0)));

        // 2 inspikes canceling each other
        let mut neuron = Neuron::new(42);
        neuron.add_inspikes(&mut vec![InSpike::new(1.0, 1.0), InSpike::new(-1.0, 1.0)]);
        assert_eq!(neuron.next_spike(0.0), None);

        // 4 inspikes producing a spike
        let mut neuron = Neuron::new(42);
        neuron.add_inspikes(&mut vec![
            InSpike::new(1.0, 1.0),
            InSpike::new(1.0, 3.0),
            InSpike::new(1.0, 4.0),
            InSpike::new(-0.25, 1.5),
        ]);
        assert_eq!(
            neuron.next_spike(0.0),
            Some(Spike::new(42, 3.2757576038986502))
        );

        // 1 inspike producing a spike after refractory period
        let mut neuron = Neuron::new(42);
        neuron.add_firing_time(2.0).unwrap();
        neuron.add_inspikes(&mut vec![InSpike::new(10.0, 1.0)]);
        assert_eq!(neuron.next_spike(0.0), Some(Spike::new(42, 3.0)));

        // many zero-weight inspikes producing no spike
        let mut neuron = Neuron::new(42);
        neuron.add_inspikes(&mut vec![InSpike::new(0.0, 1.0); 100]);
        assert_eq!(neuron.next_spike(0.0), None);

        // no inspike producing a spike because of zero firing threshold
        let mut neuron = Neuron::new(42);
        neuron.set_threshold(0.0);
        assert_eq!(neuron.next_spike(0.0), Some(Spike::new(42, 0.0)));

        // many inspikes producing no spike because of extreme firing threshold
        let mut neuron = Neuron::new(42);
        neuron.set_threshold(f64::INFINITY);
        neuron.add_inspikes(&mut vec![InSpike::new(100.0, 1.0); 100]);
        assert_eq!(neuron.next_spike(0.0), None);
    }

    #[test]
    fn test_memorize_empty_periodic_spike_train() {
        let period = 100.0;
        let lim_weights = (-1.0, 1.0);
        let max_level = 0.0;
        let min_slope = 0.0;
        let half_width = 0.5;
        let objective = Objective::L2Norm;

        let spike_train: Vec<Spike> = vec![
            Spike::new(1, 1.0),
            Spike::new(1, 3.0),
            Spike::new(1, 25.0),
            Spike::new(2, 5.0),
            Spike::new(2, 77.0),
            Spike::new(2, 89.0),
        ];

        let neuron = Neuron::new(0);

        let mut connections = vec![
            vec![Connection::build(0, 0, 0, 1.0, 1.0).unwrap()],
            vec![
                Connection::build(1, 0, 1, 1.0, 2.0).unwrap(),
                Connection::build(2, 0, 1, 1.0, 5.0).unwrap(),
            ],
            vec![Connection::build(3, 0, 2, 1.0, 0.5).unwrap()],
        ];

        assert_eq!(
            neuron.memorize_periodic_spike_trains(
                &spike_train,
                &mut connections,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                objective
            ),
            Ok(())
        );

        let expected_connections = vec![
            vec![Connection::build(0, 0, 0, 0.0, 1.0).unwrap()],
            vec![
                Connection::build(1, 0, 1, 0.0, 2.0).unwrap(),
                Connection::build(2, 0, 1, 0.0, 5.0).unwrap(),
            ],
            vec![Connection::build(3, 0, 2, 0.0, 0.5).unwrap()],
        ];

        assert_eq!(connections, expected_connections);
    }

    #[test]
    fn test_memorize_single_spike_periodic_spike_train() {
        let period = 100.0;
        let lim_weights = (-5.0, 5.0);
        let max_level = 0.5;
        let min_slope = 0.0;
        let half_width = 0.25;

        let spike_trains = vec![
            Spike::new(0, 1.55),
            Spike::new(1, 1.0),
            Spike::new(2, 1.5),
            Spike::new(3, 2.0),
            Spike::new(4, 3.5),
        ];

        let neuron = Neuron::new(0);

        let mut connections = vec![
            vec![Connection::build(0, 0, 0, 1.0, 0.0).unwrap()],
            vec![Connection::build(1, 1, 0, 1.0, 0.0).unwrap()],
            vec![Connection::build(2, 2, 0, 1.0, 0.0).unwrap()],
            vec![Connection::build(3, 3, 0, 1.0, 0.0).unwrap()],
            vec![Connection::build(4, 4, 0, 1.0, 0.0).unwrap()],
        ];

        neuron
            .memorize_periodic_spike_trains(
                &spike_trains,
                &mut connections,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                Objective::None,
            )
            .expect("Memorization failed");

        neuron
            .memorize_periodic_spike_trains(
                &spike_trains,
                &mut connections,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                Objective::L2Norm,
            )
            .expect("L2-memorization failed");

        assert_relative_eq!(connections[0][0].weight(), -0.187902, epsilon = 1e-6);
        assert_relative_eq!(connections[1][0].weight(), 1.157671, epsilon = 1e-6);
        assert_relative_eq!(connections[2][0].weight(), 0.011027, epsilon = 1e-6);
        assert_relative_eq!(connections[3][0].weight(), -0.415485, epsilon = 1e-6);
        assert_relative_eq!(connections[4][0].weight(), 0.0);

        neuron
            .memorize_periodic_spike_trains(
                &spike_trains,
                &mut connections,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                Objective::L1Norm,
            )
            .expect("L1-memorization failed");

        assert_relative_eq!(connections[0][0].weight(), -0.178366, epsilon = 1e-6);
        assert_relative_eq!(connections[1][0].weight(), 1.159324, epsilon = 1e-6);
        assert_relative_eq!(connections[2][0].weight(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(connections[3][0].weight(), -0.415485, epsilon = 1e-6);
        assert_relative_eq!(connections[4][0].weight(), 0.0);
    }
}
