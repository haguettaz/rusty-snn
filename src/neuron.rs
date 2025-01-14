//! This module provides the `Neuron` structure which composes the `Network` structure.

use core::f64;
use embed_doc_image::embed_doc_image;
use grb::prelude::*;
use log::{debug, info};
use serde::{Deserialize, Serialize};

// use super::connection::{Connection};
use super::error::SNNError;
use super::optim::*;
use super::signal::{crossing_potential, potential};
use super::spike_train::{InSpike, Spike};
use super::{FIRING_THRESHOLD, INSPIKE_MIN, REFRACTORY_PERIOD};

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
        potential(&self.inspikes, t)
    }

    // /// Returns the neuron potential at the given time, based on all its input spikes and their periodic extension.
    // /// The result only make sense if the contribution of a spike is negligible after the prescribed period (see POTENTIAL_TOLERANCE).
    // fn periodic_potential(&self, t: f64, period: f64) -> f64 {
    //     periodic_potential(&self.inspikes, t, period)
    // }

    // /// Returns the neuron potential derivative at the given time, based on all its input spikes and their periodic extension.
    // /// The result only make sense if the contribution of a spike is negligible after the prescribed period (see POTENTIAL_TOLERANCE).
    // fn periodic_potential_derivative(&self, t: f64, period: f64) -> f64 {
    //     periodic_potential_derivative(&self.inspikes, t, period)
    // }

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
                return Err(SNNError::InvalidParameters(
                    "Firing times must be finite".to_string(),
                ));
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

    /// Clear all neuron input spikes.
    pub fn clear_inspikes(&mut self) {
        self.inspikes = vec![];
    }

    /// Set input spikes and sort them by time of arrival.
    pub fn set_inspikes(&mut self, new_inspike: &mut Vec<InSpike>) {
        self.clear_inspikes();
        self.extend_inspikes(new_inspike);
    }

    /// Add new inspike to the neuron while keeping the inspikes sorted by time of arrival.
    pub fn add_inspike(&mut self, new_inspike: InSpike) {
        let pos = match self
            .inspikes
            .binary_search_by(|inspike| inspike.time().partial_cmp(&new_inspike.time()).unwrap())
        {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        self.inspikes.insert(pos, new_inspike);
    }

    /// Merge new input spikes with the existing ones and sort them by time of arrival.
    pub fn extend_inspikes(&mut self, new_inspike: &mut Vec<InSpike>) {
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

        match crossing_potential(start, self.threshold, &self.inspikes) {
            Some(t) => Some(Spike::new(self.id, t)),
            None => None,
        }
    }

    /// Compute the weights of the inputs that reproduce the neuron's firing times from the inputs.
    /// Memorization amounts to minimizing a convex objective function subject to linear constraints.
    /// No cost / uniform prior is mostly used for feasibility check.
    /// L2 cost / Gaussian prior yields low magnitude solutions.
    /// L1 cost / Laplace prior yields sparse solutions (c.f. [here](https://math.stackexchange.com/questions/1639716/how-can-l-1-norm-minimization-with-linear-equality-constraints-basis-pu))
    pub fn memorize_periodic_spike_train(
        &self,
        firing_times: &Vec<f64>, // for multiple spike trains, use Vec<Vec<f64>> instead
        inspikes: &mut Vec<InSpike>, // for multiple spike trains, use Vec<Vec<InSpike>> instead
        period: f64,
        lim_weights: (f64, f64),
        max_level: f64,
        min_slope: f64,
        half_width: f64,
        objective: Objective,
    ) -> Result<Vec<f64>, SNNError> {
        if inspikes
            .iter()
            .any(|inspike| inspike.kernel(inspike.time() + period) > INSPIKE_MIN)
        {
            return Err(SNNError::InvalidParameters(
                "The contribution of a spike to the neuron potential must be negligible after one period".to_string(),
            ));
        }

        let num_inputs = inspikes
            .iter()
            .map(|spike| spike.input_id())
            .max()
            .unwrap_or_default()
            + 1;

        // Initialize the Gurobi environment and model for memorization
        let mut model = init_gurobi(format!("neuron_{}", self.id).as_str(), "gurobi.log")?;

        // Setup the decision variables, i.e., the weights
        let weights = init_weights(&mut model, num_inputs, lim_weights)?;

        // Setup the objective function
        init_objective(&mut model, &weights, objective)?;

        // Setup the firing time constraints
        add_firing_time_constraints(&mut model, &weights, &firing_times, &inspikes, period)?;

        let mut is_valid = false;
        let mut optimal_weights: Vec<f64> = vec![0.0; num_inputs];

        while !is_valid {
            // For fixed constraints, determine the optimal weights
            debug!("1. Optimize weights...");
            model
                .optimize()
                .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

            // If the model is infeasible, return an error
            let status = model
                .status()
                .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

            if Status::Optimal != status {
                return Err(SNNError::InfeasibleMemorization);
            }

            // Store the optimal weights in a vector and update the inspikes accordingly
            for (i, weight) in weights.iter().enumerate() {
                optimal_weights[i] = model
                    .get_obj_attr(grb::attribute::VarDoubleAttr::X, weight)
                    .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
            }
            for inspike in inspikes.iter_mut() {
                inspike.set_weight(optimal_weights[inspike.input_id()]);
            }

            // Check if the current solution satifies all template constraints
            debug!("2. Refine constraints...");
            is_valid = refine_constraints(
                &mut model,
                &weights,
                &firing_times,
                &inspikes,
                period,
                max_level,
                min_slope,
                half_width,
            )?;
        }

        // Get the total number of constraints and the objective value
        let num_constrs = model
            .get_attr(grb::attribute::ModelIntAttr::NumConstrs)
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
        let obj_val = model
            .get_attr(grb::attribute::ModelDoubleAttr::ObjVal)
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

        info!(
            "Memorization done! The cost is {} for {} constraints.",
            obj_val, num_constrs
        );
        Ok(optimal_weights)
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

    #[test]
    fn test_add_inspikes() {
        let mut neuron = Neuron::new(0);

        let mut new_inspikes = vec![
            InSpike::new(0, 0.5, 0.0),
            InSpike::new(1, 1.0, 1.25),
            InSpike::new(2, -0.5, 2.0),
            InSpike::new(3, 0.75, 1.5),
            InSpike::new(1, 1.0, 0.25),
        ];
        neuron.extend_inspikes(&mut new_inspikes);

        let mut new_inspikes = vec![InSpike::new(0, -3.0, 7.5), InSpike::new(4, 5.0, 0.25)];
        neuron.extend_inspikes(&mut new_inspikes);

        let expected_inspikes = vec![
            InSpike::new(0, 0.5, 0.0),
            InSpike::new(1, 1.0, 0.25),
            InSpike::new(4, 5.0, 0.25),
            InSpike::new(1, 1.0, 1.25),
            InSpike::new(3, 0.75, 1.5),
            InSpike::new(2, -0.5, 2.0),
            InSpike::new(0, -3.0, 7.5),
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
            InSpike::new(0, 1.0, 1.0),
            InSpike::new(1, 1.0, 2.0),
            InSpike::new(2, -1.0, 1.0),
        ];
        neuron.extend_inspikes(&mut new_inspikes);

        assert_eq!(neuron.potential(0.0), 0.0);
        assert_eq!(neuron.potential(1.0), 0.0);
        assert_eq!(neuron.potential(2.0), 0.0);
        assert_eq!(neuron.potential(3.0), 1.0);
    }

    #[test]
    fn test_neuron_next_spike() {
        // 1 inspike producing a spike
        let mut neuron = Neuron::new(42);
        neuron.extend_inspikes(&mut vec![InSpike::new(0, 1.0, 1.0)]);
        assert_eq!(neuron.next_spike(0.0), Some(Spike::new(42, 2.0)));

        // 2 inspikes canceling each other
        let mut neuron = Neuron::new(42);
        neuron.extend_inspikes(&mut vec![
            InSpike::new(0, 1.0, 1.0),
            InSpike::new(1, -1.0, 1.0),
        ]);
        assert_eq!(neuron.next_spike(0.0), None);

        // 4 inspikes producing a spike
        let mut neuron = Neuron::new(42);
        neuron.extend_inspikes(&mut vec![
            InSpike::new(0, 1.0, 1.0),
            InSpike::new(1, 1.0, 3.0),
            InSpike::new(2, 1.0, 4.0),
            InSpike::new(3, -0.25, 1.5),
        ]);
        assert_eq!(
            neuron.next_spike(0.0),
            Some(Spike::new(42, 3.2757576038986502))
        );

        // 1 inspike producing a spike after refractory period
        let mut neuron = Neuron::new(42);
        neuron.add_firing_time(2.0).unwrap();
        neuron.extend_inspikes(&mut vec![InSpike::new(0, 10.0, 1.0)]);
        assert_eq!(neuron.next_spike(0.0), Some(Spike::new(42, 3.0)));

        // 1 zero-weight inspike producing no spike
        let mut neuron = Neuron::new(42);
        neuron.extend_inspikes(&mut vec![InSpike::new(0, 0.0, 1.0)]);
        assert_eq!(neuron.next_spike(0.0), None);

        // no inspike producing a spike because of zero firing threshold
        let mut neuron = Neuron::new(42);
        neuron.set_threshold(0.0);
        assert_eq!(neuron.next_spike(0.0), Some(Spike::new(42, 0.0)));

        // many inspikes producing no spike because of extreme firing threshold
        let mut neuron = Neuron::new(42);
        neuron.set_threshold(f64::INFINITY);
        neuron.extend_inspikes(&mut vec![InSpike::new(0, 1_000_000.0, 1.0)]);
        assert_eq!(neuron.next_spike(0.0), None);
    }

    // #[test]
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

        let firing_times: Vec<f64> = vec![];
        let mut inspikes: Vec<InSpike> = vec![
            InSpike::new(0, 1.0, 3.0),
            InSpike::new(0, 1.0, 5.0),
            InSpike::new(0, 1.0, 6.0),
            InSpike::new(1, 1.0, 8.0),
            InSpike::new(1, 1.0, 27.0),
            InSpike::new(1, 1.0, 30.0),
            InSpike::new(2, 1.0, 5.5),
            InSpike::new(2, 1.0, 77.5),
            InSpike::new(2, 1.0, 89.5),
        ];

        assert_eq!(
            neuron
                .memorize_periodic_spike_train(
                    &firing_times,
                    &mut inspikes,
                    period,
                    lim_weights,
                    max_level,
                    min_slope,
                    half_width,
                    objective,
                )
                .unwrap(),
            vec![0.0, 0.0, 0.0]
        );

        // let expected_connections = vec![
        //     vec![Connection::build(0, 0, 0, 0.0, 1.0).unwrap()],
        //     vec![
        //         Connection::build(1, 0, 1, 0.0, 2.0).unwrap(),
        //         Connection::build(2, 0, 1, 0.0, 5.0).unwrap(),
        //     ],
        //     vec![Connection::build(3, 0, 2, 0.0, 0.5).unwrap()],
        // ];

        // assert_eq!(connections, expected_connections);
    }

    // #[test]
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

        let mut neuron = Neuron::new(0);

        let firing_times: Vec<f64> = vec![1.55];
        let mut inspikes: Vec<InSpike> = vec![
            InSpike::new(1, 1.0, 1.0),
            InSpike::new(2, 1.0, 1.5),
            InSpike::new(0, 1.0, 1.55),
            InSpike::new(3, 1.0, 2.0),
            InSpike::new(4, 1.0, 3.5),
        ];

        // let mut connections = vec![
        //     vec![Connection::build(0, 0, 0, 1.0, 0.0).unwrap()],
        //     vec![Connection::build(1, 1, 0, 1.0, 0.0).unwrap()],
        //     vec![Connection::build(2, 2, 0, 1.0, 0.0).unwrap()],
        //     vec![Connection::build(3, 3, 0, 1.0, 0.0).unwrap()],
        //     vec![Connection::build(4, 4, 0, 1.0, 0.0).unwrap()],
        // ];

        let _weights = neuron
            .memorize_periodic_spike_train(
                &firing_times,
                &mut inspikes,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                Objective::None,
            )
            .expect("Memorization failed");

        let weights = neuron
            .memorize_periodic_spike_train(
                &firing_times,
                &mut inspikes,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                Objective::L2Norm,
            )
            .expect("L2-memorization failed");

        assert_eq!(
            weights,
            vec![
                -1.40501505062582,
                0.8276421729855583,
                2.2129265840338137,
                -1.2119262246876459,
                -0.0
            ]
        );
        // assert_relative_eq!(weights[1], 1.157671, epsilon = 1e-6);
        // assert_relative_eq!(weights[2], 0.011027, epsilon = 1e-6);
        // assert_relative_eq!(weights[3], -0.415485, epsilon = 1e-6);
        // assert_relative_eq!(weights[4], 0.0);

        let weights = neuron
            .memorize_periodic_spike_train(
                &firing_times,
                &mut inspikes,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                Objective::L1Norm,
            )
            .expect("L1-memorization failed");
        assert_eq!(
            weights,
            vec![
                -2.092002954197055,
                0.8276421729856879,
                2.2129265840329473,
                -0.41548472074177123,
                0.0
            ]
        );
    }
}
