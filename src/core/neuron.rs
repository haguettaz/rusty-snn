//! Module implementing the spiking neurons.

use embed_doc_image::embed_doc_image;
use lambert_w::lambert_w0;
use serde::{Deserialize, Serialize};

use crate::core::{FIRING_THRESHOLD, REFRACTORY_PERIOD};

use super::connection::Input;
use super::spike_train::SpikeTrainError;

/// Represents a spiking neuron.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Neuron {
    id: usize,
    threshold: f64,
    firing_times: Vec<f64>,
    inputs: Vec<Input>,
}

impl Neuron {
    pub fn new(id: usize) -> Self {
        Neuron {
            id,
            threshold: FIRING_THRESHOLD,
            firing_times: vec![],
            inputs: vec![],
        }
    }

    /// Extend the neuron's firing times with new ones.
    /// If necessary, the provided firing times are sorted before being added.
    /// The function returns an error if the refractory period is violated.
    pub fn extend_firing_times(&mut self, firing_times: &[f64]) -> Result<(), SpikeTrainError> {
        for t in firing_times {
            if !t.is_finite() {
                return Err(SpikeTrainError::InvalidTimes);
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
                return Err(SpikeTrainError::RefractoryPeriodViolation {
                    t1: ts[0],
                    t2: ts[1],
                });
            }
        }

        if let (Some(&first), Some(&last)) = (firing_times.first(), self.firing_times.last()) {
            if first <= last + REFRACTORY_PERIOD {
                return Err(SpikeTrainError::RefractoryPeriodViolation {
                    t1: last,
                    t2: first,
                });
            }
        }

        self.firing_times.extend(firing_times);
        Ok(())
    }

    /// Add a firing time to the neuron's firing times.
    /// The function returns an error if the refractory period is violated.
    pub fn add_firing_time(&mut self, t: f64) -> Result<(), SpikeTrainError> {
        if let Some(&last) = self.firing_times.last() {
            if t < last + REFRACTORY_PERIOD {
                return Err(SpikeTrainError::RefractoryPeriodViolation { t1: last, t2: t });
            }
        }

        self.firing_times.push(t);
        Ok(())
    }

    /// Make the neuron fire at the specified time and update the threshold.
    pub fn fires(&mut self, t: f64, threshold_noise: f64) -> Result<(), SpikeTrainError> {
        self.add_firing_time(t)?;
        self.threshold = FIRING_THRESHOLD + threshold_noise;
        Ok(())
    }

    /// Returns the ID of the neuron.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the threshold of the neuron.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Set the threshold of the neuron.
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    /// Returns the number of inputs of the neuron.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Add an input to the neuron with the specified parameters.
    /// If the target neuron of the connection is not the neuron, returns an error.
    pub fn add_input(&mut self, source_id: usize, weight: f64, firing_time: f64) {
        let pos = self
            .inputs
            .binary_search_by(|input| {
                input
                    .firing_time()
                    .partial_cmp(&firing_time)
                    .expect("Comparison failed")
            })
            .unwrap_or_else(|pos| pos);
        self.inputs
            .insert(pos, Input::new(source_id, weight, firing_time));
    }

    /// Returns a slice of inputs of the neuron.
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    /// Returns a slice of firing times of the neuron.
    pub fn firing_times(&self) -> &[f64] {
        &self.firing_times[..]
    }

    /// Calculates the neuron's potential at a given time by summing the contributions from its inputs.
    /// When the potential exceeds the threshold, the neuron fires.
    ///
    /// ![A Foobaring][neuron]
    ///
    #[embed_doc_image("neuron", "images/neuron.svg")]
    pub fn potential(&self, t: f64) -> f64 {
        self.inputs.iter().map(|input| input.eval(t)).sum()
    }

    /// Returns the next firing time, if any, of the neuron in the window [start, end).
    pub fn step(&self, start: &f64, end: &f64) -> Option<(usize, f64)> {
        let mut start = *start;
        if let Some(&last) = self.firing_times().last() {
            if self.potential(last + REFRACTORY_PERIOD) >= self.threshold {
                return Some((self.id, last + REFRACTORY_PERIOD));
            }
            start = start.max(last + REFRACTORY_PERIOD);
        }

        if start >= *end {
            return None;
        }

        let candidates = self.inputs().iter().scan((0.0, 0.0, f64::NAN), |state, input| {
            state.0 += input.weight() * input.firing_time().exp();
            state.1 += input.weight() * input.firing_time() * input.firing_time().exp();
            state.2 = state.1 / state.0
                - lambert_w0(- self.threshold()/state.0 * (state.1 / state.0).exp());
            Some(*state)
        });

        let firing_time = candidates
            .map(|state| state.2)
            .filter(|&t| t > start && t < *end)
            .min_by(|&t1, &t2| t1.partial_cmp(&t2).expect("Comparison failed"));
        match firing_time {
            Some(t) => Some((self.id, t)),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;
    
    #[test]
    fn test_add_input() {
        // check that the inputs are sorted by firing times
        let mut neuron = Neuron::new(0);
        neuron.add_input(1, 1.0, 1.0);
        assert_eq!(neuron.inputs[0].firing_time(), 1.0);
        neuron.add_input(1, 1.0, 3.0);
        assert_eq!(neuron.inputs[0].firing_time(), 1.0);
        assert_eq!(neuron.inputs[1].firing_time(), 3.0);
        neuron.add_input(1, 1.0, 2.0);
        assert_eq!(neuron.inputs[0].firing_time(), 1.0);
        assert_eq!(neuron.inputs[1].firing_time(), 2.0);
        assert_eq!(neuron.inputs[2].firing_time(), 3.0);
    }

    #[test]
    fn test_neuron_extend_firing_times() {
        let mut neuron = Neuron::new(0);
        assert_eq!(neuron.extend_firing_times(&[0.0, 3.0, 7.0]), Ok(()));
        assert_eq!(neuron.firing_times, [0.0, 3.0, 7.0]);
        assert_eq!(
            neuron.extend_firing_times(&[6.0]),
            Err(SpikeTrainError::RefractoryPeriodViolation { t1: 7.0, t2: 6.0 })
        );
        assert_eq!(neuron.firing_times, [0.0, 3.0, 7.0]);
        assert_eq!(neuron.extend_firing_times(&[10.0, 12.0]), Ok(()));
        assert_eq!(neuron.firing_times, [0.0, 3.0, 7.0, 10.0, 12.0]);
    }

    #[test]
    fn test_neuron_add_firing_time() {
        let mut neuron = Neuron::new(0);
        assert_eq!(neuron.add_firing_time(0.0), Ok(()));
        assert_eq!(neuron.firing_times, [0.0]);
        assert_eq!(neuron.add_firing_time(7.0), Ok(()));
        assert_eq!(neuron.firing_times, [0.0, 7.0]);
        assert_eq!(
            neuron.add_firing_time(5.0),
            Err(SpikeTrainError::RefractoryPeriodViolation { t1: 7.0, t2: 5.0 })
        );
        assert_eq!(neuron.firing_times, [0.0, 7.0]);
    }

    #[test]
    fn test_neuron_fires() {
        let mut neuron = Neuron::new(0);
        assert_eq!(neuron.fires(0.0, 0.0), Ok(()));
        assert_eq!(neuron.firing_times, [0.0]);
        assert_eq!(neuron.threshold, FIRING_THRESHOLD);
        assert_eq!(neuron.fires(7.0, 0.25), Ok(()));
        assert_eq!(neuron.firing_times, [0.0, 7.0]);
        assert_eq!(neuron.threshold, FIRING_THRESHOLD + 0.25);
        assert_eq!(
            neuron.fires(5.0, 0.0),
            Err(SpikeTrainError::RefractoryPeriodViolation { t1: 7.0, t2: 5.0 })
        );
        assert_eq!(neuron.firing_times, [0.0, 7.0]);
        assert_eq!(neuron.threshold, FIRING_THRESHOLD + 0.25);
    }

    #[test]
    fn test_neuron_getters() {
        let mut neuron = Neuron::new(42);
        assert_eq!(neuron.id(), 42);
        assert_eq!(neuron.threshold(), FIRING_THRESHOLD);
        assert_eq!(neuron.num_inputs(), 0);
        assert!(neuron.inputs().is_empty());
        assert!(neuron.firing_times().is_empty());
        
        neuron.set_threshold(1.5);
        assert_eq!(neuron.threshold(), 1.5);
    }

    #[test]
    fn test_neuron_potential() {
        let mut neuron = Neuron::new(0);
        neuron.add_input(1, 1.0, 0.0);
        neuron.add_input(2, -1.0, 1.0);
        
        // Test potential at different times
        assert!((neuron.potential(0.0) - 0.0).abs() < 1e-10);
        assert!((neuron.potential(1.0) - (-1_f64).exp()).abs() < 1e-10);
        assert!((neuron.potential(2.0) - (2_f64*(-2_f64).exp() - (-1_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_neuron_step() {
        let mut neuron = Neuron::new(0);
        neuron.set_threshold(1_f64/E);
        neuron.add_input(1, 1.0, 1.0);
        
        // Test step with no previous firing
        let (id, time) = neuron.step(&0.0, &3.0).unwrap();
        assert_eq!(id, 0);
        assert_eq!(time, 2.0);

        // Test step after refractory period
        neuron.add_firing_time(2.0).unwrap();
        neuron.add_input(1, 10.0, 2.0);
        let (id, time) = neuron.step(&0.0, &3.0).unwrap();
        assert_eq!(id, 0);
        assert_eq!(time, 3.0);
    }

    #[test]
    fn test_neuron_step_hyperactive() {
        let mut neuron = Neuron::new(0);
        neuron.set_threshold(0.0);
        neuron.add_input(1, 1.0, 0.0);
        
        let (id, time) = neuron.step(&0.0, &10.0).unwrap();
        neuron.add_firing_time(time).unwrap();
        assert_eq!(id, 0);
        assert!((time).abs() < 1e-10);

        let (id, time) = neuron.step(&0.0, &10.0).unwrap();
        neuron.add_firing_time(time).unwrap();
        assert_eq!(id, 0);
        assert!((time-1.0).abs() < 1e-10);
        
        let (id, time) = neuron.step(&0.0, &10.0).unwrap();
        neuron.add_firing_time(time).unwrap();
        assert_eq!(id, 0);
        assert!((time-2.0).abs() < 1e-10);
    }

    #[test]
    fn test_neuron_step_no_firing() {
        let mut neuron = Neuron::new(0);
        neuron.set_threshold(1.0);
        neuron.add_input(1, 0.0, 1.0);
        
        // Test when potential never reaches threshold
        let next_firing = neuron.step(&0.0, &100.0);
        assert_eq!(next_firing, None);
    }
}
