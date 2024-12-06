//! Module implementing the spiking neurons.

use core::f64;

use embed_doc_image::embed_doc_image;
use lambert_w::lambert_w0;
use serde::{Deserialize, Serialize};

use crate::core::{FIRING_THRESHOLD, REFRACTORY_PERIOD};

use super::connection::Input;
use super::spike_train::SpikeTrainError;

/// Represents a spiking neuron.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Neuron {
    // id: usize,
    threshold: f64,
    firing_times: Vec<f64>,
    inputs: Vec<Input>,
    a_frozen: f64,
    b_frozen: f64,
    pos_frozen: usize,
}

impl Neuron {
    // pub fn new(id: usize) -> Self {
    pub fn new() -> Self {
        Neuron {
            // id,
            threshold: FIRING_THRESHOLD,
            firing_times: vec![],
            inputs: vec![],
            a_frozen: 0.0,
            b_frozen: 0.0,
            pos_frozen: 0,
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

    // /// Returns the ID of the neuron.
    // pub fn id(&self) -> usize {
    //     self.id
    // }

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

    /// Add an input spike to the neuron, with the specified source ID, weight, and firing time.
    /// The inputs are sorted by firing times.
    pub fn add_input(&mut self, weight: f64, firing_time: f64) {
        match self
            .inputs
            .binary_search_by(|input| input.firing_time().partial_cmp(&firing_time).unwrap())
        {
            Ok(pos) | Err(pos) => self.inputs.insert(pos, Input::new(weight, firing_time)),
        }
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
        self.inputs[self.pos_frozen..].iter().fold(
            (1_f64 - t).exp() * (t * self.a_frozen - self.b_frozen),
            |acc, input| acc + input.eval(t),
        )
    }

    /// Update neuron's frozen inputs during simulation to improve efficiency.
    /// The values of `a` and `b` for the neuron at the current time are updated, see Comsa et al. (2022).
    pub fn update_frozen_inputs(&mut self, time: f64) {
        let last_pos_frozen = self.pos_frozen;

        self.pos_frozen = self
            .inputs
            .binary_search_by(|input| {
                if input.firing_time() > time {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            })
            .unwrap_or_else(|pos| pos);

        match last_pos_frozen > self.pos_frozen {
            true => {
                self.a_frozen -= self.inputs[self.pos_frozen..last_pos_frozen]
                    .iter()
                    .map(|input| input.weight() * (input.firing_time()).exp())
                    .sum::<f64>();
                self.b_frozen -= self.inputs[self.pos_frozen..last_pos_frozen]
                    .iter()
                    .map(|input| input.weight() * input.firing_time() * (input.firing_time()).exp())
                    .sum::<f64>();
            }
            false => {
                self.a_frozen += self.inputs[last_pos_frozen..self.pos_frozen]
                    .iter()
                    .map(|input| input.weight() * (input.firing_time()).exp())
                    .sum::<f64>();
                self.b_frozen += self.inputs[last_pos_frozen..self.pos_frozen]
                    .iter()
                    .map(|input| input.weight() * input.firing_time() * (input.firing_time()).exp())
                    .sum::<f64>();
            }
        }
        println!(
            "time: {}, pos_frozen: {}, a: {}, b: {}",
            time, self.pos_frozen, self.a_frozen, self.b_frozen
        );
    }

    /// Returns the next firing time of the neuron (up to end), if any.
    pub fn next_spike(&self, end: f64) -> Option<f64> {
        if let Some(last) = self.firing_times().last() {
            if self.potential(last + REFRACTORY_PERIOD) >= self.threshold {
                return Some(last + REFRACTORY_PERIOD);
            }
            // start = start.max(last + REFRACTORY_PERIOD);
        }

        // if start >= end {
        //     return None;
        // }

        let mut firing_time = f64::NAN;
        let mut a = self.a_frozen;
        let mut b = self.b_frozen;

        for input in &self.inputs[self.pos_frozen..] {
            if firing_time <= input.firing_time() {
                break
            }

            a += input.weight() * input.firing_time().exp();
            b += input.weight() * input.firing_time() * input.firing_time().exp();

            firing_time = b / a - lambert_w0(-self.threshold() / a * (b / a - 1.0).exp());

            // is there a cleaner way to handle this noncausal edge case???
            if firing_time < input.firing_time() {
                // println!("firing_time: {}", firing_time);
                firing_time = f64::NAN;
            }
            // println!("input: {:?}, a: {}, b: {}, firing_time: {}, arg: {}", input, a, b, firing_time, -self.threshold() / a * (b / a - 1.0).exp());
        }

        if (firing_time.is_finite()) && (firing_time < end) {
            return Some(firing_time)
        }

        None

        // let candidates = self.inputs[self.pos_frozen..]
        //     .iter()
        //     .filter(|input| input.firing_time() < end)
        //     .scan((self.a_frozen, self.b_frozen, f64::NAN), |state, input| {
        //         state.0 += input.weight() * input.firing_time().exp();
        //         state.1 += input.weight() * input.firing_time() * input.firing_time().exp();
        //         state.2 = state.1 / state.0
        //             - lambert_w0(-self.threshold() * (state.1 / state.0 - 1.0).exp() / state.0); // NaN values will be filtered out later
        //         println!("a: {}, b: {}, t: {}", state.0, state.1, state.2);
        //         Some(*state)
        //     });

        // let firing_time = candidates
        //     .map(|(_, _, t)| t)
        //     .filter(|&t| t > start && t < end)
        //     .min_by(|&t1, &t2| t1.partial_cmp(&t2).expect("Comparison failed"));
        // match firing_time {
        //     Some(t) => Some(t),
        //     None => None,
        // }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    #[test]
    fn test_add_input() {
        // check that the inputs are sorted by firing times
        let mut neuron = Neuron::new();
        neuron.add_input(1.0, 1.0);
        assert_eq!(neuron.inputs[0].firing_time(), 1.0);
        neuron.add_input(1.0, 3.0);
        assert_eq!(neuron.inputs[0].firing_time(), 1.0);
        assert_eq!(neuron.inputs[1].firing_time(), 3.0);
        neuron.add_input(1.0, 2.0);
        assert_eq!(neuron.inputs[0].firing_time(), 1.0);
        assert_eq!(neuron.inputs[1].firing_time(), 2.0);
        assert_eq!(neuron.inputs[2].firing_time(), 3.0);
        neuron.add_input(1.0, 1.0);
        assert_eq!(neuron.inputs[0].firing_time(), 1.0);
        assert_eq!(neuron.inputs[1].firing_time(), 1.0);
        assert_eq!(neuron.inputs[2].firing_time(), 2.0);
        assert_eq!(neuron.inputs[3].firing_time(), 3.0);
    }

    #[test]
    fn test_neuron_extend_firing_times() {
        let mut neuron = Neuron::new();
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
        let mut neuron = Neuron::new();
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
        let mut neuron = Neuron::new();
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
        let mut neuron = Neuron::new();
        assert_eq!(neuron.threshold(), FIRING_THRESHOLD);
        assert_eq!(neuron.num_inputs(), 0);
        assert!(neuron.inputs().is_empty());
        assert!(neuron.firing_times().is_empty());

        neuron.set_threshold(1.5);
        assert_eq!(neuron.threshold(), 1.5);
    }

    #[test]
    fn test_neuron_potential() {
        let mut neuron = Neuron::new();
        neuron.add_input(1.0, 0.0);
        neuron.add_input(-1.0, 1.0);

        // Test potential at different times
        assert_eq!(neuron.potential(0.0), 0.0);
        assert_eq!(neuron.potential(1.0), 1.0);
        assert_eq!(neuron.potential(2.0), -0.26424111765711533);
    }

    #[test]
    fn test_neuron_next_spike() {
        // Test single inputs with no previous firing
        let mut neuron = Neuron::new();
        neuron.add_input(1.0, 1.0);
        assert_eq!(neuron.next_spike(10.0).unwrap(), 2.0);
        let mut neuron = Neuron::new();
        neuron.add_input(-1.0, 1.0);
        assert_eq!(neuron.next_spike(10.0), None);

        // Test next_spike with no previous firing
        let mut neuron = Neuron::new();
        neuron.add_input(1.0, 1.0);
        assert_eq!(neuron.next_spike(10.0).unwrap(), 2.0);
        neuron.add_input(-1.0, 1.0);
        assert_eq!(neuron.next_spike(10.0), None);
        neuron.add_input(1.0, 1.0);
        assert_eq!(neuron.next_spike(10.0).unwrap(), 2.0);

        // Test next_spike after refractory period
        let mut neuron = Neuron::new();
        neuron.add_firing_time(2.0).unwrap();
        neuron.add_input(10.0, 1.0);
        assert_eq!(neuron.next_spike(10.0).unwrap(), 3.0);
    }

    #[test]
    fn test_neuron_update_frozen_inputs() {
        let mut neuron = Neuron::new();
        neuron.update_frozen_inputs(2.0);
        assert_eq!(neuron.a_frozen, 0.0);
        assert_eq!(neuron.b_frozen, 0.0);
        assert_eq!(neuron.pos_frozen, 0);

        neuron.add_input(1.0, 0.0);
        neuron.update_frozen_inputs(2.0);
        assert_eq!(neuron.a_frozen, 1.0);
        assert_eq!(neuron.b_frozen, 0.0);
        assert_eq!(neuron.pos_frozen, 1);

        neuron.add_input(1.0, 2.0);
        neuron.update_frozen_inputs(2.0);
        assert_eq!(neuron.a_frozen, 1.0 + 2.0_f64.exp());
        assert_eq!(neuron.b_frozen, 2.0 * 2.0_f64.exp());
        assert_eq!(neuron.pos_frozen, 2);

        neuron.add_input(1.0, 5.0);
        neuron.update_frozen_inputs(4.0);
        assert_eq!(neuron.a_frozen, 1.0 + 2.0_f64.exp());
        assert_eq!(neuron.b_frozen, 2.0 * 2.0_f64.exp());
        assert_eq!(neuron.pos_frozen, 2);
    }

    #[test]
    fn test_neuron_step_hyperactive() {
        let mut neuron = Neuron::new();
        neuron.set_threshold(0.0);

        neuron.add_input(1.0, 0.0);

        let time = neuron.next_spike(10.0).unwrap();
        neuron.add_firing_time(time).unwrap();
        assert!(time.abs() < 1e-10);

        let time = neuron.next_spike(10.0).unwrap();
        neuron.add_firing_time(time).unwrap();
        assert!((time - 1.0).abs() < 1e-10);

        let time = neuron.next_spike(10.0).unwrap();
        neuron.add_firing_time(time).unwrap();
        assert!((time - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_neuron_step_no_firing() {
        let mut neuron = Neuron::new();
        neuron.set_threshold(1.0);
        neuron.add_input(0.0, 1.0);

        // Test when potential never reaches threshold
        let next_firing = neuron.next_spike(100.0);
        assert_eq!(next_firing, None);
    }
}
