//! A Neuron is a basic building block of the spiking neural network.

use super::input::Input;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Neuron {
    /// Unique identifier for the neuron
    id: usize,
    /// Minimum potential required for the neuron to fire
    threshold: f64,
    /// Historical record of times when the neuron fired
    firing_times: Vec<f64>,
    /// Collection of inputs connected to this neuron
    inputs: Vec<Input>,
}

impl Neuron {
    /// Creates a new Neuron with the specified parameters.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the neuron
    /// * `threshold` - Minimum potential required for the neuron to fire
    /// * `inputs` - Collection of inputs to this neuron
    pub fn new(id: usize, threshold: f64, inputs: Vec<Input>) -> Self {
        Neuron {
            id,
            threshold,
            inputs,
            firing_times: Vec::new(),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn add_input(&mut self, input: Input) {
        self.inputs.push(input);
    }

    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    pub fn firing_times(&self) -> &[f64] {
        &self.firing_times[..]
    }

    /// Calculates the neuron's potential at a given time by summing the contributions from its inputs.
    ///
    /// # Arguments
    ///
    /// * `time` - The time at which to calculate the potential.
    ///
    /// # Returns
    ///
    /// * `f64` - The total potential of the neuron at the given time.
    pub fn potential(&self, time: f64) -> f64 {
        self.inputs.iter().map(|input| input.apply(time)).sum()
    }

    pub fn add_firing_time(&mut self, time: f64) {
        self.firing_times.push(time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron() {
        let neuron = Neuron::new(0, 1.0, Vec::new());
        assert_eq!(neuron.id, 0);
        assert_eq!(neuron.threshold, 1.0);
        assert_eq!(neuron.firing_times.len(), 0);
    }

    #[test]
    fn test_add_firing_time() {
        let mut neuron = Neuron::new(0, 1.0, Vec::new());
        neuron.add_firing_time(0.0);
        assert_eq!(neuron.firing_times.len(), 1);
        neuron.add_firing_time(1.45);
        assert_eq!(neuron.firing_times.len(), 2);
    }

    #[test]
    fn test_firing_times_accessor() {
        let mut neuron = Neuron::new(0, 1.0, Vec::new());
        neuron.add_firing_time(1.0);
        neuron.add_firing_time(2.0);
        assert_eq!(neuron.firing_times(), &vec![1.0, 2.0]);
    }

    #[test]
    fn test_clone() {
        let neuron = Neuron::new(0, 1.0, Vec::new());
        let cloned_neuron = neuron.clone();
        assert_eq!(neuron, cloned_neuron);
    }
}
