//! A Neuron is a basic building block of the spiking neural network.
//! 
//! # Properties
//! * Connected to other neurons through Inputs
//! * Has a threshold that potential must reach to fire
//! * Accumulates potential over time

use super::input::Input;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Neuron {
    /// Unique identifier for the neuron
    id: usize,
    /// Minimum potential required for the neuron to fire
    threshold: f64,
    /// Historical record of times when the neuron fired
    firing_times: Vec<f64>,
    /// Collection of inputs connected to this neuron
    inputs: Vec<Input> // or slice?
}

impl Neuron {
    /// Creates a new Neuron with the specified parameters.
    /// 
    /// # Arguments
    /// * `id` - Unique identifier for the neuron
    /// * `threshold` - Minimum potential required for the neuron to fire
    /// * `inputs` - Collection of inputs to this neuron
    pub fn new(id:usize, threshold: f64, inputs: Vec<Input>) -> Neuron {
        Neuron {
            id: id,
            threshold: threshold,
            firing_times: Vec::new(),
            inputs: inputs,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn add_input(&mut self, input: Input) {
        self.inputs.push(input);
    }

    pub fn inputs(&self) -> &Vec<Input> {
        &self.inputs
    }

    pub fn firing_times(&self) -> &Vec<f64> {
        &self.firing_times
    }

    pub fn potential(&self, time: f64) -> f64 {
        self.inputs
            .iter()
            .map(|input| input.apply(time))
            .sum()
    }

    pub fn fire(&mut self, time: f64) {
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
    fn test_fire() {
        let mut neuron = Neuron::new(0, 1.0, Vec::new());
        neuron.fire(0.0);
        assert_eq!(neuron.firing_times.len(), 1);
        neuron.fire(1.45);
        assert_eq!(neuron.firing_times.len(), 2);
    }

    #[test]
    fn test_firing_times_accessor() {
        let mut neuron = Neuron::new(0, 1.0, Vec::new());
        neuron.fire(1.0);
        neuron.fire(2.0);
        assert_eq!(neuron.firing_times(), &vec![1.0, 2.0]);
    }
}