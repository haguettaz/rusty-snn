use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use rand::distributions::{Distribution, Uniform};
// use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json;

use super::connection::Input;
use super::neuron::Neuron;

#[derive(Debug, Serialize, Deserialize)]
pub struct Network {
    neurons: Vec<Neuron>, // The network owns the neurons, use slices for Inputs to other neurons.
                          // Inputs: Vec<Input>, // The network owns the Inputs, a neuron should have references to the ones it is connected to.
}

/// The SNN struct represents a spiking neural network.
impl Network {
    /// Creates a new random SNN with a given number of neurons and inputs per neuron.
    /// Can we give a generator for random delays, weights, orders, and betas???
    pub fn new_random_fin(
        num_neurons: usize,
        num_inputs: usize,
        weight_dist: impl Distribution<f64>,
        delay_dist: impl Distribution<f64>,
        order_dist: impl Distribution<i32>,
        beta_dist: impl Distribution<f64>,
    ) -> Network {
        // Init the random number generators
        let mut rng = rand::thread_rng();

        let uniform = Uniform::from(0..num_neurons);

        let mut neurons: Vec<Neuron> = Vec::new();
        for l in 0..num_neurons {
            let mut inputs: Vec<Input> = Vec::new();
            for _ in 0..num_inputs {
                inputs.push(Input::build(
                    uniform.sample(&mut rng),
                    weight_dist.sample(&mut rng),
                    delay_dist.sample(&mut rng),
                    order_dist.sample(&mut rng),
                    beta_dist.sample(&mut rng),
                ));
            }
            neurons.push(Neuron::new(l, 1.0, inputs));
        }

        Network { neurons }
    }

    pub fn save_to(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, self)?;
        writer.flush()?;
        Ok(())
    }

    pub fn load_from(path: &str) -> std::io::Result<Network> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    const NUM_NEURONS: usize = 10;
    const NUM_INPUTS: usize = 10;
    const WEIGHT_RANGE: Range<f64> = -1.0..1.0;
    const DELAY_RANGE: Range<f64> = 1.0..10.0;
    const ORDER_RANGE: Range<i32> = 1..16;
    const BETA_RANGE: Range<f64> = 0.5..2.0;

    #[test]
    fn test_network() {
        let snn = Network::new_random_fin(
            NUM_NEURONS,
            NUM_INPUTS,
            Uniform::from(WEIGHT_RANGE),
            Uniform::from(DELAY_RANGE),
            Uniform::from(ORDER_RANGE),
            Uniform::from(BETA_RANGE)
        );
        assert_eq!(snn.neurons.len(), NUM_NEURONS);
    }

    #[test]
    fn test_save_load() {
        let snn = Network::new_random_fin(
            NUM_NEURONS,
            NUM_INPUTS,
            Uniform::from(WEIGHT_RANGE),
            Uniform::from(DELAY_RANGE),
            Uniform::from(ORDER_RANGE),
            Uniform::from(BETA_RANGE)
        );
        snn.save_to("tests/network.json").unwrap();
        let snn2 = Network::load_from("tests/network.json").unwrap();
        assert_eq!(snn.neurons.len(), snn2.neurons.len());
    }
}
