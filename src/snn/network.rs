

use std::fs::File;
use std::io::{Write, Read, BufWriter, BufReader};

use rand::distributions::{Distribution, Uniform};
// use rand::Rng;
use serde::{Serialize, Deserialize};
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
        delay_dist: impl Distribution<f64>,
        weight_dist: impl Distribution<f64>,
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
                inputs.push(Input::new(
                    uniform.sample(&mut rng),
                    delay_dist.sample(&mut rng),
                    weight_dist.sample(&mut rng),
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
        serde_json::from_reader(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network() {
        let snn = Network::new_random_fin(10, 10,Uniform::from(0.0..1.0), Uniform::from(0.0..1.0), Uniform::from(0..10), Uniform::from(0.0..1.0));
        assert_eq!(snn.neurons.len(), 10);
    }

    #[test]
    fn test_save_load() {
        let snn = Network::new_random_fin(10, 10,Uniform::from(0.0..1.0), Uniform::from(0.0..1.0), Uniform::from(0..10), Uniform::from(0.0..1.0));
        snn.save_to("tests/network.json").unwrap();
        let snn2 = Network::load_from("tests/network.json").unwrap();
        assert_eq!(snn.neurons.len(), snn2.neurons.len());
    }
}
