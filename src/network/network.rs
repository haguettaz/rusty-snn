use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use super::input::Input;
use super::neuron::Neuron;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Network {
    neurons: Vec<Neuron>, // The network owns the neurons, use slices for Connections to other neurons.
}

/// The Network struct represents a spiking neural network.
impl Network {
    /// Creates a new random Network with a specific number of neurons and connections.
    pub fn new_random<F: Distribution<f64>, P: Distribution<i32>, R: Rng>(
        num_neurons: usize,
        num_connections: usize,
        weight_dist: &F,
        delay_dist: &F,
        beta_dist: &F,
        order_dist: &P,
        rng: &mut R,
    ) -> Result<Self, &'static str> {
        let mut neurons: Vec<Neuron> = (0..num_neurons)
            .map(|id| Neuron::new(id, 1.0, Vec::new()))
            .collect();

        for _ in 0..num_connections {
            let src = rng.gen_range(0..num_neurons);
            let tgt = rng.gen_range(0..num_neurons);
            let weight = rng.sample(weight_dist);
            let delay = rng.sample(delay_dist);
            let order = rng.sample(order_dist);
            let beta = rng.sample(beta_dist);
            neurons[tgt].add_input(Input::build(src, weight, delay, order, beta)?);
        }

        Ok(Network { neurons })
    }

    /// Creates a new random Network with a specific number of neurons and connections.
    /// Each neuron sends exactly num_connections / num_neurons outputs.
    /// Example: In a network with 100 connections and 10 neurons, each neuron sends exactly 10 outputs.    
    pub fn new_random_fout<F: Distribution<f64>, P: Distribution<i32>, R: Rng>(
        num_neurons: usize,
        num_connections: usize,
        weight_dist: &F,
        delay_dist: &F,
        beta_dist: &F,
        order_dist: &P,
        rng: &mut R,
    ) -> Result<Self, &'static str> {
        if num_connections % num_neurons != 0 {
            return Err("Number of connections must be divisible by number of neurons for the network to be perfectly out-balanced.");
        }

        let mut neurons: Vec<Neuron> = (0..num_neurons)
            .map(|id| Neuron::new(id, 1.0, Vec::new()))
            .collect();

        for k in 0..num_connections {
            let src = k % num_neurons;
            let tgt = rng.gen_range(0..num_neurons);
            let weight = rng.sample(weight_dist);
            let delay = rng.sample(delay_dist);
            let order = rng.sample(order_dist);
            let beta = rng.sample(beta_dist);
            neurons[tgt].add_input(Input::build(src, weight, delay, order, beta)?);
        }

        Ok(Network { neurons })
    }

    /// Creates a new random Network with a specific number of neurons and connections.
    /// Each neuron receives exactly num_connections / num_neurons inputs.
    /// Example: In a network with 100 connections and 10 neurons, each neuron receives exactly 10 connections.    
    pub fn new_random_fin<F: Distribution<f64>, P: Distribution<i32>, R: Rng>(
        num_neurons: usize,
        num_connections: usize,
        weight_dist: &F,
        delay_dist: &F,
        beta_dist: &F,
        order_dist: &P,
        rng: &mut R,
    ) -> Result<Self, &'static str> {
        if num_connections % num_neurons != 0 {
            return Err("Number of connections must be divisible by number of neurons for the network to be perfectly in-balanced.");
        }

        let mut neurons: Vec<Neuron> = (0..num_neurons)
            .map(|id| Neuron::new(id, 1.0, Vec::new()))
            .collect();

        for k in 0..num_connections {
            let src = rng.gen_range(0..num_neurons);
            let tgt = k % num_neurons;
            let weight = rng.sample(weight_dist);
            let delay = rng.sample(delay_dist);
            let order = rng.sample(order_dist);
            let beta = rng.sample(beta_dist);
            neurons[tgt].add_input(Input::build(src, weight, delay, order, beta)?);
        }

        Ok(Network { neurons })
    }

    /// Creates a new random Network with a specific number of neurons and connections.
    /// Each neuron both receives and sends exactly num_connections / num_neurons connections.
    /// Example: In a network with 100 connections and 10 neurons, each neuron receives and sends exactly 10 connections.
    pub fn new_random_fin_fout<F: Distribution<f64>, P: Distribution<i32>, R: Rng>(
        num_neurons: usize,
        num_connections: usize,
        weight_dist: &F,
        delay_dist: &F,
        beta_dist: &F,
        order_dist: &P,
        rng: &mut R,
    ) -> Result<Self, &'static str> {
        if num_connections % num_neurons != 0 {
            return Err("Number of connections must be divisible by number of neurons for the network to be perfectly in/out-balanced.");
        }

        let mut neurons: Vec<Neuron> = (0..num_neurons)
            .map(|id| Neuron::new(id, 1.0, Vec::new()))
            .collect();

        let mut target_ids = (0..num_neurons)
            .cycle()
            .take(num_connections)
            .collect::<Vec<usize>>();
        target_ids.shuffle(rng);
        for k in 0..num_connections {
            let src = k % num_neurons;
            let tgt = target_ids[k];
            let weight = rng.sample(weight_dist);
            let delay = rng.sample(delay_dist);
            let order = rng.sample(order_dist);
            let beta = rng.sample(beta_dist);
            neurons[tgt].add_input(Input::build(src, weight, delay, order, beta)?);
        }
        Ok(Network { neurons })
    }

    pub fn save_to<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, self)?;
        writer.flush()?;
        Ok(())
    }

    pub fn load_from<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Uniform;
    use rand::{rngs::StdRng, SeedableRng};

    fn validate_num_outputs(network: &Network, num_outputs: usize) {
        for neuron in network.neurons.iter() {
            assert_eq!(
                network
                    .neurons
                    .iter()
                    .flat_map(|n| n.inputs())
                    .filter(|i| i.source_id() == neuron.id())
                    .count(),
                num_outputs
            );
        }
    }

    fn validate_num_inputs(network: &Network, num_inputs: usize) {
        for neuron in network.neurons.iter() {
            assert_eq!(neuron.inputs().len(), num_inputs);
        }
    }

    #[test]
    fn test_network_connectivity() {
        let weight_dist = Uniform::new_inclusive(-0.1, 0.1);
        let delay_dist = Uniform::new_inclusive(0.1, 10.0);
        let order_dist = Uniform::new_inclusive(1, 8);
        let beta_dist = Uniform::new_inclusive(0.1, 2.0);
        let mut rng = StdRng::seed_from_u64(42);

        let network = Network::new_random_fout(
            3,
            12,
            &weight_dist,
            &delay_dist,
            &beta_dist,
            &order_dist,
            &mut rng,
        )
        .unwrap();
        validate_num_outputs(&network, 4);

        let network = Network::new_random_fin(
            3,
            12,
            &weight_dist,
            &delay_dist,
            &beta_dist,
            &order_dist,
            &mut rng,
        )
        .unwrap();
        validate_num_inputs(&network, 4);

        let network = Network::new_random_fin_fout(
            3,
            12,
            &weight_dist,
            &delay_dist,
            &beta_dist,
            &order_dist,
            &mut rng,
        )
        .unwrap();
        validate_num_inputs(&network, 4);
        validate_num_outputs(&network, 4);
    }

    #[test]
    fn test_network_params() {
        let mut rng = StdRng::seed_from_u64(42);

        let weight_dist = Uniform::new_inclusive(-1.0, 1.0);
        let order_dist = Uniform::new_inclusive(1, 8);
        let beta_dist = Uniform::new_inclusive(1.0, 10.0);

        let delay_dist = Uniform::new_inclusive(-1.0, 0.0);
        assert_eq!(
            Err("Delay must be positive."),
            Network::new_random(
                3,
                12,
                &weight_dist,
                &delay_dist,
                &beta_dist,
                &order_dist,
                &mut rng
            )
        );

        let delay_dist = Uniform::new_inclusive(1.0, 10.0);
        let order_dist = Uniform::new_inclusive(-8, -1);
        assert_eq!(
            Err("Order must be positive."),
            Network::new_random(
                3,
                12,
                &weight_dist,
                &delay_dist,
                &beta_dist,
                &order_dist,
                &mut rng
            )
        );

        let order_dist = Uniform::new_inclusive(1, 8);
        let beta_dist = Uniform::new_inclusive(-1.0, 0.0);
        assert_eq!(
            Err("Beta must be positive."),
            Network::new_random(
                3,
                12,
                &weight_dist,
                &delay_dist,
                &beta_dist,
                &order_dist,
                &mut rng
            )
        );
    }

    #[test]
    fn test_clone() {
        let weight_dist = Uniform::new_inclusive(-0.1, 0.1);
        let delay_dist = Uniform::new_inclusive(0.1, 10.0);
        let order_dist = Uniform::new_inclusive(1, 8);
        let beta_dist = Uniform::new_inclusive(0.1, 2.0);
        let mut rng = StdRng::seed_from_u64(42);

        let network = Network::new_random(
            3,
            12,
            &weight_dist,
            &delay_dist,
            &beta_dist,
            &order_dist,
            &mut rng,
        )
        .unwrap();
        let cloned_network = network.clone();
        assert_eq!(cloned_network, network);
    }
}
