use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::ops::Range;
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
    pub fn new_random(
        num_neurons: usize,
        num_connections: usize,
        weight_range: Range<f64>,
        delay_range: Range<f64>,
        order_range: Range<i32>,
        beta_range: Range<f64>,
        rng: &mut impl Rng,
    ) -> Network {
        let mut neurons: Vec<Neuron> = (0..num_neurons)
            .map(|id| Neuron::new(id, 1.0, Vec::new()))
            .collect();

        let weight_dist = Uniform::from(weight_range);
        let delay_dist = Uniform::from(delay_range);
        let order_dist = Uniform::from(order_range);
        let beta_dist = Uniform::from(beta_range);

        for _ in 0..num_connections {
            let src = rng.gen_range(0..num_neurons);
            let tgt = rng.gen_range(0..num_neurons);
            let weight = weight_dist.sample(rng);
            let delay = delay_dist.sample(rng);
            let order = order_dist.sample(rng);
            let beta = beta_dist.sample(rng);
            neurons[tgt].add_input(Input::build(src, weight, delay, order, beta));
        }

        Network { neurons }
    }

    /// Creates a new random Network with a specific number of neurons and connections.
    /// Each neuron sends exactly num_connections / num_neurons outputs.
    /// Example: In a network with 100 connections and 10 neurons, each neuron sends exactly 10 outputs.    
    pub fn new_random_fout(
        num_neurons: usize,
        num_connections: usize,
        weight_range: Range<f64>,
        delay_range: Range<f64>,
        order_range: Range<i32>,
        beta_range: Range<f64>,
        rng: &mut impl Rng,
    ) -> Network {
        let mut neurons: Vec<Neuron> = (0..num_neurons)
            .map(|id| Neuron::new(id, 1.0, Vec::new()))
            .collect();

        let weight_dist = Uniform::from(weight_range);
        let delay_dist = Uniform::from(delay_range);
        let order_dist = Uniform::from(order_range);
        let beta_dist = Uniform::from(beta_range);

        for k in 0..num_connections {
            let src = k % num_neurons;
            let tgt = rng.gen_range(0..num_neurons);
            let weight = weight_dist.sample(rng);
            let delay = delay_dist.sample(rng);
            let order = order_dist.sample(rng);
            let beta = beta_dist.sample(rng);
            neurons[tgt].add_input(Input::build(src, weight, delay, order, beta));
        }

        Network { neurons }
    }

    /// Creates a new random Network with a specific number of neurons and connections.
    /// Each neuron receives exactly num_connections / num_neurons inputs.
    /// Example: In a network with 100 connections and 10 neurons, each neuron receives exactly 10 connections.    
    pub fn new_random_fin(
        num_neurons: usize,
        num_connections: usize,
        weight_range: Range<f64>,
        delay_range: Range<f64>,
        order_range: Range<i32>,
        beta_range: Range<f64>,
        rng: &mut impl Rng,
    ) -> Network {
        let mut neurons: Vec<Neuron> = (0..num_neurons)
            .map(|id| Neuron::new(id, 1.0, Vec::new()))
            .collect();

        let weight_dist = Uniform::from(weight_range);
        let delay_dist = Uniform::from(delay_range);
        let order_dist = Uniform::from(order_range);
        let beta_dist = Uniform::from(beta_range);

        for k in 0..num_connections {
            let src = rng.gen_range(0..num_neurons);
            let tgt = k % num_neurons;
            let weight = weight_dist.sample(rng);
            let delay = delay_dist.sample(rng);
            let order = order_dist.sample(rng);
            let beta = beta_dist.sample(rng);
            neurons[tgt].add_input(Input::build(src, weight, delay, order, beta));
        }

        Network { neurons }
    }

    /// Creates a new random Network with a specific number of neurons and connections.
    /// Each neuron both receives and sends exactly num_connections / num_neurons connections.
    /// Example: In a network with 100 connections and 10 neurons, each neuron receives and sends exactly 10 connections.
    pub fn new_random_fin_fout(
        num_neurons: usize,
        num_connections: usize,
        weight_range: Range<f64>,
        delay_range: Range<f64>,
        order_range: Range<i32>,
        beta_range: Range<f64>,
        rng: &mut impl Rng,
    ) -> Network {
        if num_connections % num_neurons != 0 {
            panic!("Number of connections must be divisible by number of neurons for the network to be perfectly in/out balanced.");
        }

        let mut neurons: Vec<Neuron> = (0..num_neurons)
            .map(|id| Neuron::new(id, 1.0, Vec::new()))
            .collect();

        let weight_dist = Uniform::from(weight_range);
        let delay_dist = Uniform::from(delay_range);
        let order_dist = Uniform::from(order_range);
        let beta_dist = Uniform::from(beta_range);

        let mut target_ids = (0..num_neurons)
            .cycle()
            .take(num_connections)
            .collect::<Vec<usize>>();
        target_ids.shuffle(rng);
        for k in 0..num_connections {
            let src = k % num_neurons;
            let tgt = target_ids[k];
            let weight = weight_dist.sample(rng);
            let delay = delay_dist.sample(rng);
            let order = order_dist.sample(rng);
            let beta = beta_dist.sample(rng);
            neurons[tgt].add_input(Input::build(src, weight, delay, order, beta));
        }
        Network { neurons }
    }

    pub fn save_to<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, self)?;
        writer.flush()?;
        Ok(())
    }

    pub fn load_from<P: AsRef<Path>>(path: P) -> std::io::Result<Network> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use std::ops::Range;

    const NUM_NEURONS: usize = 3;
    const NUM_CONNECTIONS: usize = 6;
    const WEIGHT_RANGE: Range<f64> = -1.0..1.0;
    const DELAY_RANGE: Range<f64> = 1.0..10.0;
    const ORDER_RANGE: Range<i32> = 1..16;
    const BETA_RANGE: Range<f64> = 0.5..2.0;

    #[test]
    fn test_unbalanced_network() {
        let seed = [1; 32];
        let mut rng = StdRng::from_seed(seed);

        let network = Network::new_random(
            NUM_NEURONS,
            NUM_CONNECTIONS,
            WEIGHT_RANGE,
            DELAY_RANGE,
            ORDER_RANGE,
            BETA_RANGE,
            &mut rng,
        );

        assert_eq!(network.neurons.len(), NUM_NEURONS);
        assert_eq!(
            network.neurons.iter().flat_map(|n| n.inputs()).count(),
            NUM_CONNECTIONS
        );
        for id in 0..NUM_NEURONS {
            assert_eq!(network.neurons[id].id(), id);
            for input in network.neurons[id].inputs().iter() {
                assert!(input.source_id() < NUM_NEURONS);
                assert!(input.weight() >= WEIGHT_RANGE.start);
                assert!(input.weight() < WEIGHT_RANGE.end);
                assert!(input.delay() >= DELAY_RANGE.start);
                assert!(input.delay() < DELAY_RANGE.end);
                assert!(input.kernel().order() >= ORDER_RANGE.start);
                assert!(input.kernel().order() < ORDER_RANGE.end);
                assert!(input.kernel().beta() >= BETA_RANGE.start);
                assert!(input.kernel().beta() < BETA_RANGE.end);
            }
        }
    }

    #[test]
    fn test_in_out_balanced_network() {
        let seed = [1; 32];
        let mut rng = StdRng::from_seed(seed);

        let network = Network::new_random_fin_fout(
            NUM_NEURONS,
            NUM_CONNECTIONS,
            WEIGHT_RANGE,
            DELAY_RANGE,
            ORDER_RANGE,
            BETA_RANGE,
            &mut rng,
        );

        assert_eq!(network.neurons.len(), NUM_NEURONS);
        for id in 0..NUM_NEURONS {
            assert_eq!(network.neurons[id].id(), id);
            assert_eq!(
                network
                    .neurons
                    .iter()
                    .flat_map(|n| n.inputs())
                    .filter(|i| i.source_id() == id)
                    .count(),
                NUM_CONNECTIONS / NUM_NEURONS
            );
            assert_eq!(
                network.neurons[id].inputs().len(),
                NUM_CONNECTIONS / NUM_NEURONS
            );
            for input in network.neurons[id].inputs().iter() {
                assert!(input.source_id() < NUM_NEURONS);
                assert!(input.weight() >= WEIGHT_RANGE.start);
                assert!(input.weight() < WEIGHT_RANGE.end);
                assert!(input.delay() >= DELAY_RANGE.start);
                assert!(input.delay() < DELAY_RANGE.end);
                assert!(input.kernel().order() >= ORDER_RANGE.start);
                assert!(input.kernel().order() < ORDER_RANGE.end);
                assert!(input.kernel().beta() >= BETA_RANGE.start);
                assert!(input.kernel().beta() < BETA_RANGE.end);
            }
        }
    }

    #[test]
    fn test_in_balanced_network() {
        let seed = [1; 32];
        let mut rng = StdRng::from_seed(seed);

        let network = Network::new_random_fin(
            NUM_NEURONS,
            NUM_CONNECTIONS,
            WEIGHT_RANGE,
            DELAY_RANGE,
            ORDER_RANGE,
            BETA_RANGE,
            &mut rng,
        );

        assert_eq!(network.neurons.len(), NUM_NEURONS);
        for id in 0..NUM_NEURONS {
            assert_eq!(network.neurons[id].id(), id);
            assert_eq!(
                network.neurons[id].inputs().len(),
                NUM_CONNECTIONS / NUM_NEURONS
            );
            for input in network.neurons[id].inputs().iter() {
                assert!(input.source_id() < NUM_NEURONS);
                assert!(input.weight() >= WEIGHT_RANGE.start);
                assert!(input.weight() < WEIGHT_RANGE.end);
                assert!(input.delay() >= DELAY_RANGE.start);
                assert!(input.delay() < DELAY_RANGE.end);
                assert!(input.kernel().order() >= ORDER_RANGE.start);
                assert!(input.kernel().order() < ORDER_RANGE.end);
                assert!(input.kernel().beta() >= BETA_RANGE.start);
                assert!(input.kernel().beta() < BETA_RANGE.end);
            }
        }
    }

    #[test]
    fn test_out_balanced_network() {
        let seed = [1; 32];
        let mut rng = StdRng::from_seed(seed);

        let network = Network::new_random_fout(
            NUM_NEURONS,
            NUM_CONNECTIONS,
            WEIGHT_RANGE,
            DELAY_RANGE,
            ORDER_RANGE,
            BETA_RANGE,
            &mut rng,
        );

        assert_eq!(network.neurons.len(), NUM_NEURONS);
        for id in 0..NUM_NEURONS {
            assert_eq!(network.neurons[id].id(), id);
            assert_eq!(
                network
                    .neurons
                    .iter()
                    .flat_map(|n| n.inputs())
                    .filter(|i| i.source_id() == id)
                    .count(),
                NUM_CONNECTIONS / NUM_NEURONS
            );
            for input in network.neurons[id].inputs().iter() {
                assert!(input.source_id() < NUM_NEURONS);
                assert!(input.weight() >= WEIGHT_RANGE.start);
                assert!(input.weight() < WEIGHT_RANGE.end);
                assert!(input.delay() >= DELAY_RANGE.start);
                assert!(input.delay() < DELAY_RANGE.end);
                assert!(input.kernel().order() >= ORDER_RANGE.start);
                assert!(input.kernel().order() < ORDER_RANGE.end);
                assert!(input.kernel().beta() >= BETA_RANGE.start);
                assert!(input.kernel().beta() < BETA_RANGE.end);
            }
        }
    }

    #[test]
    fn test_clone() {
        let seed = [1; 32];
        let mut rng = StdRng::from_seed(seed);
        let network = Network::new_random(
            NUM_NEURONS,
            NUM_CONNECTIONS,
            WEIGHT_RANGE,
            DELAY_RANGE,
            ORDER_RANGE,
            BETA_RANGE,
            &mut rng,
        );
        let cloned_network = network.clone();
        assert_eq!(cloned_network, network);
    }
}
