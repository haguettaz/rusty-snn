use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use rand::distributions::{Distribution, Uniform};
// use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json;
use std::ops::Range;

use super::input::Input;
use super::neuron::Neuron;
use super::sampler::{IOSamplerEnum, IOSampler, UnbalancedSampler, InBalancedSampler, OutBalancedSampler, IOBalancedSampler};

pub enum BalancedType {
    Unbalanced,
    InBalanced, // fixed number of inputs per neuron
    OutBalanced, // fixed number of inputs per neuron
    IOBalanced, // fixed number of inputs and outputs per neuron
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Network {
    neurons: Vec<Neuron>, // The network owns the neurons, use slices for Connections to other neurons.
                          // Connections: Vec<Connection>, // The network owns the Connections, a neuron should have references to the ones it is connected to.
}

/// The SNN struct represents a spiking neural network.
impl Network {
    /// Creates a new random SNN with a given number of neurons and inputs per neuron.
    /// Can we give a generator for random delays, weights, orders, and betas???
    pub fn new_random(
        num_neurons: usize,
        num_connections: usize,
        weight_range: Range<f64>,
        delay_range: Range<f64>,
        order_range: Range<i32>,
        beta_range: Range<f64>,
        balanced_type: BalancedType,
    ) -> Network {

        let mut sampler: IOSamplerEnum = match balanced_type {
            BalancedType::Unbalanced => IOSamplerEnum::Unbalanced(UnbalancedSampler {
                num_neurons,
                num_connections,
            }),
            BalancedType::InBalanced => IOSamplerEnum::InBalanced(InBalancedSampler {
                num_neurons,
                num_connections,
            }),
            BalancedType::OutBalanced => IOSamplerEnum::OutBalanced(OutBalancedSampler {
                num_neurons,
                num_connections,
            }),
            BalancedType::IOBalanced => IOSamplerEnum::IOBalanced(IOBalancedSampler {
                num_neurons,
                num_connections,
            }),
        };
    
        let weight_dist = Uniform::from(weight_range);
        let delay_dist = Uniform::from(delay_range);
        let order_dist = Uniform::from(order_range);
        let beta_dist = Uniform::from(beta_range);

        let mut rng = rand::thread_rng();
        let mut neurons: Vec<Neuron> = Vec::new();
        for id in 0..num_neurons {
            let mut inputs: Vec<Input> = Vec::new();
            for (source_id, _) in sampler.iter().filter(|(_, target_id) | *target_id == id) {
                inputs.push(Input::build(
                    source_id,
                    weight_dist.sample(&mut rng),
                    delay_dist.sample(&mut rng),
                    order_dist.sample(&mut rng),
                    beta_dist.sample(&mut rng),
                ));
            }
            neurons.push(Neuron::new(id, 1.0, inputs));
        }
        // for (source_id, target_id) in sampler.iter() {
        //     connections.push(Connection::build(
        //         source_id,
        //         target_id,
        //         weight_dist.sample(&mut rng),
        //         delay_dist.sample(&mut rng),
        //         order_dist.sample(&mut rng),
        //         beta_dist.sample(&mut rng),
        //     ));
        // }

        // for id in 0..num_neurons {
        //     neurons.push(Neuron::new(id, 1.0, connections.iter().filter(|c| c.target_id == id).clone().collect()));
        // }

        Network { neurons }
    }

    // pub fn new_random_fout(
    //     num_neurons: usize,
    //     num_outputs: usize,
    //     weight_dist: impl Distribution<f64>,
    //     delay_dist: impl Distribution<f64>,
    //     order_dist: impl Distribution<i32>,
    //     beta_dist: impl Distribution<f64>,
    // ) -> Network {
    //     // Init the random number generators
    //     let mut rng = rand::thread_rng();

    //     // Generate random connections, where every neuron have a fixed number of outputs
    //     let uniform = Uniform::from(0..num_neurons);
    //     let mut connections: Vec<Connection> = Vec::new();
    //     for id in 0..num_neurons {
    //         for _ in 0..num_outputs {
    //             connections.push(Connection::build(
    //                 id,
    //                 uniform.sample(&mut rng),
    //                 weight_dist.sample(&mut rng),
    //                 delay_dist.sample(&mut rng),
    //                 order_dist.sample(&mut rng),
    //                 beta_dist.sample(&mut rng),
    //             ));
    //         }
    //     }

    //     let mut neurons: Vec<Neuron> = Vec::new();
    //     for id in 0..num_neurons {
    //         neurons.push(Neuron::new(id, 1.0, connections.iter().filter(|c| c.target_id == id).clone().collect()));
    //     }

    //     Network { neurons }
    // }

    // // pub fn new_random_fin_fout(
    // //     num_neurons: usize,
    // //     num_connections: usize, // what about a connection distribution instead???
    // //     weight_dist: impl Distribution<f64>,
    // //     delay_dist: impl Distribution<f64>,
    // //     order_dist: impl Distribution<i32>,
    // //     beta_dist: impl Distribution<f64>,
    // // ) -> Network {
    // //     // Init the random number generators
    // //     let mut rng = rand::thread_rng();

    // //     // Generate random connections, where every neuron have a fixed number of inputs AND outputs
    // //     let uniform = Uniform::from(0..num_neurons);
    // //     let mut connections: Vec<Connection> = Vec::new();
    // //     for id in 0..num_neurons {
    // //         for _ in 0..num_outputs {
    // //             connections.push(Connection::build(
    // //                 id,
    // //                 uniform.sample(&mut rng),
    // //                 weight_dist.sample(&mut rng),
    // //                 delay_dist.sample(&mut rng),
    // //                 order_dist.sample(&mut rng),
    // //                 beta_dist.sample(&mut rng),
    // //             ));
    // //         }
    // //     }

    // //     let mut neurons: Vec<Neuron> = Vec::new();
    // //     for id in 0..num_neurons {
    // //         neurons.push(Neuron::new(id, 1.0, connections.iter().filter(|c| c.target_id == id).clone().collect()));
    // //     }

    // //     Network { neurons }
    // // }


    // pub fn new_random(
    //     // num_neurons: usize,
    //     co_sampler: ConnectivityRandomIterator,
    //     // weight_dist: impl Distribution<f64>,
    //     // delay_dist: impl Distribution<f64>,
    //     // order_dist: impl Distribution<i32>,
    //     // beta_dist: impl Distribution<f64>,
    // ) -> Network {
    //     // Init the random number generators
    //     let mut rng = rand::thread_rng();


    //     let mut connections = co_sampler.iter().for_each(|(source_id, target_id, weight, delay, order, beta)| {
    //         Connection::build(
    //             *source_id,
    //             *target_id,
    //             weight,
    //             delay,
    //             order,
    //             beta
    //         )}).collect();

    //     // Generate random connections, where every neuron have a fixed number of inputs AND outputs
    //     let uniform = Uniform::from(0..num_neurons);
    //     let mut connections: Vec<Connection> = Vec::new();
    //     for id in 0..num_neurons {
    //         for _ in 0..num_outputs {
    //             connections.push(Connection::build(
    //                 id,
    //                 uniform.sample(&mut rng),
    //                 weight_dist.sample(&mut rng),
    //                 delay_dist.sample(&mut rng),
    //                 order_dist.sample(&mut rng),
    //                 beta_dist.sample(&mut rng),
    //             ));
    //         }
    //     }

    //     let mut neurons: Vec<Neuron> = Vec::new();
    //     for id in 0..num_neurons {
    //         neurons.push(Neuron::new(id, 1.0, connections.iter().filter(|c| c.target_id == id).clone().collect()));
    //     }

    //     Network { neurons }
    // }


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
    const NUM_CONNECTIONS: usize = 100;
    const WEIGHT_RANGE: Range<f64> = -1.0..1.0;
    const DELAY_RANGE: Range<f64> = 1.0..10.0;
    const ORDER_RANGE: Range<i32> = 1..16;
    const BETA_RANGE: Range<f64> = 0.5..2.0;

    #[test]
    fn test_network() {
        let snn = Network::new_random(
            NUM_NEURONS,
            NUM_CONNECTIONS,
            WEIGHT_RANGE,
            DELAY_RANGE,
            ORDER_RANGE,
            BETA_RANGE,
            BalancedType::Unbalanced,
        );
        assert_eq!(snn.neurons.len(), NUM_NEURONS);
    }

    #[test]
    fn test_save_load() {
        let snn = Network::new_random(
            NUM_NEURONS,
            NUM_CONNECTIONS,
            WEIGHT_RANGE,
            DELAY_RANGE,
            ORDER_RANGE,
            BETA_RANGE,
            BalancedType::Unbalanced,
        );
        snn.save_to("tests/network.json").unwrap();
        let snn2 = Network::load_from("tests/network.json").unwrap();
        assert_eq!(snn.neurons.len(), snn2.neurons.len());
    }
}
