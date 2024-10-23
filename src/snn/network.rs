use rand::distributions::{Distribution, Uniform};
// use rand::Rng;

use super::connection::Input;
use super::neuron::Neuron;

pub struct Network {
    neurons: Vec<Neuron>, // The network owns the neurons, use slices for Inputs to other neurons.
                          // Inputs: Vec<Input>, // The network owns the Inputs, a neuron should have references to the ones it is connected to.
}

/// The SNN struct represents a spiking neural network.
impl Network {
    /// Creates a new random SNN with a given number of neurons and inputs per neuron.
    /// Can we give a generator for random delays, weights, orders, and betas???
    pub fn new_random_fin(num_neurons: usize, num_inputs: usize) -> Network {
        // Init the random number generators to setup the random Inputs
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0..num_neurons);

        // Create a vector of Inputs with fixed delays, weights, orders, and betas (for now)
        // let mut inputs: Vec<Input> = Vec::new();
        // for l in 0..num_neurons {
        //     for _ in 0..num_inputs {
        //         inputs.push(Input::new(uniform.sample(&mut rng), l, 1.0, 1.0, 1, 1.0));
        //     }
        // }

        let mut neurons: Vec<Neuron> = Vec::new();
        for l in 0..num_neurons {
            let mut inputs: Vec<Input> = Vec::new();
            for _ in 0..num_inputs {
                inputs.push(Input::new(uniform.sample(&mut rng), 1.0, 0.0, 1, 1.0));
            }
            neurons.push(Neuron::new(l, 1.0, inputs));
        }

        Network { neurons }
    }

    // pub fn load_from() -> Network {
    //     todo!()
    // }

    // pub fn save_to() -> Network {
    //     todo!()
    // }
}

// write tests for the snn module...
