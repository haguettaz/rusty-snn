//! Alpha network related implementations.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::alpha::neuron::{AlphaInputSpikeTrain, AlphaNeuron};
use crate::core::network::{Connection, Network};
use crate::error::SNNError;

/// An alpha network composed of alpha neurons.
#[derive(Serialize, Deserialize)]
pub struct AlphaNetwork {
    pub neurons: Vec<AlphaNeuron>,
}

impl AlphaNetwork {
    /// Create a new empty network of alpha neurons
    pub fn new_empty(num_neurons: usize, seed: u64) -> Self {
        let neurons = (0..num_neurons)
            .map(|neuron_id| AlphaNeuron::new_empty(neuron_id, seed + neuron_id as u64))
            .collect();
        AlphaNetwork { neurons }
    }

    /// Creates a new network of alpha neurons with the provided neurons and connections.
    pub fn new_from(neurons: Vec<AlphaNeuron>, connections: Vec<Connection>) -> Self {
        let mut network = AlphaNetwork { neurons };
        for connection in connections {
            network.add_connection(&connection);
        }
        network
    }

    /// Returns a random network of alpha neurons, where each neuron has the same number of inputs.
    /// The delays are randomly generated between the specified limits.
    /// The weights are initialized to NaN.
    /// TODO: Example
    pub fn rand_fin(
        num_neurons: usize,
        num_inputs: usize,
        lim_delays: (f64, f64),
        seed: u64,
    ) -> Result<Self, SNNError> {
        let neurons = (0..num_neurons)
            .map(|neuron_id| AlphaNeuron::new_empty(neuron_id, seed + neuron_id as u64))
            .collect();
        let connections = Connection::rand_fin(
            num_inputs,
            (0, num_neurons),
            (0, num_neurons),
            lim_delays,
            seed,
        )?;

        Ok(Self::new_from(neurons, connections))
    }

    /// Returns a random network of alpha neurons, where each neuron has the same number of outputs.
    /// The delays are randomly generated between the specified limits.
    /// The weights are initialized to NaN.
    /// TODO: Example
    pub fn rand_fout(
        num_neurons: usize,
        num_outputs: usize,
        lim_delays: (f64, f64),
        seed: u64,
    ) -> Result<Self, SNNError> {
        let neurons = (0..num_neurons)
            .map(|neuron_id| AlphaNeuron::new_empty(neuron_id, seed + neuron_id as u64))
            .collect();
        let connections = Connection::rand_fout(
            num_outputs,
            (0, num_neurons),
            (0, num_neurons),
            lim_delays,
            seed,
        )?;
        Ok(Self::new_from(neurons, connections))
    }

    /// Returns a random network of alpha neurons, where each neuron has the same number of inputs and outputs.
    /// The delays are randomly generated between the specified limits.
    /// The weights are initialized to NaN.
    pub fn rand_fin_fout(
        num_neurons: usize,
        num_inputs_outputs: usize,
        lim_delays: (f64, f64),
        seed: u64,
    ) -> Result<Self, SNNError> {
        let neurons = (0..num_neurons)
            .map(|neuron_id| AlphaNeuron::new_empty(neuron_id, seed + neuron_id as u64))
            .collect();
        let connections =
            Connection::rand_fin_fout(num_neurons, num_inputs_outputs, lim_delays, seed)?;

        Ok(Self::new_from(neurons, connections))
    }

    /// Save the network to a file.
    pub fn save_to<P: AsRef<Path>>(&self, path: P) -> Result<(), SNNError> {
        let file = File::create(path).map_err(|e| SNNError::IOError(e.to_string()))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, self)
            .map_err(|e| SNNError::IOError(e.to_string()))?;
        writer.flush().map_err(|e| SNNError::IOError(e.to_string()))
    }

    /// Load a network from a file.
    pub fn load_from<P: AsRef<Path>>(path: P) -> Result<Self, SNNError> {
        let file = File::open(path).map_err(|e| SNNError::IOError(e.to_string()))?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| SNNError::IOError(e.to_string()))
    }
}

impl Network for AlphaNetwork {
    type InputSpikeTrain = AlphaInputSpikeTrain;
    type Neuron = AlphaNeuron;

    fn neurons_iter(&self) -> impl Iterator<Item = &Self::Neuron> + '_ {
        self.neurons.iter().map(|neuron| neuron as &Self::Neuron)
    }

    fn neurons_par_iter(&self) -> impl ParallelIterator<Item = &Self::Neuron> + '_ {
        self.neurons
            .par_iter()
            .map(|neuron| neuron as &Self::Neuron)
    }

    fn neurons_iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Neuron> + '_ {
        self.neurons
            .iter_mut()
            .map(|neuron| neuron as &mut Self::Neuron)
    }

    fn neurons_par_iter_mut(&mut self) -> impl ParallelIterator<Item = &mut Self::Neuron> + '_ {
        self.neurons
            .par_iter_mut()
            .map(|neuron| neuron as &mut Self::Neuron)
    }

    fn neuron_ref(&self, neuron_id: usize) -> Option<&Self::Neuron> {
        self.neurons
            .get(neuron_id)
            .map(|neuron| neuron as &Self::Neuron)
    }

    fn neuron_mut(&mut self, neuron_id: usize) -> Option<&mut Self::Neuron> {
        self.neurons
            .get_mut(neuron_id)
            .map(|neuron| neuron as &mut Self::Neuron)
    }
}
