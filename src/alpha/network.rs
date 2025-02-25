//! Alpha network related implementations.
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::alpha::neuron::{AlphaInputSpikeTrain, AlphaNeuron};
use crate::core::network::{Connection, Network};
use crate::core::neuron::Neuron;
use crate::error::SNNError;

/// An alpha network composed of alpha neurons.
#[derive(Serialize, Deserialize)]
pub struct AlphaNetwork {
    pub neurons: Vec<AlphaNeuron>,
    min_delays: Option<Vec<Vec<f64>>>, // Minimum delays between each pair of neurons,
}

impl AlphaNetwork {
    /// Create a new empty network of alpha neurons
    pub fn new_empty(num_neurons: usize, seed: u64) -> Self {
        let neurons = (0..num_neurons)
            .map(|neuron_id| AlphaNeuron::new_empty(neuron_id, seed + neuron_id as u64))
            .collect();
        AlphaNetwork {
            neurons,
            min_delays: None,
        }
    }

    /// Creates a new network of alpha neurons with the provided neurons and connections.
    pub fn new_from_iter(neurons: impl Iterator<Item = AlphaNeuron>, connections: impl Iterator<Item = Connection>) -> Self {
        let neurons= neurons.collect();
        let mut network = AlphaNetwork {
            neurons,
            min_delays: None,
        };
        for connection in connections {
            network.push_connection(&connection);
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
            .map(|neuron_id| AlphaNeuron::new_empty(neuron_id, seed + neuron_id as u64));        let connections = Connection::rand_fin(
            num_inputs,
            (0, num_neurons),
            (0, num_neurons),
            lim_delays,
            seed,
        )?;

        Ok(Self::new_from_iter(neurons, connections))
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
            .map(|neuron_id| AlphaNeuron::new_empty(neuron_id, seed + neuron_id as u64));
        let connections = Connection::rand_fout(
            num_outputs,
            (0, num_neurons),
            (0, num_neurons),
            lim_delays,
            seed,
        )?;
        Ok(Self::new_from_iter(neurons, connections))
    }

    /// Returns a random network of alpha neurons, where each neuron has the same number of inputs and outputs.
    /// The delays are randomly generated between the specified limits.
    /// The weights are initialized to NaN.
    pub fn rand_fc(
        num_neurons: usize,
        lim_delays: (f64, f64),
        seed: u64,
    ) -> Result<Self, SNNError> {
        let neurons = (0..num_neurons)
            .map(|neuron_id| AlphaNeuron::new_empty(neuron_id, seed + neuron_id as u64));
        let connections =
            Connection::rand_fc(num_neurons, lim_delays, seed)?;

        Ok(Self::new_from_iter(neurons, connections))
    }

    // /// Returns a random network of alpha neurons, where each neuron has the same number of inputs and outputs.
    // /// The delays are randomly generated between the specified limits.
    // /// The weights are initialized to NaN.
    // pub fn rand_fin_fout(
    //     num_neurons: usize,
    //     num_inputs_outputs: usize,
    //     lim_delays: (f64, f64),
    //     seed: u64,
    // ) -> Result<Self, SNNError> {
    //     let neurons = (0..num_neurons)
    //         .map(|neuron_id| AlphaNeuron::new_empty(neuron_id, seed + neuron_id as u64))
    //         .collect();
    //     let connections =
    //         Connection::rand_fin_fout(num_neurons, num_inputs_outputs, lim_delays, seed)?;

    //     Ok(Self::new_from(neurons, connections))
    // }

    /// Save the network to a gzip-compressed file.
    /// The network is serialized to JSON and compressed using gzip compression.
    pub fn save_to<P: AsRef<Path>>(&self, path: P) -> Result<(), SNNError> {
        let file = File::create(path).map_err(|e| SNNError::IOError(e.to_string()))?;
        // Use a higher compression level for better size reduction
        let encoder = GzEncoder::new(file, Compression::best());
        let writer = BufWriter::new(encoder);
        serde_json::to_writer(writer, self).map_err(|e| SNNError::IOError(e.to_string()))
    }

    /// Load the network from a gzip-compressed file.
    /// Reads and decompresses a network previously saved with `save_to`.
    pub fn load_from<P: AsRef<Path>>(path: P) -> Result<Self, SNNError> {
        let file = File::open(path).map_err(|e| SNNError::IOError(e.to_string()))?;
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);
        serde_json::from_reader(reader).map_err(|e| SNNError::IOError(e.to_string()))
    }

    /// Init the minimum delays between each pair of neurons in the network.
    /// The delay from a neuron to itself is zero (due to its refractory period).
    /// Otherwise, the delay is the minimum delay between all possible paths connecting the two neurons.
    /// They are computed using the [Floyd-Warshall algorithm](https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm).
    pub fn init_min_delays(&mut self) {
        let mut min_delays = self
            .neurons_iter()
            .map(|target_neuron| {
                self.neurons_iter()
                    .map(|source_neuron| {
                        if target_neuron.id() == source_neuron.id() {
                            0.0
                        } else {
                            target_neuron
                                .inputs_iter()
                                .filter(|input| input.source_id == source_neuron.id())
                                .map(|input| input.delay)
                                .min_by(|a, b| a.partial_cmp(b).expect("Invalid delay"))
                                .unwrap_or(f64::INFINITY)
                        }
                    })
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();

        for inter_neuron in self.neurons_iter() {
            for target_neuron in self.neurons_iter() {
                for source_neuron in self.neurons_iter() {
                    let source_target_delay = min_delays[target_neuron.id()][source_neuron.id()];
                    let source_inter_target_delay = min_delays[inter_neuron.id()]
                        [source_neuron.id()]
                        + min_delays[target_neuron.id()][inter_neuron.id()];
                    if source_target_delay > source_inter_target_delay {
                        min_delays[target_neuron.id()][source_neuron.id()] =
                            source_inter_target_delay;
                    }
                }
            }
        }
        self.min_delays = Some(min_delays);
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

    /// Returns the minimum delay in spike propagation between each pair of neurons in the network.
    fn min_delay_from_to(&mut self, source_id: usize, target_id: usize) -> f64 {
        if let None = self.min_delays {
            self.init_min_delays();
        }

        self.min_delays.as_ref().unwrap()[target_id][source_id]
    }
}
