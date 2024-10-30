//! Network module with utilities for instantiating and managing networks of neurons.

use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use super::neuron::Neuron;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Network {
    neurons: Vec<Neuron>, // The network owns the neurons, use slices for Connections to other neurons.
}

/// The Network struct represents a spiking neural network.
impl Network {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Network { neurons }
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

    pub fn add_connection(&mut self, source_id: usize, target_id: usize, weight: f64, delay: f64) {
        self.neurons[target_id].add_input(source_id, weight, delay);
    }

    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }

    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    pub fn num_connections(&self) -> usize {
        self.neurons.iter().flat_map(|n| n.inputs()).count()
    }
}

#[cfg(test)]
mod tests {}