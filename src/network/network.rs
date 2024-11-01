//! Network module with utilities for instantiating and managing networks of neurons.

use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use super::neuron::Neuron;

#[derive(Debug, PartialEq)]
pub enum NetworkError {
    /// Error for invalid neuron id.
    InvalidNeuronId,
    /// Error for invalid delay value.
    InvalidDelay,
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NetworkError::InvalidNeuronId => write!(f, "Invalid neuron id: out of bounds"),
            NetworkError::InvalidDelay => write!(f, "Invalid delay value: must be non-negative"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Network {
    neurons: Vec<Neuron>, // The network owns the neurons, use slices for Connections to other neurons.
}

/// The Network struct represents a spiking neural network.
impl Network {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Network { neurons }
    }

    /// Save the network to a file.
    ///
    /// # Example
    /// ```rust
    /// use std::path::Path;
    /// use rusty_snn::network::network::Network;
    /// use rusty_snn::network::neuron::Neuron;
    ///
    /// // Create network from a vector of neurons and add connections
    /// let mut network = Network::new((0..3).map(|id| Neuron::new(id, 1.0)).collect());
    /// network.add_connection(0, 1, 0.0, 1.0);
    /// network.add_connection(1, 2, 0.0, 1.0);
    ///
    /// // Save the network to a file
    /// network.save_to(Path::new("network.json")).unwrap();
    /// ```
    pub fn save_to<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, self)?;
        writer.flush()?;
        Ok(())
    }

    /// Load a network from a file.
    ///
    /// # Example
    /// ```rust
    /// use rusty_snn::network::network::Network;
    ///
    /// // Load the network from a file
    /// let network = Network::load_from("network.json").unwrap();
    /// ```
    pub fn load_from<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    pub fn add_connection(
        &mut self,
        source_id: usize,
        target_id: usize,
        weight: f64,
        delay: f64,
    ) -> Result<(), NetworkError> {
        if source_id >= self.num_neurons() {
            return Err(NetworkError::InvalidNeuronId);
        }
        if target_id >= self.num_neurons() {
            return Err(NetworkError::InvalidNeuronId);
        }
        if delay < 0.0 {
            return Err(NetworkError::InvalidDelay);
        }
        self.neurons[target_id].add_input(source_id, weight, delay);
        Ok(())
    }

    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }

    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    pub fn num_connections(&self) -> usize {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.inputs())
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_network() {
        let mut network = Network::new((0..3).map(|id| Neuron::new(id, 1.0)).collect());
        assert_eq!(
            network.add_connection(999, 0, 1.0, 1.0),
            Err(NetworkError::InvalidNeuronId)
        );
        assert_eq!(
            network.add_connection(0, 999, 1.0, 1.0),
            Err(NetworkError::InvalidNeuronId)
        );
        assert_eq!(
            network.add_connection(0, 1, 1.0, -1.0),
            Err(NetworkError::InvalidDelay)
        );

        assert_eq!(network.add_connection(1, 2, 1.0, 1.0), Ok(()));
        assert_eq!(network.num_neurons(), 3);
        assert_eq!(network.num_connections(), 1);
    }

    #[test]
    fn test_save_load() {
        let mut network = Network::new((0..3).map(|id| Neuron::new(id, 1.0)).collect());
        assert_eq!(network.add_connection(0, 1, 0.0, 1.0), Ok(()));
        assert_eq!(network.add_connection(1, 2, 0.0, 1.0), Ok(()));

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        // Save network to the temporary file
        network.save_to(temp_file.path()).unwrap();

        // Load the network from the temporary file
        let loaded_network = Network::load_from(temp_file.path()).unwrap();

        assert_eq!(network, loaded_network);
    }
}
