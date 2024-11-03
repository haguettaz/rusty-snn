//! Network module with utilities for instantiating and managing networks of neurons.

use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use super::connection::Connection;
use super::neuron::Neuron;
use super::error::NetworkError;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Network {
    neurons: Vec<Neuron>,
    connections: Vec<Connection>,
}

/// The Network struct represents a spiking neural network.
impl Network {
    /// Create a new network from neurons and connections between them.
    /// The function returns an error if the connections refer to non-existent neurons.
    ///
    /// # Example
    /// ```rust
    /// use rusty_snn::network::network::Network;
    /// use rusty_snn::network::neuron::Neuron;
    /// use rusty_snn::network::connection::Connection;
    ///
    /// let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
    /// let connections = vec![
    ///     Connection::build(0, 1, 1.0, 1.0).unwrap(),
    ///     Connection::build(1, 2, -1.0, 2.0).unwrap(),
    ///     Connection::build(2, 0, -0.125, 0.5).unwrap(),
    ///     Connection::build(2, 2, 0.25, 0.5).unwrap()
    /// ];
    /// let network = Network::build(neurons, connections).unwrap();
    /// 
    /// assert_eq!(network.num_neurons(), 3);
    /// assert_eq!(network.num_connections(), 4);
    /// ```
    pub fn build(neurons: Vec<Neuron>, connections: Vec<Connection>) -> Result<Self, NetworkError> {
        for connection in connections.iter() {
            if None
                == neurons
                    .iter()
                    .find(|neuron| neuron.id() == connection.source_id())
            {
                return Err(NetworkError::InvalidNeuronId);
            }

            if None
                == neurons
                    .iter()
                    .find(|neuron| neuron.id() == connection.target_id())
            {
                return Err(NetworkError::InvalidNeuronId);
            }
        }
        Ok(Network {
            neurons,
            connections,
        })
    }

    /// Save the network to a file.
    ///
    /// # Example
    /// ```rust
    /// use std::path::Path;
    /// use rusty_snn::network::network::Network;
    /// use rusty_snn::network::neuron::Neuron;
    /// use rusty_snn::network::connection::Connection;
    ///
    /// let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
    /// let connections = vec![
    ///     Connection::build(0, 1, 1.0, 1.0).unwrap(),
    ///     Connection::build(1, 2, -1.0, 2.0).unwrap(),
    ///     Connection::build(2, 2, 0.25, 0.5).unwrap()
    /// ];
    /// let network = Network::build(neurons, connections).unwrap();
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

    /// Add a connection to the network.
    /// The function returns an error if the connection refers to non-existent neurons.
    pub fn add_connection(
        &mut self,
        connection: Connection,
    ) -> Result<(), NetworkError> {
        if None
            == self.neurons
                .iter()
                .find(|neuron| neuron.id() == connection.source_id())
        {
            return Err(NetworkError::InvalidNeuronId);
        }

        if None
            == self.neurons
                .iter()
                .find(|neuron| neuron.id() == connection.target_id())
        {
            return Err(NetworkError::InvalidNeuronId);
        }

        self.connections.push(connection);
        Ok(())
    }

    /// Add a neuron to the network.
    pub fn add_neuron(
        &mut self,
        neuron: Neuron,
    ) -> Result<(), NetworkError> {
        self.neurons.push(neuron);
        Ok(())
    }

    /// Returns a reference to the neurons in the network.
    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }

    /// Returns the number of neurons in the network.
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Returns the number of connections in the network.
    pub fn num_connections(&self) -> usize {
        self.connections.len()
        // self.neurons
        //     .iter()
        //     .flat_map(|neuron| neuron.inputs())
        //     .count()
    }

    /// Get a collection of connections transmitting to the specified neuron.
    pub fn transmit_to(&self, neuron_id: usize) -> impl Iterator<Item = &Connection> {
        self.connections
            .iter()
            .filter(move |c| c.target_id() == neuron_id)
    }

    /// Get a collection of connections transmitting from the specified neuron.
    pub fn transmit_from(&self, neuron_id: usize) -> impl Iterator<Item = &Connection> {
        self.connections
            .iter()
            .filter(move |c| c.source_id() == neuron_id)
    }

    pub fn simulate(&self) -> Result<(), NetworkError> {
        // alternate between:
        // 1. update neuron states based on connections (in-mode, neurons read connections)
        // 2. update connections based on neuron states (out-mode, neurons write connections)
        // both phases are parallelizable, as the connections are completely partioned between neurons
        // connections should be sorted by target neuron id
        todo!()
    }

    fn update_neurons(&self) {
        // each neuron has a read permission on its input connections
        todo!()
    }

    fn update_connections(&self) {
        // each neuron has a write permission on its output connections
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_build_network_invalid_id() {
        let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
        let connections = vec![Connection::build(0, 999, 1.0, 1.0).unwrap()];
        let result = Network::build(neurons, connections);
        assert_eq!(result, Err(NetworkError::InvalidNeuronId));

        let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
        let connections = vec![Connection::build(999, 0, 1.0, 1.0).unwrap()];
        let result = Network::build(neurons, connections);
        assert_eq!(result, Err(NetworkError::InvalidNeuronId));
    }

    #[test]
    fn test_add_connection_invalid_id() {
        let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
        let mut network = Network::build(neurons, vec![]).unwrap();
        let connection = Connection::build(0, 999, 1.0, 1.0).unwrap();
        assert_eq!(network.add_connection(connection), Err(NetworkError::InvalidNeuronId));
    }

    #[test]
    fn test_save_load() {
        let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
        let connections = vec![
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(1, 2, -1.0, 2.0).unwrap(),
            Connection::build(2, 2, 0.25, 0.5).unwrap(),
        ];
        let network = Network::build(neurons, connections).unwrap();

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        // Save network to the temporary file
        network.save_to(temp_file.path()).unwrap();

        // Load the network from the temporary file
        let loaded_network = Network::load_from(temp_file.path()).unwrap();

        assert_eq!(network, loaded_network);
    }

    #[test]
    fn test_transmit_to() {
        let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
        let connections = vec![
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(1, 2, -1.0, 2.0).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
        ];
        let network = Network::build(neurons, connections.clone()).unwrap();

        let mut connection_to_2 = network.transmit_to(2);
        assert_eq!(connection_to_2.next(), Some(&connections[1]));
        assert_eq!(connection_to_2.next(), Some(&connections[2]));
        assert_eq!(connection_to_2.next(), None);
    }

    #[test]
    fn test_transmit_from() {
        let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
        let connections = vec![
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(1, 2, -1.0, 2.0).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
        ];
        let network = Network::build(neurons, connections.clone()).unwrap();

        let mut connection_from_0 = network.transmit_from(0);
        assert_eq!(connection_from_0.next(), Some(&connections[0]));
        assert_eq!(connection_from_0.next(), Some(&connections[2]));
        assert_eq!(connection_from_0.next(), None);
    }
}
