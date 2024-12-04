//! Module implementing the spiking neural networks.
use itertools::Itertools;

use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::connection::{self, Connection, ConnectionError};
use super::neuron::Neuron;
use super::spike_train::SpikeTrainError;
use crate::simulator::simulator::SimulationProgram;

/// Represents a spiking neural network.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Network {
    neurons: HashMap<usize, Neuron>,
    connections: Vec<Connection>,
}

impl Network {
    /// Create a new network from a list of connections.
    ///
    /// # Example
    /// ```rust
    /// use rusty_snn::core::network::Network;
    /// use rusty_snn::core::connection::Connection;
    ///
    /// let connections = vec![Connection::build(0, 1, 1.0, 1.0).unwrap(), Connection::build(1, 2, -1.0, 2.0).unwrap()];
    /// let network = Network::new(connections);
    ///
    /// assert_eq!(network.num_neurons(), 3);
    /// assert_eq!(network.num_connections(), 2);
    /// ```
    pub fn new(connections: Vec<Connection>) -> Self {
        let unique_ids = connections
            .iter()
            .flat_map(|c| vec![c.source_id(), c.target_id()])
            .unique();
        let neurons = unique_ids
            .map(|id| (id, Neuron::new()))
            .collect::<HashMap<usize, Neuron>>();

        Network {
            neurons,
            connections,
        }
    }

    /// Save the network to a file.
    ///
    /// # Example
    /// ```rust
    /// use rusty_snn::core::network::Network;
    /// use rusty_snn::core::connection::Connection;
    /// use std::path::Path;
    ///
    /// let connections = vec![Connection::build(0, 1, 1.0, 1.0).unwrap(), Connection::build(1, 2, -1.0, 2.0).unwrap()];
    /// let network = Network::new(connections);
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
    /// use rusty_snn::core::network::Network;
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

    /// Add a neuron to the network.
    /// If such a neuron already exists, do nothing.
    pub fn add_neuron(&mut self, id: usize) {
        self.neurons.entry(id).or_insert(Neuron::new());
    }

    /// Get the neuron with the specified id.
    pub fn neuron(&self, id: usize) -> Option<&Neuron> {
        self.neurons.get(&id)
    }

    /// Add a connection to the network, creating source and/or target neurons if necessary.
    /// The function returns an error for invalid connection.
    pub fn add_connection(
        &mut self,
        source_id: usize,
        target_id: usize,
        weight: f64,
        delay: f64,
    ) -> Result<(), ConnectionError> {
        self.add_neuron(source_id);
        self.add_neuron(target_id);
        self.connections.push(connection::Connection::build(
            source_id, target_id, weight, delay,
        )?);

        Ok(())
    }

    /// Add a firing time to each inputs whose source id is equal to the given id.
    pub fn add_input_firing_time(&mut self, id: usize, t: f64) {
        for connection in self.connections.iter().filter(|c| c.source_id() == id) {
            if let Some(neuron) = self.neurons.get_mut(&connection.target_id()) {
                neuron.add_input(connection.weight(), t + connection.delay());
            }
        }
    }

    /// Add a firing time to the neuron with the given id.
    pub fn add_firing_time(&mut self, id: usize, t: f64) -> Result<(), SpikeTrainError> {
        if let Some(neuron) = self.neurons.get_mut(&id) {
            neuron.add_firing_time(t)?;
        }
        Ok(())
    }

    pub fn fires(&mut self, id: usize, t: f64, noise: f64) -> Result<(), SpikeTrainError> {
        if let Some(neuron) = self.neurons.get_mut(&id) {
            neuron.fires(t, noise)?;
        }
        Ok(())
    }

    /// Read-only access to the neurons in the network.
    pub fn neurons(&self) -> &HashMap<usize, Neuron> {
        &self.neurons
    }

    /// Read-only access to the connections in the network.
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Returns the number of neurons in the network.
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Returns the number of connections in the network.
    pub fn num_connections(&self) -> usize {
        self.connections.len()
    }

    /// Returns the number of inputs to the neuron with the specified id.
    pub fn num_inputs(&self, id: usize) -> usize {
        self.connections
            .iter()
            .filter(move |c| c.target_id() == id)
            .count()
    }

    /// Returns the number of outputs to the neuron with the specified id.
    pub fn num_outputs(&self, id: usize) -> usize {
        self.connections
            .iter()
            .filter(|c| c.source_id() == id)
            .count()
    }

    /// Event-based simulation of the network.
    pub fn run<R: Rng>(
        &mut self,
        program: &SimulationProgram,
        rng: &mut R,
    ) -> Result<(), SpikeTrainError> {
        let normal = Normal::new(0.0, program.threshold_noise()).unwrap();

        // Set up neuron control
        for spike_train in program.spike_trains() {
            for t in spike_train.firing_times() {
                self.add_firing_time(spike_train.id(), *t)?;
                self.add_input_firing_time(spike_train.id(), *t);
            }
        }

        loop {
            let next_spike = self
                .neurons()
                .iter()
                .filter_map(|(id, neuron)| {
                    neuron
                        .next_spike(program.start(), program.end())
                        .map(|time| (*id, time))
                })
                .min_by(|(_, t1), (_, t2)| t1.partial_cmp(t2).unwrap());

            match next_spike {
                Some((id, t)) => {
                    self.fires(id, t, normal.sample(rng))?;
                    self.add_input_firing_time(id, t);
                }
                None => return Ok(()),
            };
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum NetworkError {
    /// Error for existing neurons.
    NeuronAlreadExists,
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NetworkError::NeuronAlreadExists => {
                write!(f, "The neuron with the specified ID already exists")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::core::spike_train::SpikeTrain;

    const SEED: u64 = 42;

    #[test]
    fn test_num_neurons() {
        let connections = vec![];
        let network = Network::new(connections);
        assert_eq!(network.num_neurons(), 0);

        let connections = vec![
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(1, 2, -1.0, 2.0).unwrap(),
            Connection::build(0, 0, 0.25, 0.5).unwrap(),
        ];
        let network = Network::new(connections);
        assert_eq!(network.num_neurons(), 3);
    }

    #[test]
    fn test_num_connections() {
        let connections = vec![];
        let network = Network::new(connections);
        assert_eq!(network.num_connections(), 0);

        let connections = vec![
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(1, 3, -1.0, 2.0).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
        ];
        let network = Network::new(connections);
        assert_eq!(network.num_connections(), 7);
    }

    #[test]
    fn test_num_inputs() {
        let connections = vec![
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(1, 3, -1.0, 2.0).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
        ];
        let network = Network::new(connections);

        assert_eq!(network.num_inputs(0), 0);
        assert_eq!(network.num_inputs(1), 2);
        assert_eq!(network.num_inputs(2), 4);
        assert_eq!(network.num_inputs(3), 1);
    }

    #[test]
    fn test_num_outputs() {
        let connections = vec![
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(1, 3, -1.0, 2.0).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
        ];
        let network = Network::new(connections);

        // assert_eq!(network.num_outputs(0), 6);
        // assert_eq!(network.num_outputs(1), 1);
        // assert_eq!(network.num_outputs(2), 0);
        // assert_eq!(network.num_outputs(3), 0);
    }

    #[test]
    fn test_add_neuron() {
        let mut network = Network::new(vec![Connection::build(0, 1, 1.0, 1.0).unwrap()]);
        assert_eq!(network.num_neurons(), 2);

        network.add_neuron(0);
        assert_eq!(network.num_neurons(), 2);

        network.add_neuron(42);
        assert_eq!(network.num_neurons(), 3);

        network.add_neuron(7);
        assert_eq!(network.num_neurons(), 4);
    }

    #[test]
    fn test_add_connection() {
        let mut network = Network::new(vec![]);
        assert_eq!(network.num_neurons(), 0);
        assert_eq!(network.num_connections(), 0);

        network.add_connection(0, 1, 1.0, 1.0).unwrap();
        assert_eq!(network.num_neurons(), 2);
        assert_eq!(network.num_connections(), 1);

        network.add_connection(0, 2, 1.0, 1.0).unwrap();
        assert_eq!(network.num_neurons(), 3);
        assert_eq!(network.num_connections(), 2);

        network.add_connection(1, 0, 1.0, 1.0).unwrap();
        assert_eq!(network.num_neurons(), 3);
        assert_eq!(network.num_connections(), 3);
    }

    #[test]
    fn test_save_load() {
        // Create a network
        let connections = vec![
            Connection::build(0, 1, 1.0, 1.0).unwrap(),
            Connection::build(1, 2, -1.0, 2.0).unwrap(),
            Connection::build(0, 2, 0.25, 0.5).unwrap(),
        ];
        let network = Network::new(connections);

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();

        // Save network to the temporary file
        network.save_to(temp_file.path()).unwrap();

        // Load the network from the temporary file
        let loaded_network = Network::load_from(temp_file.path()).unwrap();

        assert_eq!(network, loaded_network);
    }

    #[test]
    fn test_run_with_empty_network() {
        let mut network = Network::new(vec![]);

        let spike_trains = vec![];
        let program = SimulationProgram::build(0.0, 1.0, 0.0, &spike_trains).unwrap();
        let mut rng = StdRng::seed_from_u64(SEED);

        assert_eq!(network.run(&program, &mut rng), Ok(()));
    }

    #[test]
    fn test_run_with_disconnected_network() {
        let mut network = Network::new(vec![]);
        for i in 0..3 {
            network.add_neuron(i);
        }

        let spike_trains = vec![
            SpikeTrain::build(0, &[0.0, 2.0, 5.0]).unwrap(),
            SpikeTrain::build(1, &[1.0, 7.0]).unwrap(),
        ];
        let program = SimulationProgram::build(0.0, 10.0, 0.0, &spike_trains).unwrap();

        let mut rng = StdRng::seed_from_u64(SEED);
        assert_eq!(network.run(&program, &mut rng), Ok(()));
        assert_eq!(network.neuron(0).unwrap().firing_times(), &[0.0, 2.0, 5.0]);
        assert_eq!(network.neuron(1).unwrap().firing_times(), &[1.0, 7.0]);
        assert!(network.neuron(2).unwrap().firing_times().is_empty());
    }
}
