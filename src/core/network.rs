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

    /// Get an immutable reference to the neuron with the specified id.
    pub fn neuron(&self, id: usize) -> Option<&Neuron> {
        self.neurons.get(&id)
    }

    /// Get a mutable reference to the neuron with the specified id.
    pub fn mut_neuron(&mut self, id: usize) -> Option<&mut Neuron> {
        self.neurons.get_mut(&id)
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

    /// Add a firing time to the neuron with the given id.
    pub fn add_firing_time(&mut self, id: usize, t: f64) -> Result<(), SpikeTrainError> {
        if let Some(neuron) = self.neurons.get_mut(&id) {
            neuron.add_firing_time(t)?;
        }
        Ok(())
    }

    pub fn firing_times(&self, id: usize) -> Option<&[f64]> {
        self.neurons.get(&id).map(|neuron| neuron.firing_times())
    }

    /// Add a firing time to the neuron with the given id.
    fn fires(&mut self, id: usize, t: f64, noise: f64) -> Result<(), SpikeTrainError> {
        if let Some(neuron) = self.neurons.get_mut(&id) {
            neuron.fires(t, noise)?;
        }
        Ok(())
    }

    /// Add inputs to all neurons that receive input from the neuron with the specified id.
    pub fn add_inputs(&mut self, id: usize, t: f64) {
        for connection in self.connections.iter().filter(|c| c.source_id() == id) {
            if let Some(neuron) = self.neurons.get_mut(&connection.target_id()) {
                neuron.add_input(connection.weight(), t + connection.delay());
            }
        }
    }

    pub fn update_frozen_inputs(&mut self, time: f64) {
        for neuron in self.neurons.values_mut() {
            neuron.update_frozen_inputs(time);
        }
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

        let total_duration = program.end() - program.start();
        let mut last_log_time = program.start();
        let log_interval = 1.0;

        // Set up neuron control
        for spike_train in program.spike_trains() {
            for t in spike_train.firing_times() {
                self.add_firing_time(spike_train.id(), *t)?;
                self.add_inputs(spike_train.id(), *t);
            }
        }

        for (id, neuron) in self.neurons.iter() {
            println!{"id:{}, inputs:{:?}", id, neuron.inputs()};
        }

        let mut time = program.start();

        while time < program.end() {
            // Since a new firing time can only be added as an input in the future, all inputs with firing times < time are frozen.
            self.update_frozen_inputs(time);

            // Collect the candidate next spikes from all neurons
            let next_spikes = self
                .neurons()
                .iter()
                .filter_map(|(id, neuron)| {
                    println!("{:?}", id);
                    neuron
                        .next_spike(program.end())
                        .map(|t| (*id, t))
                }).collect::<Vec<(usize, f64)>>();
                
            println!("{:?}", next_spikes);

            // If no neuron can fire, we're done
            if next_spikes.is_empty() {
                return Ok(());
            }

            // Otherwise, we find the next spike time, and handle the really unlikely case of multiple neurons firing at the same time
            time = next_spikes.iter().fold(f64::INFINITY, |acc, (_, t)| acc.min(*t));

            for (id, _) in next_spikes.iter().filter(|(_, t)| *t == time) {
                self.fires(*id, time, normal.sample(rng))?;
                self.add_inputs(*id, time);
            }

            // Check if it's time to log progress
            if time - last_log_time >= log_interval {
                let progress = ((time - program.start()) / total_duration) * 100.0;
                println!(
                    "Simulation progress: {:.2}% (Time: {:.2}/{:.2})",
                    progress,
                    time,
                    program.end()
                );
                last_log_time = time;
            }
            
        }
        Ok(())
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
    use core::panic;

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

        assert_eq!(network.num_outputs(0), 6);
        assert_eq!(network.num_outputs(1), 1);
        assert_eq!(network.num_outputs(2), 0);
        assert_eq!(network.num_outputs(3), 0);
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
    fn test_run_with_tiny_network() {
        let connections = vec![
            Connection::build(0, 2, 0.5, 0.5).unwrap(),
            Connection::build(1, 2, 0.5, 0.25).unwrap(),
            Connection::build(1, 2, -0.75, 3.5).unwrap(),
            Connection::build(2, 3, 2.0, 1.0).unwrap(),
            Connection::build(0, 3, -1.0, 2.5).unwrap(),
        ];
        let mut network = Network::new(connections);

        let spike_trains = vec![
            SpikeTrain::build(0, &[0.5]).unwrap(),
            SpikeTrain::build(1, &[0.75]).unwrap(),
        ];
        let program = SimulationProgram::build(0.0, 10.0, 0.0, &spike_trains).unwrap();
        let mut rng = StdRng::seed_from_u64(SEED);

        assert_eq!(network.run(&program, &mut rng), Ok(()));
        assert_eq!(network.firing_times(0).unwrap(), &[0.5]);
        assert_eq!(network.firing_times(1).unwrap(), &[0.75]);
        assert_eq!(network.firing_times(2).unwrap(), &[2.0]);
        assert_eq!(network.firing_times(3).unwrap(), &[4.0]);
        panic!();
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
        assert_eq!(network.firing_times(0).unwrap(), &[0.0, 2.0, 5.0]);
        assert_eq!(network.firing_times(1).unwrap(), &[1.0, 7.0]);
        assert!(network.firing_times(2).unwrap().is_empty());
    }
}
