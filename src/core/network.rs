//! Module implementing the spiking neural networks.

use serde::{Deserialize, Serialize};
use serde::ser::{SerializeStruct, Serializer};
use serde::de::Deserializer;
use serde_json;
use std::collections::HashMap;
use std::fs::File;

use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;
use rand::seq::SliceRandom;

use rand::Rng;

use rayon::prelude::*;

use super::connection::Connection;
use super::error::CoreError;
use super::neuron::Neuron;
use crate::simulator::simulator::SimulationProgram;


use crate::core::MIN_PARALLEL_NEURONS;

#[derive(Debug, PartialEq)]
pub enum Topology {
    /// Random topology
    Random,
    /// Fixed in-degree topology, i.e., each neuron has the same number of inputs.
    /// Requires `num_connections & num_neurons == 0`.
    Fin,
    /// Fixed out-degree topology, i.e., each neuron has the same number of outputs.
    /// Requires `num_connections & num_neurons == 0`.
    Fout,
    /// Fixed in-degree and out-degree topology, i.e., each neuron has the same number of inputs and outputs.
    /// Requires `num_connections & num_neurons == 0`.
    FinFout,
}

/// Represents a spiking neural network.
// #[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[derive(Debug, PartialEq, Clone)]
pub struct Network {
    neurons: HashMap<usize, Neuron>,
    connections: HashMap<(usize, usize), Vec<Connection>>,
}

impl Serialize for Network {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("Network", 2)?;
        s.serialize_field("neurons", &self.neurons)?;

        // Serialize the connections HashMap with tuple keys
        let connections: HashMap<String, &Vec<Connection>> = self
            .connections
            .iter()
            .map(|(&(id1, id2), connections)| {
                (format!("({}, {})", id1, id2), connections)
            })
            .collect();
        s.serialize_field("connections", &connections)?;

        s.end()
    }
}

impl<'de> Deserialize<'de> for Network {
    fn deserialize<D>(deserializer: D) -> Result<Network, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct NetworkDef {
            neurons: HashMap<usize, Neuron>,
            connections: HashMap<String, Vec<Connection>>,
        }

        let network_def = NetworkDef::deserialize(deserializer)?;

        // Convert the connections HashMap with string keys back to tuple keys
        let connections: HashMap<(usize, usize), Vec<Connection>> = network_def
            .connections
            .into_iter()
            .map(|(key, connections)| {
                let key = key.trim_matches(|c| c == '(' || c == ')' || c == ' ');
                let mut parts = key.split(',');
                let id1 = parts.next().unwrap().trim().parse().unwrap();
                let id2 = parts.next().unwrap().trim().parse().unwrap();
                ((id1, id2), connections)
            })
            .collect();

        Ok(Network {
            neurons: network_def.neurons,
            connections,
        })
    }
}

impl Network {
    /// Create a new empty network.
    pub fn new() -> Self {
        Network {
            neurons:HashMap::new(),
            connections:HashMap::new(),
        }
    }

    /// Sample a network from the distribution.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rusty_snn::core::network::{Network, Topology};
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// let num_neurons = 10;
    /// let num_connections = 100;
    /// let lim_weights = (-0.1, 0.1);
    /// let lim_delays = (0.1, 10.0);
    /// let topology = Topology::Random;
    /// let mut rng = StdRng::seed_from_u64(42);
    /// 
    /// let network = Network::rand(num_neurons, num_connections, lim_weights, lim_delays, topology, &mut rng).unwrap();
    /// 
    /// assert_eq!(network.num_neurons(), 10);
    /// assert_eq!(network.num_connections(), 100);
    /// ```
    pub fn rand<R: Rng>(num_neurons: usize, num_connections: usize, lim_weights: (f64, f64),lim_delays: (f64, f64), topology: Topology, rng: &mut R) -> Result<Network, CoreError> {
        let mut network = Network::new();

        let (min_weight, max_weight) = lim_weights;
        let weight_dist = Uniform::new_inclusive(min_weight, max_weight);
        let (min_delay, max_delay) = lim_delays;
        if min_delay < 0.0 {
            return Err(CoreError::InvalidDelay);
        }
        let delay_dist = Uniform::new_inclusive(min_delay, max_delay);

        if !matches!(topology, Topology::Random) && num_connections % num_neurons != 0 {
            return Err(CoreError::IncompatibleTopology);
        }

        let (source_ids, target_ids) = match topology {
            Topology::Random => {
                let dist = Uniform::new(0, num_neurons);
                let source_ids = (0..num_connections).map(|_| dist.sample(rng)).collect::<Vec<usize>>();
                let target_ids = (0..num_connections).map(|_| dist.sample(rng)).collect::<Vec<usize>>();
                (source_ids, target_ids)
            },
            Topology::Fin => {
                let dist = Uniform::new(0, num_neurons);
                let source_ids = (0..num_connections).map(|_| dist.sample(rng)).collect::<Vec<usize>>();
                let target_ids = (0..num_connections).map(|i| i % num_neurons).collect::<Vec<usize>>();
                (source_ids, target_ids)
            },
            Topology::Fout => {
                let dist = Uniform::new(0, num_neurons);
                let source_ids = (0..num_connections).map(|i| i % num_neurons).collect::<Vec<usize>>();
                let target_ids = (0..num_connections).map(|_| dist.sample(rng)).collect::<Vec<usize>>();
                (source_ids, target_ids)
            },
            Topology::FinFout => {
                let source_ids = (0..num_connections).map(|i| i % num_neurons).collect::<Vec<usize>>();
                let mut target_ids = (0..num_connections).map(|i| i % num_neurons).collect::<Vec<usize>>();
                target_ids.shuffle(rng);
                (source_ids, target_ids)
            },
        };
        
        for (source_id, target_id) in source_ids.into_iter().zip(target_ids.into_iter()) {
            let weight = weight_dist.sample(rng);
            let delay = delay_dist.sample(rng);
            network.add_connection(source_id, target_id, weight, delay)?;
        }

        Ok(network)
    }

    /// Save the network to a file.
    ///
    /// # Example
    /// ```rust
    /// use rusty_snn::core::network::Network;
    /// use std::path::Path;
    ///
    /// let mut network = Network::new();
    /// network.add_connection(0, 1, 1.0, 1.0).unwrap();
    /// network.add_connection(1, 2, -1.0, 2.0).unwrap();
    /// network.add_connection(0, 3, 1.0, 1.0).unwrap();
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

    /// Add a neuron with the given ID to the network.
    /// If already exists, the function does nothing.
    pub fn add_neuron(&mut self, id: usize) {
        self.neurons.entry(id).or_insert(Neuron::new());
    }

    /// Add a connection to the network, creating source and/or target neurons if necessary.
    /// The function returns an error for invalid connection.
    pub fn add_connection(
        &mut self,
        source_id: usize,
        target_id: usize,
        weight: f64,
        delay: f64,
    ) -> Result<(), CoreError> {
        self.add_neuron(source_id);
        self.add_neuron(target_id);

        match self
            .connections
            .entry((source_id, target_id))
            .or_insert(vec![])
            .binary_search_by(|connection| connection.delay().partial_cmp(&delay).unwrap())
        {
            Ok(pos) | Err(pos) => self
                .connections
                .get_mut(&(source_id, target_id))
                .unwrap()
                .insert(pos, Connection::build(weight, delay)?),
        };

        Ok(())
    }

    /// Add a firing time to the neuron with the given id.
    pub fn add_firing_time(&mut self, id: usize, t: f64) -> Result<(), CoreError> {
        if let Some(neuron) = self.neurons.get_mut(&id) {
            neuron.add_firing_time(t)?;
        }
        Ok(())
    }

    pub fn firing_times(&self, id: usize) -> Option<&[f64]> {
        self.neurons.get(&id).map(|neuron| neuron.firing_times())
    }

    /// Add a firing time to the neuron with the given id.
    fn fires(&mut self, id: usize, t: f64, noise: f64) -> Result<(), CoreError> {
        if let Some(neuron) = self.neurons.get_mut(&id) {
            neuron.fires(t, noise)?;
            Ok(())
        } else {
            return Err(CoreError::NeuronNotFound);
        }
    }

    /// Add inputs to all neurons that receive input from the neuron with the specified id.
    pub fn add_inputs(&mut self, source_id: usize, t: f64) {
        for (target_id, neuron) in self.neurons.iter_mut() {
            if let Some(connections) = self.connections.get(&(source_id, *target_id)) {
                for connection in connections {
                    neuron.add_input(connection.weight(), t + connection.delay());
                }
            }
        }
    }

    /// Read-only access to the neurons in the network.
    pub fn neurons(&self) -> &HashMap<usize, Neuron> {
        &self.neurons
    }

    /// Read-only access to the connections in the network.
    pub fn connections(&self) -> &HashMap<(usize, usize), Vec<Connection>> {
        &self.connections
    }

    /// Returns the number of neurons in the network.
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Returns the number of connections in the network.
    pub fn num_connections(&self) -> usize {
        self.connections.iter().map(|(_, v)| v.len()).sum()
    }

    /// Returns the number of connections in the network between neurons with the specified ids.
    pub fn num_connections_from_to(&self, source_id: usize, target_id: usize) -> usize {
        self.connections
            .get(&(source_id, target_id))
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Returns the number of inputs to the neuron with the specified id.
    pub fn num_connections_from(&self, id: usize) -> usize {
        self.connections
            .iter()
            .filter(|&(&(source_id, _), _)| source_id == id)
            .map(|(_, connections)| connections.len())
            .sum()
    }

    /// Returns the number of outputs to the neuron with the specified id.
    pub fn num_connections_to(&self, id: usize) -> usize {
        self.connections
            .iter()
            .filter(|&(&(_, target_id), _)| target_id == id)
            .map(|(_, connections)| connections.len())
            .sum()
    }

    /// Event-based simulation of the network.
    pub fn run<R: Rng>(
        &mut self,
        program: &SimulationProgram,
        rng: &mut R,
    ) -> Result<(), CoreError> {
        let normal = Normal::new(0.0, program.threshold_noise()).unwrap();

        let total_duration = program.end() - program.start();
        let mut last_log_time = program.start();
        let log_interval = total_duration / 100.0;

        // Set up neuron control
        for spike_train in program.spike_trains() {
            for t in spike_train.firing_times() {
                self.add_firing_time(spike_train.id(), *t)?;
                self.add_inputs(spike_train.id(), *t);
            }
        }

        // for (id, neuron) in self.neurons.iter() {
        //     println!{"id:{}, inputs:{:?}", id, neuron.inputs()};
        // }

        let mut time = program.start();

        while time < program.end() {
            // Collect the candidate next spikes from all neurons, using parallel processing if the number of neurons is large
            let candidate_next_spikes = match self.neurons.len() > MIN_PARALLEL_NEURONS {
                true => self
                    .neurons()
                    .par_iter()
                    .filter_map(|(id, neuron)| {
                        neuron.next_spike(time).map(|t| (*id, t))
                    })
                    .collect::<Vec<(usize, f64)>>(),
                false => self
                    .neurons()
                    .iter()
                    .filter_map(|(id, neuron)| {
                        neuron.next_spike(time).map(|t| (*id, t))
                    })
                    .collect::<Vec<(usize, f64)>>(),
            };

            // If no neuron can fire, we're done
            if candidate_next_spikes.is_empty() {
                println!("Network activity has ceased...");
                return Ok(());
            }

            // Accept as many spikes as possible at the current time
            let mut next_spikes = vec![];
            for (id_target, t_target) in candidate_next_spikes.iter() {
                if candidate_next_spikes.iter().all(|(id_source, t_source)| {
                    match self.connections.get(&(*id_source, *id_target)) {
                        Some(connections) => {
                            *t_target <= *t_source + connections.first().unwrap().delay()
                        }
                        None => true,
                    }
                }) {
                    next_spikes.push((*id_target, *t_target));
                }
            }

            // Get the greatest among all accepted spikes
            time = next_spikes
                .iter()
                .fold(-f64::INFINITY, |acc, (_, t)| acc.max(*t));

            for (id, t) in next_spikes.iter() {
                self.fires(*id, *t, normal.sample(rng))?;
                self.add_inputs(*id, *t);
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

            // if num_spikes % 10 == 0 {
            //     println!("num_spikes: {}", num_spikes);
            // }
        }

        println!("Simulation completed successfully!");
        Ok(())
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
    fn test_add_neuron_and_connection() {
        let mut network = Network::new();

        network.add_neuron(0);
        network.add_neuron(7);
        network.add_neuron(42);

        assert_eq!(network.num_neurons(), 3);
        assert_eq!(network.num_connections(), 0);
    }

    #[test]
    fn test_add_connection() {
        let mut network = Network::new();

        network.add_connection(0, 1, 1.0, 1.0).unwrap();
        network.add_connection(2, 3, -1.0, 1.0).unwrap();
        network.add_connection(0, 3, 1.0, 1.0).unwrap();
        network.add_connection(2, 3, 1.0, 0.25).unwrap();
        network.add_connection(2, 3, 1.0, 5.0).unwrap();

        assert_eq!(network.num_neurons(), 4);
        assert_eq!(network.num_connections(), 5);

        assert_eq!(network.num_connections_from(0), 2);
        assert_eq!(network.num_connections_from(1), 0);
        assert_eq!(network.num_connections_to(0), 0);
        assert_eq!(network.num_connections_to(3), 4);
        assert_eq!(network.num_connections_from_to(0, 2), 0);
        assert_eq!(network.num_connections_from_to(0, 3), 1);
        assert_eq!(network.num_connections_from_to(2, 3), 3);

        assert_eq!(network.connections.get(&(2, 3)).unwrap().first().unwrap().delay(), 0.25);
        assert_eq!(network.connections.get(&(2, 3)).unwrap().last().unwrap().delay(), 5.0);
    }

    #[test]
    fn test_save_load() {
        // Create a network
        let mut network = Network::new();
        network.add_connection(0, 1, 1.0, 1.0).unwrap();
        network.add_connection(1, 2, -1.0, 2.0).unwrap();
        network.add_connection(0, 3, 1.0, 1.0).unwrap();

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();

        // Save network to the temporary file
        network.save_to(temp_file.path()).unwrap();

        // Load the network from the temporary file
        let loaded_network = Network::load_from(temp_file.path()).unwrap();

        assert_eq!(network, loaded_network);
    }

    #[test]
    fn test_rand_network() {
        let mut rng = StdRng::seed_from_u64(SEED);

        let network = Network::rand(277, 769, (-0.1, 0.1), (0.1, 10.0), Topology::Random, &mut rng).unwrap();
        assert_eq!(network.num_neurons(), 277);
        assert_eq!(network.num_connections(), 769);
        assert!(network.connections.iter().all(|((source_id, target_id), connections)| {
            *source_id < 277 && *target_id < 277 &&
                connections.iter().all(|connection| {
                    connection.weight() >= -0.1 && connection.weight() <= 0.1
                        && connection.delay() >= 0.1 && connection.delay() <= 10.0
                })
            }));

        let network = Network::rand(20, 400, (-0.1, 0.1), (0.1, 10.0), Topology::FinFout, &mut rng).unwrap();
        assert_eq!(network.num_neurons(), 20);
        assert_eq!(network.num_connections(), 400);
        assert!((0..20).all(|id| {
            network.num_connections_from(id) == 20 && network.num_connections_to(id) == 20
        }));

        let network = Network::rand(20, 400, (-0.1, 0.1), (0.1, 10.0), Topology::Fin, &mut rng).unwrap();
        assert_eq!(network.num_neurons(), 20);
        assert_eq!(network.num_connections(), 400);
        assert!((0..20).all(|id| {
            network.num_connections_to(id) == 20
            }));

        let network = Network::rand(20, 400, (-0.1, 0.1), (0.1, 10.0), Topology::Fout, &mut rng).unwrap();
        assert_eq!(network.num_neurons(), 20);
        assert_eq!(network.num_connections(), 400);
        assert!((0..20).all(|id| {
            network.num_connections_from(id) == 20
            }));
    }

    #[test]
    fn test_run_with_tiny_network() {
        let mut network = Network::new();

        network.add_connection(0,2,0.5,0.5).unwrap();
        network.add_connection(1,2,0.5,0.25).unwrap();
        network.add_connection(1,2,-0.75,3.5).unwrap();
        network.add_connection(2,3,2.0,1.0).unwrap();
        network.add_connection(0,3,-1.0,2.5).unwrap();

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
    }

    #[test]
    fn test_run_with_empty_network() {
        let mut network = Network::new();

        let spike_trains = vec![];
        let program = SimulationProgram::build(0.0, 1.0, 0.0, &spike_trains).unwrap();
        let mut rng = StdRng::seed_from_u64(SEED);

        assert_eq!(network.run(&program, &mut rng), Ok(()));
    }

    #[test]
    fn test_run_with_disconnected_network() {
        let mut network = Network::new();
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
