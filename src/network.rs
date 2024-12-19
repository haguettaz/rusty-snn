//! This module implements the `Network` structure and some core functionalities.

use core::f64;
use log::{debug, error, info, trace, warn};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::Normal;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use super::error::SNNError;
use super::neuron::Neuron;
use super::simulator::SimulationProgram;
use super::spike_train::SpikeTrain;
use super::optim::Objective;

/// Minimum number of neurons to use parallel processing.
pub const MIN_PARALLEL_NEURONS: usize = 10000;

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
// #[derive(Debug, PartialEq, Clone)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Network {
    // A collection of neurons in the network, indexed by their IDs.
    neurons: Vec<Neuron>,
    // A collection of connections between neurons, indexed by target (outer map) and source (inner map) neuron IDs.
    // connections: HashMap<usize, HashMap<usize, Vec<Connection>>>,
}


impl Network {
    /// Create a new empty network.
    pub fn new() -> Self {
        Network {
            neurons: vec![],
            // connections: HashMap::new(),
        }
    }

    /// Create a random network with the specified parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rusty_snn::network::{Network, Topology};
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// let num_neurons = 10;
    /// let num_connections = 100;
    /// let lim_weights = (-0.1, 0.1);
    /// let lim_delays = (0.1, 10.0);
    /// let topology = Topology::Fin;
    /// let mut rng = StdRng::seed_from_u64(42);
    ///
    /// let network = Network::rand(num_neurons, num_connections, lim_weights, lim_delays, topology, &mut rng).unwrap();
    ///
    /// assert_eq!(network.num_neurons(), 10);
    /// assert_eq!(network.num_connections(), 100);
    /// ```
    pub fn rand<R: Rng>(
        num_neurons: usize,
        num_connections: usize,
        lim_weights: (f64, f64),
        lim_delays: (f64, f64),
        topology: Topology,
        rng: &mut R,
    ) -> Result<Network, SNNError> {
        if !matches!(topology, Topology::Random) && num_connections % num_neurons != 0 {
            return Err(SNNError::IncompatibleTopology {
                num_neurons,
                num_connections,
            });
        }

        let (min_weight, max_weight) = lim_weights;
        let weight_dist = Uniform::new_inclusive(min_weight, max_weight);

        let (min_delay, max_delay) = lim_delays;
        if min_delay < 0.0 {
            return Err(SNNError::InvalidDelay);
        }
        let delay_dist = Uniform::new_inclusive(min_delay, max_delay);

        let (source_ids, target_ids) = match topology {
            Topology::Random => {
                let dist = Uniform::new(0, num_neurons);
                let source_ids = (0..num_connections)
                    .map(|_| dist.sample(rng))
                    .collect::<Vec<usize>>();
                let target_ids = (0..num_connections)
                    .map(|_| dist.sample(rng))
                    .collect::<Vec<usize>>();
                (source_ids, target_ids)
            }
            Topology::Fin => {
                let dist = Uniform::new(0, num_neurons);
                let source_ids = (0..num_connections)
                    .map(|_| dist.sample(rng))
                    .collect::<Vec<usize>>();
                let target_ids = (0..num_connections)
                    .map(|i| i % num_neurons)
                    .collect::<Vec<usize>>();
                (source_ids, target_ids)
            }
            Topology::Fout => {
                let dist = Uniform::new(0, num_neurons);
                let source_ids = (0..num_connections)
                    .map(|i| i % num_neurons)
                    .collect::<Vec<usize>>();
                let target_ids = (0..num_connections)
                    .map(|_| dist.sample(rng))
                    .collect::<Vec<usize>>();
                (source_ids, target_ids)
            }
            Topology::FinFout => {
                let source_ids = (0..num_connections)
                    .map(|i| i % num_neurons)
                    .collect::<Vec<usize>>();
                let mut target_ids = (0..num_connections)
                    .map(|i| i % num_neurons)
                    .collect::<Vec<usize>>();
                target_ids.shuffle(rng);
                (source_ids, target_ids)
            }
        };

        let mut network = Network::new();

        for id in 0..num_neurons {
            network.add_neuron(id);
        }

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
    /// use rusty_snn::network::Network;
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
    /// use rusty_snn::network::Network;
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
        match self.neurons.binary_search_by_key(&id, |neuron| neuron.id()) {
            Ok(_) => (),
            Err(pos) => {
                self.neurons.insert(pos, Neuron::new(id));
            }
        }
    }

    /// Add a connection to the network, creating source and/or target neurons if necessary.
    /// The function returns an error for invalid connection.
    pub fn add_connection(
        &mut self,
        source_id: usize,
        target_id: usize,
        weight: f64,
        delay: f64,
    ) -> Result<(), SNNError> {
        if delay < 0.0 {
            return Err(SNNError::InvalidDelay);
        }

        // Add both source and target neurons if they don't exist
        self.add_neuron(source_id);
        self.add_neuron(target_id);

        let target_neuron = self.neuron_mut(target_id)?;
        target_neuron.add_input(source_id, weight, delay);

        Ok(())
    }

    // /// Set neurons' firing times from the given spike trains.
    // pub fn init_from_spike_trains(
    //     &mut self,
    //     spike_trains: Vec<SpikeTrain>,
    // ) -> Result<(), SNNError> {
    //     for spike_train in spike_trains.iter() {
    //         let neuron = self.neuron_mut(spike_train.id())?;
    //         neuron.set_firing_times(spike_train.firing_times())?;
    //     }
    //     Ok(())
    // }

    /// Returns a slice of firing times of the neuron with the specified id.
    /// If the neuron does not exist, the function returns an error.
    pub fn firing_times(&self, id: usize) -> Result<&[f64], SNNError> {
        let neuron = self.neuron(id)?;
        Ok(neuron.firing_times())
    }

    /// Add a firing time to the neuron with the given id.
    pub fn add_firing_time(&mut self, id: usize, t: f64) -> Result<(), SNNError> {
        let neuron = self.neuron_mut(id)?;
        neuron.add_firing_time(t)?;
        Ok(())
    }

    /// Add a firing time to the neuron with the given id.
    pub fn extend_firing_times(&mut self, id: usize, firing_times: &[f64]) -> Result<(), SNNError> {
        let neuron = self.neuron_mut(id)?;
        neuron.extend_firing_times(firing_times)?;
        Ok(())
    }

    /// Add a firing time to the neuron with the given id.
    fn fires(&mut self, id: usize, t: f64, noise: f64) -> Result<(), SNNError> {
        let neuron = self.neuron_mut(id)?;
        neuron.fires(t, noise)
    }

    /// Returns a slice of neurons in the network.
    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }

    /// Returns a reference to the neuron with the specified id.
    /// Reminder: the neurons are sorted by id.
    pub fn neuron(&self, id: usize) -> Result<&Neuron, SNNError> {
        let pos = match self.neurons.binary_search_by_key(&id, |neuron| neuron.id()) {
            Ok(pos) => pos,
            Err(_) => {
                return Err(SNNError::NeuronNotFound);
            }
        };
        Ok(&self.neurons[pos])
    }

    /// Returns a mutable reference to the neuron with the specified id.
    /// Reminder: the neurons are sorted by id.
    pub fn neuron_mut(&mut self, id: usize) -> Result<&mut Neuron, SNNError> {
        let pos = match self.neurons.binary_search_by_key(&id, |neuron| neuron.id()) {
            Ok(pos) => pos,
            Err(_) => {
                return Err(SNNError::NeuronNotFound);
            }
        };
        Ok(&mut self.neurons[pos])
    }

    /// Returns the number of neurons in the network.
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Returns the number of connections in the network.
    pub fn num_connections(&self) -> usize {
        self.neurons
            .iter()
            .fold(0, |acc, neuron| acc + neuron.num_inputs())
    }

    /// Returns the number of connections targeting the neuron with the specified id.
    pub fn num_connections_to(&self, target_id: usize) -> usize {
        self.neuron(target_id)
            .map_or(0, |neuron| neuron.num_inputs())
    }

    /// Returns the number of connections originating from the neuron with the specified id.
    pub fn num_connections_from(&self, source_id: usize) -> usize {
        self.neurons
            .iter()
            .fold(0, |acc, neuron| acc + neuron.num_inputs_from(source_id))
    }

    /// Returns the minimum delay in spike propagation between each pair of neurons in the network.
    /// The delay from a neuron to itself is zero (refractory period).
    /// Otherwise, the delay is the minimum delay between all possible paths connecting the two neurons.
    /// They are computed using the Floyd-Warshall algorithm.
    pub fn min_delays(&self) -> HashMap<(usize, usize), f64> {
        let mut min_delays: HashMap<(usize, usize), f64> = HashMap::new();

        for target_neuron in self.neurons.iter() {
            for source_neuron in self.neurons.iter() {
                if let Some(delay) = target_neuron.min_delay_path(source_neuron.id()) {
                    min_delays.insert((source_neuron.id(), target_neuron.id()), delay);
                }
            }
        }

        for inter_neuron in self.neurons.iter() {
            for target_neuron in self.neurons.iter() {
                for source_neuron in self.neurons.iter() {
                    let source_target_delay = *min_delays
                        .get(&(source_neuron.id(), target_neuron.id()))
                        .unwrap_or(&f64::INFINITY);
                    let source_inter_target_delay = *min_delays
                        .get(&(source_neuron.id(), inter_neuron.id()))
                        .unwrap_or(&f64::INFINITY)
                        + *min_delays
                            .get(&(inter_neuron.id(), target_neuron.id()))
                            .unwrap_or(&f64::INFINITY);
                    if source_target_delay > source_inter_target_delay {
                        min_delays.insert(
                            (source_neuron.id(), target_neuron.id()),
                            source_inter_target_delay,
                        );
                    }
                }
            }
        }

        min_delays
    }

    /// Event-based simulation of the network.
    pub fn run<R: Rng>(
        &mut self,
        program: &SimulationProgram,
        rng: &mut R,
    ) -> Result<(), SNNError> {
        let normal = Normal::new(0.0, program.threshold_noise()).unwrap();

        let total_duration = program.end() - program.start();
        let mut last_log_time = program.start();
        let log_interval = total_duration / 100.0;

        // Setup neuron control
        for spike_train in program.spike_trains() {
            self.extend_firing_times(spike_train.id(), spike_train.firing_times())?;

            for neuron in self.neurons.iter_mut() {
                neuron.add_input_spikes_for_source(spike_train.id(), spike_train.firing_times());
            }
        }

        // Compute the minimum delays between each pair of neurons using Floyd-Warshall algorithm
        let min_delays = self.min_delays();

        let mut time = program.start();

        while time < program.end() {
            // Collect the candidate next spikes from all neurons, using parallel processing if the number of neurons is large
            let candidate_next_spikes = match self.neurons.len() > MIN_PARALLEL_NEURONS {
                true => self
                    .neurons()
                    .par_iter()
                    .filter_map(|neuron| neuron.next_spike(time).map(|t| (neuron.id(), t)))
                    .collect::<Vec<(usize, f64)>>(),
                false => self
                    .neurons()
                    .iter()
                    .filter_map(|neuron| neuron.next_spike(time).map(|t| (neuron.id(), t)))
                    .collect::<Vec<(usize, f64)>>(),
            };

            // If no neuron can fire, we're done
            if candidate_next_spikes.is_empty() {
                info!("Network activity has ceased...");
                return Ok(());
            }

            // Accept as many spikes as possible at the current time
            let next_spikes: Vec<(usize, f64)> =
                match self.neurons.len() > MIN_PARALLEL_NEURONS {
                    true => candidate_next_spikes
                        .par_iter()
                        .filter(|(target_id, target_ft)| {
                            candidate_next_spikes.iter().all(|(source_id, source_ft)| {
                                match min_delays.get(&(*source_id, *target_id)) {
                                    Some(min_delay) => *target_ft <= *source_ft + min_delay,
                                    None => true,
                                }
                            })
                        })
                        .cloned()
                        .collect(),
                    false => candidate_next_spikes
                        .iter()
                        .filter(|(target_id, target_ft)| {
                            candidate_next_spikes.iter().all(|(source_id, source_ft)| {
                                match min_delays.get(&(*source_id, *target_id)) {
                                    Some(min_delay) => *target_ft <= *source_ft + min_delay,
                                    None => true,
                                }
                            })
                        })
                        .cloned()
                        .collect(),
                };

            // Get the greatest among all accepted spikes
            time = next_spikes
                .iter()
                .fold(f64::NEG_INFINITY, |acc, (_, t)| acc.max(*t));

            for (id, t) in next_spikes.iter() {
                self.fires(*id, *t, normal.sample(rng))?;
                for neuron in self.neurons.iter_mut() {
                    neuron.add_input_spikes_for_source(*id, &[*t]);
                }
            }

            // Check if it's time to log progress
            if time - last_log_time >= log_interval {
                let progress = ((time - program.start()) / total_duration) * 100.0;
                debug!(
                    "Simulation progress: {:.2}% (Time: {:.2}/{:.2})",
                    progress,
                    time,
                    program.end()
                );
                last_log_time = time;
            }
        }

        info!("Simulation completed successfully!");
        Ok(())
    }

    pub fn memorize_periodic_spike_trains(
        &mut self,
        spike_trains: &[SpikeTrain], // to be adapted to &Vec<HashMap<usize, Vec<f64>>> for multiple memories
        period: f64,
        lim_weights: (f64, f64),
        max_level: f64,
        min_slope: f64,
        half_width: f64,
        objective: Objective
    ) -> Result<(), SNNError> {
        for neuron in self.neurons.iter_mut() {
            info!("Optimizing neuron {}", neuron.id());
            neuron.memorize_periodic_spike_trains(
                spike_trains,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                objective
            )?;
            info!("Neuron {} successfully optimized", neuron.id());
        }
        Ok(())
    }

    // pub fn memorize_many_periodic_spike_trains(
    //     &mut self,
    //     id: usize,
    //     spike_trains: &[SpikeTrain], // to be adapted to &Vec<HashMap<usize, Vec<f64>>> for multiple memories
    //     period: f64,
    //     lim_weights: (f64, f64),
    //     max_level: f64,
    //     min_slope: f64,
    //     half_width: f64,
    //     feas_only: bool,
    // ) -> Result<(), SNNError> {
    //     todo!();
    // }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::spike_train::SpikeTrain;

    const SEED: u64 = 42;

    #[test]
    fn test_network_add_neuron() {
        let mut network = Network::new();

        network.add_neuron(0);
        network.add_neuron(7);
        network.add_neuron(42);

        assert_eq!(network.num_neurons(), 3);
        assert_eq!(network.num_connections(), 0);
    }

    #[test]
    fn test_network_add_connection() {
        let mut network = Network::new();

        network.add_connection(0, 1, 1.0, 1.0).unwrap();
        network.add_connection(2, 3, -1.0, 1.0).unwrap();
        network.add_connection(0, 3, 1.0, 0.0).unwrap();
        network.add_connection(2, 3, 1.0, 0.25).unwrap();
        network.add_connection(2, 3, 1.0, 5.0).unwrap();
        assert_eq!(
            network.add_connection(2, 3, 1.0, -1.0).unwrap_err(),
            SNNError::InvalidDelay
        );

        assert_eq!(network.num_neurons(), 4);
        assert_eq!(network.num_connections(), 5);

        assert_eq!(network.neuron(0).unwrap().num_inputs(), 0);
        assert_eq!(network.neuron(3).unwrap().num_inputs(), 4);

        // let connections_from_to = network.connections_from_to(2, 3).unwrap();
        // assert_eq!(
        //     connections_from_to.first().unwrap(),
        //     &Connection::build(1.0, 0.25).unwrap()
        // );
        // assert_eq!(
        //     connections_from_to.last().unwrap(),
        //     &Connection::build(1.0, 5.0).unwrap()
        // );
    }

    #[test]
    fn test_network_save_load() {
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
    fn test_network_rand() {
        let mut rng = StdRng::seed_from_u64(SEED);

        let network = Network::rand(
            277,
            769,
            (-0.1, 0.1),
            (0.1, 10.0),
            Topology::Random,
            &mut rng,
        )
        .unwrap();
        assert_eq!(network.num_neurons(), 277);
        assert_eq!(network.num_connections(), 769);
        assert!(network
            .neurons()
            .iter()
            .all(
                |neuron| neuron.inputs().iter().all(|input| input.weight() >= -0.1
                    && input.weight() <= 0.1
                    && input.delay() >= 0.1
                    && input.delay() <= 10.0)
            ));

        let network = Network::rand(
            20,
            400,
            (-0.1, 0.1),
            (0.1, 10.0),
            Topology::FinFout,
            &mut rng,
        )
        .unwrap();
        assert_eq!(network.num_neurons(), 20);
        assert_eq!(network.num_connections(), 400);
        assert!((0..20).all(|id| {
            network.num_connections_from(id) == 20 && network.num_connections_to(id) == 20
        }));

        let network =
            Network::rand(20, 400, (-0.1, 0.1), (0.1, 10.0), Topology::Fin, &mut rng).unwrap();
        assert_eq!(network.num_neurons(), 20);
        assert_eq!(network.num_connections(), 400);
        assert!((0..20).all(|id| { network.num_connections_to(id) == 20 }));

        let network =
            Network::rand(20, 400, (-0.1, 0.1), (0.1, 10.0), Topology::Fout, &mut rng).unwrap();
        assert_eq!(network.num_neurons(), 20);
        assert_eq!(network.num_connections(), 400);
        assert!((0..20).all(|id| { network.num_connections_from(id) == 20 }));
    }

    #[test]
    fn test_network_min_delays() {
        let mut network = Network::new();

        network.add_connection(0, 1, 1.0, 1.0).unwrap();
        network.add_connection(0, 2, 1.0, 5.0).unwrap();
        network.add_connection(0, 3, 1.0, 4.0).unwrap();
        network.add_connection(1, 3, 1.0, 1.0).unwrap();
        network.add_connection(3, 0, 1.0, 0.5).unwrap();
        network.add_connection(3, 2, 1.0, 2.0).unwrap();
        network.add_connection(3, 2, 1.0, 0.25).unwrap();
        network.add_connection(3, 3, 1.0, 2.0).unwrap();

        let min_delays = network.min_delays();
        assert_eq!(min_delays.get(&(0, 0)), Some(&0.0));
        assert_eq!(min_delays.get(&(0, 1)), Some(&1.0));
        assert_eq!(min_delays.get(&(0, 2)), Some(&2.25));
        assert_eq!(min_delays.get(&(0, 3)), Some(&2.0));
        assert_eq!(min_delays.get(&(1, 0)), Some(&1.5));
        assert_eq!(min_delays.get(&(1, 1)), Some(&0.0));
        assert_eq!(min_delays.get(&(1, 2)), Some(&1.25));
        assert_eq!(min_delays.get(&(1, 3)), Some(&1.0));
        assert_eq!(min_delays.get(&(2, 0)), None);
        assert_eq!(min_delays.get(&(2, 1)), None);
        assert_eq!(min_delays.get(&(2, 2)), Some(&0.0));
        assert_eq!(min_delays.get(&(2, 3)), None);
        assert_eq!(min_delays.get(&(3, 0)), Some(&0.5));
        assert_eq!(min_delays.get(&(3, 1)), Some(&1.5));
        assert_eq!(min_delays.get(&(3, 2)), Some(&0.25));
        assert_eq!(min_delays.get(&(3, 3)), Some(&0.0));
    }

    #[test]
    fn test_network_run_with_tiny_network() {
        let mut network = Network::new();

        network.add_connection(0, 2, 0.5, 0.5).unwrap();
        network.add_connection(1, 2, 0.5, 0.25).unwrap();
        network.add_connection(1, 2, -0.75, 3.5).unwrap();
        network.add_connection(2, 3, 2.0, 1.0).unwrap();
        network.add_connection(0, 3, -1.0, 2.5).unwrap();

        let spike_trains = vec![
            SpikeTrain::build(0, &[0.5]).unwrap(),
            SpikeTrain::build(1, &[0.75]).unwrap(),
        ];
        let program = SimulationProgram::build(0.0, 10.0, 0.0, &spike_trains).unwrap();
        let mut rng = StdRng::seed_from_u64(SEED);

        network.run(&program, &mut rng).unwrap();
        assert_eq!(network.firing_times(0).unwrap(), &[0.5]);
        assert_eq!(network.firing_times(1).unwrap(), &[0.75]);
        assert_eq!(network.firing_times(2).unwrap(), &[2.0]);
        assert_eq!(network.firing_times(3).unwrap(), &[4.0]);
    }

    #[test]
    fn test_network_run_with_empty_network() {
        let mut network = Network::new();

        let spike_trains = vec![];
        let program = SimulationProgram::build(0.0, 1.0, 0.0, &spike_trains).unwrap();
        let mut rng = StdRng::seed_from_u64(SEED);

        network.run(&program, &mut rng).unwrap();
    }

    #[test]
    fn test_network_run_with_disconnected_network() {
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
        network.run(&program, &mut rng).unwrap();
        assert_eq!(network.firing_times(0).unwrap(), &[0.0, 2.0, 5.0]);
        assert_eq!(network.firing_times(1).unwrap(), &[1.0, 7.0]);
        assert!(network.firing_times(2).unwrap().is_empty());
    }

    #[test]
    fn test_network_memorize_empty_periodic_spike_trains() {
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut network = Network::rand(
            100,
            500 * 100,
            (-0.2, 0.2),
            (0.1, 10.0),
            Topology::Fin,
            &mut rng,
        )
        .unwrap();
        let spike_trains = SpikeTrain::rand(200, 100.0, 0.0, &mut rng).unwrap();

        network
            .memorize_periodic_spike_trains(
                &spike_trains,
                100.0,
                (-0.2, 0.2),
                0.0,
                0.2,
                0.2,
                Objective::None
            )
            .unwrap();
    }

    // #[test]
    // fn test_network_memorize_rand_periodic_spike_trains() {
    //     let mut rng = StdRng::seed_from_u64(SEED);
    //     let mut network = Network::rand(
    //         200,
    //         500 * 200,
    //         (-0.2, 0.2),
    //         (0.1, 10.0),
    //         Topology::Fin,
    //         &mut rng,
    //     )
    //     .unwrap();
    //     let spike_trains = SpikeTrain::rand(200, 50.0, 0.2, &mut rng).unwrap();

    //     network
    //         .memorize_periodic_spike_trains(
    //             &spike_trains,
    //             50.0,
    //             (-0.2, 0.2),
    //             0.0,
    //             0.2,
    //             0.2,
    //             &mut rng,
    //         )
    //         .unwrap();
    //     todo!("Implement eigenvalue computation for the network")
    // }
}
