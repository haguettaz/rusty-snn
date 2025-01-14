//! This module implements the `Network` structure and some core functionalities.

use core::f64;
use log::{debug, info};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::Normal;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::{FIRING_THRESHOLD, REFRACTORY_PERIOD};

use super::connection::Connection;
use super::error::SNNError;
use super::neuron::Neuron;
use super::optim::Objective;
// use super::simulator::SimulationProgram;
use super::spike_train::*;

/// Minimum number of neurons to use parallel processing.
pub const MIN_PARALLEL_NEURONS: usize = 100;

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

impl Topology {
    /// Returns the topology from the string representation.
    pub fn from_str(s: &str) -> Result<Self, SNNError> {
        match s {
            "rand" => Ok(Topology::Random),
            "fin" => Ok(Topology::Fin),
            "fout" => Ok(Topology::Fout),
            "finfout" => Ok(Topology::FinFout),
            _ => Err(SNNError::InvalidParameters("Invalid topology".to_string())),
        }
    }
}

/// Represents a spiking neural network.
// #[derive(Debug, PartialEq, Clone)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Network {
    // A collection of neurons in the network, indexed by their IDs.
    neurons: Vec<Neuron>,
    // A collection of connections between neurons, indexed by target IDs.
    connections: Vec<Vec<Connection>>,
}

impl Network {
    /// Create a new network with the given number of neurons
    pub fn new(num_neurons: usize) -> Self {
        let neurons = (0..num_neurons).map(|id| Neuron::new(id)).collect();
        Network {
            neurons: neurons,
            connections: vec![vec![]; num_neurons],
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
            return Err(SNNError::InvalidParameters(
                "Connection delay must be non-negative".to_string(),
            ));
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

        let mut network = Network::new(num_neurons);

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

    /// Add a new empty neuron to the network.
    pub fn add_neuron(&mut self) {
        let id = self.num_neurons();
        self.neurons.push(Neuron::new(id));
        // match self.neurons.binary_search_by_key(&id, |neuron| neuron.id()) {
        //     Ok(_) => (),
        //     Err(pos) => {
        //         self.neurons.insert(pos, Neuron::new(id));
        //     }
        // }
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
        // // Add both source and target neurons if they don't exist
        // self.add_neuron(source_id);
        // self.add_neuron(target_id);
        if source_id >= self.num_neurons() || target_id >= self.num_neurons() {
            return Err(SNNError::NeuronNotFound);
        }

        // insert the connection to preserve the delay order
        let pos = match self.connections[target_id]
            .binary_search_by(|connection| connection.delay().partial_cmp(&delay).unwrap())
        {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        // let id = self.num_connections();
        self.connections[target_id]
            .insert(pos, Connection::build(source_id, target_id, weight, delay)?);
        // self.connections[source_id][target_id]
        //     .push(Connection::build(source_id, target_id, weight, delay)?);

        // let target_neuron = self.neuron_mut(target_id)?;
        // self.neurons[target_id].add_input(source_id, weight, delay);

        Ok(())
    }

    /// Borrow the neurons in the network.
    pub fn neurons(&self) -> &Vec<Neuron> {
        &self.neurons
    }

    /// Borrow the connections in the network.
    pub fn connections(&self) -> &Vec<Vec<Connection>> {
        &self.connections
    }

    // /// Returns a slice of firing times of the neuron with the specified id.
    // /// If the neuron does not exist, the function returns an error.
    // fn firing_times(&self, id: usize) -> Result<&[f64], SNNError> {
    //     let neuron = self.neuron(id)?;
    //     Ok(neuron.firing_times())
    // }

    // /// Add a firing time to the neuron with the given id.
    // fn add_firing_time(&mut self, id: usize, t: f64) -> Result<(), SNNError> {
    //     self.neurons[id].add_firing_time(t)?;
    //     Ok(())
    // }

    // /// Add a firing time to the neuron with the given id.
    // fn extend_firing_times(&mut self, id: usize, firing_times: &[f64]) -> Result<(), SNNError> {
    //     self.neurons[id].extend_firing_times(firing_times)?;
    //     Ok(())
    // }

    // Clear the firing times of all neurons in the network.
    pub fn clear_all_firing_times(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.clear_firing_times();
        }
    }

    // Clear the inspikes of all neurons in the network.
    pub fn clear_all_inspikes_from_spike_train(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.clear_inspikes();
        }
    }

    // Extend the firing times of all neurons in the network from the provided spike train.
    // The spikes are assumed to be sorted by time.
    pub fn extend_all_firing_times_from_spike_train(
        &mut self,
        spike_train: &[Spike],
    ) -> Result<(), SNNError> {
        for spike in spike_train.iter() {
            self.neurons[spike.source_id()].add_firing_time(spike.time())?;
        }
        Ok(())
    }

    pub fn extend_all_inspikes_from_spike_train(&mut self, spike_train: &[Spike]) {
        for neuron in self.neurons.iter_mut() {
            let mut new_inspikes =
                extract_spike_train_into_inspikes(spike_train, &self.connections[neuron.id()]);
            neuron.extend_inspikes(&mut new_inspikes);
        }
    }

    // /// Add a firing time to the neuron with the given id.
    // fn fires(&mut self, id: usize, t: f64, noise: f64) -> Result<(), SNNError> {
    //     let neuron = self.neuron_mut(id)?;
    //     neuron.fires(t, noise)
    // }

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
        self.connections
            .iter()
            .flat_map(|connection| {
                connection.iter()
                // .flat_map(|connection_to| connection_to.iter())
            })
            .count()
        // self.neurons
        //     .iter()
        //     .fold(0, |acc, neuron| acc + neuron.num_inputs())
    }

    /// Returns the number of connections targeting the specified neuron.
    pub fn num_connections_to(&self, target_id: usize) -> usize {
        self.connections[target_id].len()
        // .iter()
        // .flat_map(|connection_to| connection_to.iter())
        // .count()
        // self.neuron(target_id)
        //     .map_or(0, |neuron| neuron.num_inputs())
    }

    /// Returns the number of connections originating from the specified neuron.
    pub fn num_connections_from(&self, source_id: usize) -> usize {
        self.connections
            .iter()
            .flat_map(|connection_to| {
                connection_to
                    .iter()
                    .filter(|connection| connection.source_id() == source_id)
            })
            .count()
        // self.neurons
        //     .iter()
        //     .fold(0, |acc, neuron| acc + neuron.num_inputs_from(source_id))
    }

    /// Returns the number of (parallel) connections between the source and target neurons.
    pub fn num_connections_between(&self, source_id: usize, target_id: usize) -> usize {
        self.connections[target_id]
            .iter()
            .filter(|connection| connection.source_id() == source_id)
            .count()
        // self.connections[source_id][target_id].len()
    }

    pub fn firing_times(&self) -> Vec<Vec<f64>> {
        self.neurons
            .iter()
            .map(|neuron| neuron.firing_times().to_vec())
            .collect()
    }

    /// Returns the minimum delay in spike propagation between each pair of neurons in the network.
    /// The delay from a neuron to itself is zero (refractory period).
    /// Otherwise, the delay is the minimum delay between all possible paths connecting the two neurons.
    /// They are computed using the Floyd-Warshall algorithm.
    fn min_delays(&self) -> Vec<Vec<f64>> {
        let mut min_delays: Vec<Vec<f64>> =
            vec![vec![f64::INFINITY; self.num_neurons()]; self.num_neurons()];

        for target_neuron in self.neurons.iter() {
            for source_neuron in self.neurons.iter() {
                if target_neuron.id() == source_neuron.id() {
                    min_delays[target_neuron.id()][source_neuron.id()] = 0.0;
                } else if let Some(input) = self.connections[target_neuron.id()]
                    .iter()
                    .filter(|input| input.source_id() == source_neuron.id())
                    .min_by(|a, b| a.delay().partial_cmp(&b.delay()).unwrap())
                {
                    min_delays[target_neuron.id()][source_neuron.id()] = input.delay();
                };
            }
        }

        for inter_neuron in self.neurons.iter() {
            for target_neuron in self.neurons.iter() {
                for source_neuron in self.neurons.iter() {
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

        min_delays
    }

    /// Continuous-time event-based simulation of the network.
    pub fn run<R: Rng>(
        &mut self,
        // spike_train: &Vec<Spike>,
        start: f64,
        end: f64,
        threshold_noise: f64,
        rng: &mut R,
    ) -> Result<(), SNNError> {
        let normal = Normal::new(0.0, threshold_noise).unwrap();

        let total_duration = end - start;
        let mut last_log_time = start;
        let log_interval = total_duration / 100.0;

        // Compute the minimum delays between each pair of neurons using Floyd-Warshall algorithm
        let min_delays = self.min_delays();

        let mut time = start;

        while time < end {
            // Collect the candidate next spikes from all neurons, using parallel processing if the number of neurons is large
            // let now = Instant::now();
            let candidate_next_spikes = match self.num_neurons() > MIN_PARALLEL_NEURONS {
                true => self
                    .neurons()
                    .par_iter()
                    .filter_map(|neuron| neuron.next_spike(time))
                    .collect::<Vec<Spike>>(),
                false => self
                    .neurons()
                    .iter()
                    .filter_map(|neuron| neuron.next_spike(time))
                    .collect::<Vec<Spike>>(),
            };
            // let elapsed = now.elapsed();
            // info!("Next spike duration: {:?}", elapsed);

            // If no neuron can fire, we're done
            if candidate_next_spikes.is_empty() {
                info!("Network activity has ceased...");
                return Ok(());
            }

            // Accept as many spikes as possible at the current time
            // let now =  Instant::now();
            let next_spikes: Vec<Spike> = match self.num_neurons() > MIN_PARALLEL_NEURONS {
                true => candidate_next_spikes
                    .par_iter()
                    .filter(|target_spike| {
                        candidate_next_spikes.iter().all(|source_spike| {
                            target_spike.time()
                                <= source_spike.time()
                                    + min_delays[target_spike.source_id()][source_spike.source_id()]
                        })
                    })
                    .cloned()
                    .collect(),
                false => candidate_next_spikes
                    .iter()
                    .filter(|target_spike| {
                        candidate_next_spikes.iter().all(|source_spike| {
                            target_spike.time()
                                <= source_spike.time()
                                    + min_delays[target_spike.source_id()][source_spike.source_id()]
                        })
                    })
                    .cloned()
                    .collect(),
            };
            // let elapsed = now.elapsed();
            // info!("Acceptance duration: {:?}", elapsed);

            // Get the greatest among all accepted spikes
            time = next_spikes
                .iter()
                .map(|spike| spike.time())
                .fold(f64::NEG_INFINITY, f64::max);

            // let now = Instant::now();
            for spike in next_spikes.iter() {
                self.neurons[spike.source_id()].fires(spike.time(), normal.sample(rng))?;

                for neuron in self.neurons.iter_mut() {
                    let mut new_inspikes = self.connections[neuron.id()]
                        .iter()
                        .enumerate()
                        .filter(|(_, input)| input.source_id() == spike.source_id())
                        .map(|(input_id, input)| {
                            InSpike::new(input_id, input.weight(), spike.time() + input.delay())
                        })
                        .collect::<Vec<InSpike>>();
                    neuron.extend_inspikes(&mut new_inspikes);

                    // let mut new_inspikes = self.connections[neuron.id()][spike.source_id()]
                    //     .iter()
                    //     .map(|input| InSpike::new(input.weight(), spike.time() + input.delay()))
                    //     .collect::<Vec<InSpike>>();
                    // neuron.extend_inspikes(&mut new_inspikes);
                }
            }
            // let elapsed = now.elapsed();
            // info!("Fires duration: {:?}", elapsed);

            // Check if it's time to log progress
            if time - last_log_time >= log_interval {
                let progress = ((time - start) / total_duration) * 100.0;
                debug!(
                    "Simulation progress: {:.2}% (Time: {:.2}/{:.2})",
                    progress, time, end
                );
                last_log_time = time;
            }
        }

        info!("Simulation completed successfully!");
        Ok(())
    }

    /// Optimize the network to reproduce the given spike trains.
    /// The function returns the error if the optimization fails.
    /// Otherwise, returns the jitter's stability value (read more...)
    pub fn memorize_periodic_spike_train(
        &mut self,
        spike_train: &Vec<Spike>, // to be adapted to &Vec<HashMap<usize, Vec<f64>>> for multiple memories
        period: f64,
        lim_weights: (f64, f64),
        max_level: f64,
        min_slope: f64,
        half_width: f64,
        objective: Objective,
    ) -> Result<(), SNNError> {
        if max_level >= FIRING_THRESHOLD {
            return Err(SNNError::InvalidParameters(
                "maximum level must be smaller than the nominal firing threshold".to_string(),
            ));
        }
        if min_slope < 0.0 {
            return Err(SNNError::InvalidParameters(
                "minimum slope must be positive".to_string(),
            ));
        }
        if half_width <= 0.0 {
            return Err(SNNError::InvalidParameters(
                "half-width must be positive".to_string(),
            ));
        }
        if half_width > REFRACTORY_PERIOD / 2.0 {
            return Err(SNNError::InvalidParameters(
                "half-width must be larger than half the refractory period".to_string(),
            ));
        }

        for neuron in self.neurons.iter_mut() {
            info!("Optimizing neuron {}", neuron.id());

            // Extract the neuron's firing times from the provided spike train
            let firing_times: Vec<f64> =
                extract_spike_train_into_firing_times(spike_train, neuron.id());
            debug!("Firing times: {:?}", firing_times);

            // Extract the neuron's input spikes from the provided spike train
            let mut inspikes =
                extract_spike_train_into_inspikes(spike_train, &self.connections[neuron.id()]);

            let weights = neuron.memorize_periodic_spike_train(
                &firing_times,
                &mut inspikes,
                period,
                lim_weights,
                max_level,
                min_slope,
                half_width,
                objective,
            )?;
            for (id, weight) in weights.iter().enumerate() {
                self.connections[neuron.id()][id].set_weight(*weight);
            }

            // // Sanity check
            // let inspikes: Vec<InSpike> = self.connections[neuron.id()]
            //     .iter()
            //     .enumerate()
            //     .flat_map(|(id, input)| {
            //         spike_train
            //             .iter()
            //             .filter(|spike| spike.source_id() == input.source_id())
            //             .flat_map(move |spike| {
            //                 (-2..1).map(move |i| {
            //                     InSpike::new(
            //                         id,
            //                         input.weight(),
            //                         spike.time() + input.delay() + i as f64 * period,
            //                     )
            //                 })
            //             })
            //     })
            //     .collect();
            // for &ft in firing_times.iter() {
            //     debug!(
            //         "Potential at firing time {}: {}",
            //         ft,
            //         potential(&inspikes, ft)
            //     );
            // }
            // for &ft in firing_times.iter() {
            //     debug!(
            //         "Potential at time {}: {}",
            //         ft + REFRACTORY_PERIOD,
            //         potential(&inspikes, ft + REFRACTORY_PERIOD)
            //     );
            // }
            // for t in (0..(period as usize)).map(|i| i as f64) {
            //     debug!("Potential at time {}: {}", t, potential(&inspikes, t));
            // }

            info!("Neuron {} successfully optimized", neuron.id());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use tempfile::NamedTempFile;

    use crate::spike_train::rand_spike_train;

    use super::*;

    const SEED: u64 = 42;

    #[test]
    fn test_network_add_neuron() {
        let mut network = Network::new(2);

        assert_eq!(network.num_neurons(), 2);
        assert_eq!(network.num_connections(), 0);

        network.add_neuron();
        network.add_neuron();
        network.add_neuron();

        assert_eq!(network.num_neurons(), 5);
        assert_eq!(network.num_connections(), 0);
    }

    #[test]
    fn test_network_add_connection() {
        let mut network = Network::new(4);

        network.add_connection(0, 1, 1.0, 1.0).unwrap();
        network.add_connection(2, 3, -1.0, 1.0).unwrap();
        network.add_connection(0, 3, 1.0, 0.0).unwrap();
        network.add_connection(2, 3, 1.0, 0.25).unwrap();
        network.add_connection(2, 3, 1.0, 5.0).unwrap();
        assert_eq!(
            network.add_connection(2, 3, 1.0, -1.0).unwrap_err(),
            SNNError::InvalidParameters("Connection delay must be non-negative".to_string())
        );
        assert_eq!(
            network.add_connection(5, 3, 1.0, 0.0).unwrap_err(),
            SNNError::NeuronNotFound
        );

        assert_eq!(network.num_neurons(), 4);
        assert_eq!(network.num_connections(), 5);

        let expected_connections = vec![
            vec![],
            vec![Connection::build(0, 1, 1.0, 1.0).unwrap()],
            vec![],
            vec![
                Connection::build(0, 3, 1.0, 0.0).unwrap(),
                Connection::build(2, 3, 1.0, 0.25).unwrap(),
                Connection::build(2, 3, -1.0, 1.0).unwrap(),
                Connection::build(2, 3, 1.0, 5.0).unwrap(),
            ],
        ];
        assert_eq!(network.connections, expected_connections);

        // assert_eq!(network.neuron(0).unwrap().num_inputs(), 0);
        // assert_eq!(network.neuron(3).unwrap().num_inputs(), 4);

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
        let mut network = Network::new(4);
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
            .connections
            .iter()
            .all(|inputs| inputs.iter().all(|input| {
                input.weight() >= -0.1
                    && input.weight() <= 0.1
                    && input.delay() >= 0.1
                    && input.delay() <= 10.0
            })));

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
        let mut network = Network::new(4);

        network.add_connection(0, 1, 1.0, 1.0).unwrap();
        network.add_connection(0, 2, 1.0, 5.0).unwrap();
        network.add_connection(0, 3, 1.0, 4.0).unwrap();
        network.add_connection(1, 3, 1.0, 1.0).unwrap();
        network.add_connection(3, 0, 1.0, 0.5).unwrap();
        network.add_connection(3, 2, 1.0, 2.0).unwrap();
        network.add_connection(3, 2, 1.0, 0.25).unwrap();
        network.add_connection(3, 3, 1.0, 2.0).unwrap();

        let min_delays = network.min_delays();
        assert_eq!(min_delays[0][0], 0.0);
        assert_eq!(min_delays[1][0], 1.0);
        assert_eq!(min_delays[2][0], 2.25);
        assert_eq!(min_delays[3][0], 2.0);
        assert_eq!(min_delays[0][1], 1.5);
        assert_eq!(min_delays[1][1], 0.0);
        assert_eq!(min_delays[2][1], 1.25);
        assert_eq!(min_delays[3][1], 1.0);
        assert_eq!(min_delays[0][2], f64::INFINITY);
        assert_eq!(min_delays[1][2], f64::INFINITY);
        assert_eq!(min_delays[2][2], 0.0);
        assert_eq!(min_delays[3][2], f64::INFINITY);
        assert_eq!(min_delays[0][3], 0.5);
        assert_eq!(min_delays[1][3], 1.5);
        assert_eq!(min_delays[2][3], 0.25);
        assert_eq!(min_delays[3][3], 0.0);
    }

    #[test]
    fn test_network_run_with_tiny_network() {
        let mut rng = StdRng::seed_from_u64(SEED);

        let mut network = Network::new(4);

        network.add_connection(0, 2, 0.5, 0.5).unwrap();
        network.add_connection(1, 2, 0.5, 0.25).unwrap();
        network.add_connection(1, 2, -0.75, 3.5).unwrap();
        network.add_connection(2, 3, 2.0, 1.0).unwrap();
        network.add_connection(0, 3, -1.0, 2.5).unwrap();

        let spike_train = vec![Spike::new(0, 0.5), Spike::new(1, 0.75)];
        network
            .extend_all_firing_times_from_spike_train(&spike_train)
            .unwrap();
        network.extend_all_inspikes_from_spike_train(&spike_train);

        network.run(0.0, 10.0, 0.0, &mut rng).unwrap();
        assert_eq!(
            network.firing_times(),
            vec![vec![0.5], vec![0.75], vec![2.0], vec![4.0]]
        );
    }

    #[test]
    fn test_network_run_with_empty_network() {
        let mut rng = StdRng::seed_from_u64(SEED);

        let mut network = Network::new(0);

        let spike_train = vec![];
        network
            .extend_all_firing_times_from_spike_train(&spike_train)
            .unwrap();
        network.extend_all_inspikes_from_spike_train(&spike_train);

        network.run(0.0, 10.0, 0.0, &mut rng).unwrap();
    }

    #[test]
    fn test_network_run_with_disconnected_network() {
        let mut rng = StdRng::seed_from_u64(SEED);

        let mut network = Network::new(3);

        let spike_train = vec![
            Spike::new(0, 0.0),
            Spike::new(0, 2.0),
            Spike::new(0, 5.0),
            Spike::new(1, 1.0),
            Spike::new(1, 7.0),
        ];
        network
            .extend_all_firing_times_from_spike_train(&spike_train)
            .unwrap();
        network.extend_all_inspikes_from_spike_train(&spike_train);

        network.run(0.0, 10.0, 0.0, &mut rng).unwrap();
        assert_eq!(
            network.firing_times(),
            vec![vec![0.0, 2.0, 5.0], vec![1.0, 7.0], vec![]]
        );
    }

    // #[test]
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
        let spike_trains = rand_spike_train(200, 100.0, 0.0, &mut rng).unwrap();

        network
            .memorize_periodic_spike_train(
                &spike_trains,
                100.0,
                (-0.2, 0.2),
                0.0,
                0.2,
                0.2,
                Objective::L2Norm,
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
    //     let spike_trains = rand_spike_train(200, 50.0, 0.2, &mut rng).unwrap();

    //     network
    //         .memorize_periodic_spike_train(
    //             &spike_trains,
    //             50.0,
    //             (-0.2, 0.2),
    //             0.0,
    //             0.2,
    //             0.2,
    //             Objective::L2Norm,
    //             &mut rng,
    //         )
    //         .unwrap();
    //     todo!("Implement eigenvalue computation for the network")
    // }
}
