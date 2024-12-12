//! Module implementing the spiking neural networks.

use core::{f64, num};
use itertools::izip;
use serde::de::Deserializer;
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs::{File, FileTimes};

use grb::expr::{LinExpr, QuadExpr};
use grb::ConstrSense;
use grb::ModelSense;
use grb::{add_ctsvar, c};
use grb::{Model, Var};

use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand_distr::Normal;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use rand::Rng;

use rayon::prelude::*;

use crate::spike_train;

// use super::connection::{Connection, ConnectionFrom, ConnectionTo};
use super::error::SNNError;
use super::neuron::{Neuron, FIRING_THRESHOLD, REFRACTORY_PERIOD};
use super::utils::mod_dist;

use super::optimizer::OptimConfig;
use super::simulator::SimulationProgram;

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

/// Represents a connection between two neurons in a network.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Connection {
    /// Connection weight
    weight: f64,
    /// Connection delay (must be non-negative)
    delay: f64,
}

impl Connection {
    /// Create a new connection with the specified parameters.
    /// Returns an error if the delay is negative.
    pub fn build(weight: f64, delay: f64) -> Result<Self, SNNError> {
        if delay < 0.0 {
            return Err(SNNError::InvalidDelay);
        }

        Ok(Connection { weight, delay })
    }

    /// Returns the weight of the connection.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Returns the delay of the connection.
    pub fn delay(&self) -> f64 {
        self.delay
    }
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
            .map(|(&(source_id, target_id), connections)| {
                (format!("{} -> {}", source_id, target_id), connections)
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
                let key = key.trim_matches(|c| c == ' ').replace("->", ",");
                let mut parts = key.split(',');
                let source_id = parts.next().unwrap().trim().parse().unwrap();
                let target_id = parts.next().unwrap().trim().parse().unwrap();
                ((source_id, target_id), connections)
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
            neurons: HashMap::new(),
            connections: HashMap::new(),
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
    ) -> Result<(), SNNError> {
        self.add_neuron(source_id);
        self.add_neuron(target_id);

        match self
            .connections
            .entry((source_id, target_id))
            .or_insert(vec![])
            .binary_search_by(|connection| connection.delay().partial_cmp(&delay).unwrap())
        {
            Ok(pos) | Err(pos) => {
                let connection = Connection::build(weight, delay)?;
                self.connections
                    .get_mut(&(source_id, target_id))
                    .unwrap()
                    .insert(pos, connection);
            }
        };

        Ok(())
    }

    /// Add a firing time to the neuron with the given id.
    pub fn add_firing_time(&mut self, id: usize, t: f64) -> Result<(), SNNError> {
        if let Some(neuron) = self.neurons.get_mut(&id) {
            neuron.add_firing_time(t)?;
        }
        Ok(())
    }

    pub fn firing_times(&self, id: usize) -> Option<&[f64]> {
        self.neurons.get(&id).map(|neuron| neuron.firing_times())
    }

    /// Add a firing time to the neuron with the given id.
    fn fires(&mut self, id: usize, t: f64, noise: f64) -> Result<(), SNNError> {
        if let Some(neuron) = self.neurons.get_mut(&id) {
            neuron.fires(t, noise)?;
            Ok(())
        } else {
            return Err(SNNError::NeuronNotFound);
        }
    }

    /// Add inputs to all neurons that receive input from the neuron with the specified id.
    pub fn add_input_spikes(&mut self, source_id: usize, t: f64) -> Result<(), SNNError> {
        for (target_id, neuron) in self.neurons.iter_mut() {
            if let Some(connections) = self.connections.get(&(source_id, *target_id)) {
                for connection in connections {
                    neuron.add_input_spike(connection.weight(), t + connection.delay())?;
                }
            }
        }
        Ok(())
    }

    /// Read-only access to the neurons in the network.
    pub fn neurons(&self) -> &HashMap<usize, Neuron> {
        &self.neurons
    }

    /// Read-only access to the connections in the network.
    pub fn connections(&self) -> &HashMap<(usize, usize), Vec<Connection>> {
        &self.connections
    }

    // /// Returns a collection of connections originating from the neuron with the specified id.
    // pub fn connections_from(&self, id: usize) -> Vec<ConnectionTo> {
    //     self.connections
    //         .iter()
    //         .filter(|&((source_id, _), _)| *source_id == id)
    //         .flat_map(|((_, target_id), connections)| {
    //             connections.iter().map(|connection| {
    //                 ConnectionTo::new(*target_id, connection.weight(), connection.delay())
    //             })
    //         })
    //         .collect()
    // }

    // /// Returns a collection of connections targeting the neuron with the specified id.
    // pub fn connections_to(&self, id: usize) -> Vec<ConnectionFrom> {
    //     self.connections
    //         .iter()
    //         .filter(|&((_, target_id), _)| *target_id == id)
    //         .flat_map(|((source_id, _), connections)| {
    //             connections.iter().map(|connection| {
    //                 ConnectionFrom::new(*source_id, connection.weight(), connection.delay())
    //             })
    //         })
    //         .collect()
    // }

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

    /// Returns the number of connections targeting the neuron with the specified id.
    pub fn num_connections_to(&self, id: usize) -> usize {
        self.connections.iter().filter(|&((_, target_id), _)| *target_id == id).fold(
            0,
            |acc, (_, connections)| acc + connections.len(),
        )
    }

    /// Returns the number of connections originating from the neuron with the specified id.
    pub fn num_connections_from(&self, id: usize) -> usize {
        self.connections.iter().filter(|&((source_id, _), _)| *source_id == id).fold(
            0,
            |acc, (_, connections)| acc + connections.len(),
        )
    }

    /// Returns the minimum delay between each pair of neurons in the network.
    /// The function uses the Floyd-Warshall algorithm to compute the minimum delays.
    pub fn min_delays(&self) -> HashMap<(usize, usize), f64> {
        let mut min_delays: HashMap<(usize, usize), f64> = HashMap::new();

        for &source_id in self.neurons.keys() {
            for &target_id in self.neurons.keys() {
                match self.connections.get(&(source_id, target_id)) {
                    Some(connections) => {
                        min_delays
                            .insert((source_id, target_id), connections.first().unwrap().delay());
                    }
                    None => {
                        min_delays.insert((source_id, target_id), f64::INFINITY);
                    }
                }
            }
        }

        for &inter_id in self.neurons.keys() {
            for &source_id in self.neurons.keys() {
                for &target_id in self.neurons.keys() {
                    let source_target_delay = *min_delays.get(&(source_id, target_id)).unwrap();
                    let source_inter_target_delay =
                        *min_delays.get(&(source_id, inter_id)).unwrap()
                            + *min_delays.get(&(inter_id, target_id)).unwrap();
                    if source_target_delay > source_inter_target_delay {
                        min_delays.insert((source_id, target_id), source_inter_target_delay);
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

        // Set up neuron control
        for spike_train in program.spike_trains() {
            for t in spike_train.firing_times() {
                self.add_firing_time(spike_train.id(), *t)?;
                self.add_input_spikes(spike_train.id(), *t)?;
            }
        }

        // Compute the minimum delays between each pair of neurons using Floyd-Warshall algorithm
        let min_delays = self.min_delays();

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
                    .filter_map(|(id, neuron)| neuron.next_spike(time).map(|t| (*id, t)))
                    .collect::<Vec<(usize, f64)>>(),
                false => self
                    .neurons()
                    .iter()
                    .filter_map(|(id, neuron)| neuron.next_spike(time).map(|t| (*id, t)))
                    .collect::<Vec<(usize, f64)>>(),
            };

            // If no neuron can fire, we're done
            if candidate_next_spikes.is_empty() {
                println!("Network activity has ceased...");
                return Ok(());
            }

            // Accept as many spikes as possible at the current time
            let next_spikes: Vec<(usize, f64)> = match self.neurons.len() > MIN_PARALLEL_NEURONS {
                true => candidate_next_spikes
                    .par_iter()
                    .filter(|(id_target, t_target)| {
                        candidate_next_spikes.iter().all(|(id_source, t_source)| {
                            match min_delays.get(&(*id_source, *id_target)) {
                                Some(min_delay) => *t_target <= *t_source + min_delay,
                                None => true,
                            }
                        })
                    })
                    .map(|(id_target, t_target)| (*id_target, *t_target))
                    .collect(),
                false => candidate_next_spikes
                    .iter()
                    .filter(|(id_target, t_target)| {
                        candidate_next_spikes.iter().all(|(id_source, t_source)| {
                            match min_delays.get(&(*id_source, *id_target)) {
                                Some(min_delay) => *t_target <= *t_source + min_delay,
                                None => true,
                            }
                        })
                    })
                    .map(|(id_target, t_target)| (*id_target, *t_target))
                    .collect(),
            };
            // let mut next_spikes = vec![];
            // for (id_target, t_target) in candidate_next_spikes.iter() {
            //     if candidate_next_spikes.iter().all(|(id_source, t_source)| {
            //         match self.connections.get(&(*id_source, *id_target)) {
            //             Some(connections) => {
            //                 *t_target <= *t_source + connections.first().unwrap().delay()
            //             }
            //             None => true,
            //         }
            //     }) {
            //         next_spikes.push((*id_target, *t_target));
            //     }
            // }

            // Get the greatest among all accepted spikes
            time = next_spikes
                .iter()
                .fold(f64::NEG_INFINITY, |acc, (_, t)| acc.max(*t));

            for (id, t) in next_spikes.iter() {
                self.fires(*id, *t, normal.sample(rng))?;
                self.add_input_spikes(*id, *t)?;
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

    // /// To do at the neuron level...
    // pub fn memorize<R: Rng>(
    //     &mut self,
    //     spike_trains: HashMap<usize, Vec<f64>>,
    //     optim_config: OptimConfig,
    // ) -> Result<(), SNNError> {
    //     // iterate over all neurons
    //     // for each neuron, take the corresponding spike train to termine the potential template
    //     // only set constraints at input times and at exact firing times
    //     for (id, neuron) in self.neurons.iter() {
    //         // Get all connections to the neuron
    //         let inputs = self.connections_to(*id);

    //         let mut model = Model::new(format!("neuron_{}", id).as_str()).unwrap();

    //         // Add one decision variable for each input
    //         let weights: Vec<Var> = inputs
    //             .iter()
    //             .map(|_| {
    //                 add_ctsvar!(model, bounds: optim_config.weight_min()..optim_config.weight_max())
    //                     .unwrap()
    //             })
    //             .collect();
    //         println!("{:?}", weights);

    //         // Set the objective function
    //         if optim_config.l2_norm() {
    //             let mut objective = QuadExpr::new();
    //             for weight in weights.iter() {
    //                 objective.add_qterm(1.0, *weight, *weight);
    //             }
    //             model
    //                 .set_objective(objective, ModelSense::Minimize)
    //                 .unwrap();
    //         }

    //         // Collect all times an input arrives to the neuron
    //         let mut times = vec![];
    //         for input in inputs {
    //             if let Some(firing_times) = spike_trains.get(&input.source_id()) {
    //                 times.extend(firing_times.iter().map(|t| t + input.delay()));
    //             }
    //         }
    //         // times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    //         // Add linear equality constraints at every firing time
    //         let firing_times = spike_trains.get(&id).unwrap();

    //         // Add linear inequality constraints
    //         for t in inputs.iter().flat_map(|input| {
    //             spike_trains
    //                 .get(&input.source_id())
    //                 .unwrap()
    //                 .into_iter()
    //                 .map(|t| t + input.delay())
    //         }) {
    //             // If close to firing times, add a minimum slope constraint
    //             if firing_times.iter().any(|ft| {
    //                 mod_dist(*ft, t, spike_trains.period()) < optim_config.activity_window()
    //             }) {
    //                 let mut lin_expr = LinExpr::new();
    //                 for (weight, input) in izip!(weights.iter(), inputs.iter()) {
    //                     // lin_expr.add_term()
    //                     todo!();
    //                 }
    //             }
    //             // Otherwise, if not in refractory period, add a minimum gap constraint
    //             else if firing_times
    //                 .iter()
    //                 .map(|ft| t - ft)
    //                 .all(|dt| (dt > REFRACTORY_PERIOD) || dt < 0.0)
    //             {
    //                 {
    //                     let mut lin_expr = LinExpr::new();
    //                     for (weight, input) in izip!(weights.iter(), inputs.iter()) {
    //                         // lin_expr.add_term()
    //                         todo!();
    //                     }
    //                 }
    //             }
    //         }
    //         // let constraints = model.add_constr("c0", c!(w[0] + w[2] >= 1.0));
    //     }
    //     Ok(())
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
    fn test_connection_build() {
        let connection = Connection::build(0.5, 1.0).unwrap();
        assert_eq!(connection.weight, 0.5);
        assert_eq!(connection.delay, 1.0);
    }

    #[test]
    fn test_connection_build_invalid_delay() {
        let connection = Connection::build(0.5, -1.0);
        assert_eq!(connection, Err(SNNError::InvalidDelay));
    }

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

        assert_eq!(
            network.connections.get(&(0, 3)).unwrap(),
            &[Connection {
                weight: 1.0,
                delay: 0.0
            }]
        );
        assert_eq!(network.connections.get(&(1, 0)), None);
        assert_eq!(
            network.connections.get(&(2, 3)).unwrap(),
            &[
                Connection {
                    weight: 1.0,
                    delay: 0.25
                },
                Connection {
                    weight: -1.0,
                    delay: 1.0
                },
                Connection {
                    weight: 1.0,
                    delay: 5.0
                }
            ]
        );
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
    fn test_network_rand_network() {
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
            .all(|((source_id, target_id), connections)| {
                *source_id < 277
                    && *target_id < 277
                    && connections.iter().all(|connection| {
                        connection.weight() >= -0.1
                            && connection.weight() <= 0.1
                            && connection.delay() >= 0.1
                            && connection.delay() <= 10.0
                    })
            }));

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

        assert_eq!(network.add_connection(0, 1, 1.0, 1.0), Ok(()));
        assert_eq!(network.add_connection(0, 3, 1.0, 4.0), Ok(()));
        assert_eq!(network.add_connection(1, 3, 1.0, 2.0), Ok(()));
        assert_eq!(network.add_connection(3, 0, 1.0, 0.5), Ok(()));
        assert_eq!(network.add_connection(3, 2, 1.0, 2.0), Ok(()));
        assert_eq!(network.add_connection(3, 2, 1.0, 0.25), Ok(()));
        assert_eq!(network.add_connection(3, 3, 1.0, 2.0), Ok(()));

        let min_delays = network.min_delays();
        assert_eq!(*min_delays.get(&(0, 0)).unwrap(), 3.5);
        assert_eq!(*min_delays.get(&(0, 1)).unwrap(), 1.0);
        assert_eq!(*min_delays.get(&(0, 2)).unwrap(), 3.25);
        assert_eq!(*min_delays.get(&(0, 3)).unwrap(), 3.0);
        assert_eq!(*min_delays.get(&(1, 0)).unwrap(), 2.5);
        assert_eq!(*min_delays.get(&(1, 1)).unwrap(), 3.5);
        assert_eq!(*min_delays.get(&(1, 2)).unwrap(), 2.25);
        assert_eq!(*min_delays.get(&(1, 3)).unwrap(), 2.0);
        assert_eq!(*min_delays.get(&(2, 0)).unwrap(), f64::INFINITY);
        assert_eq!(*min_delays.get(&(2, 1)).unwrap(), f64::INFINITY);
        assert_eq!(*min_delays.get(&(2, 2)).unwrap(), f64::INFINITY);
        assert_eq!(*min_delays.get(&(2, 3)).unwrap(), f64::INFINITY);
        assert_eq!(*min_delays.get(&(3, 0)).unwrap(), 0.5);
        assert_eq!(*min_delays.get(&(3, 1)).unwrap(), 1.5);
        assert_eq!(*min_delays.get(&(3, 2)).unwrap(), 0.25);
        assert_eq!(*min_delays.get(&(3, 3)).unwrap(), 2.0);
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
}
