//! Module implementing the spiking neural networks.

use itertools::Itertools;

use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

// use std::sync::Arc;
// use tokio::sync::broadcast;
// use tokio::task;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::{Distribution, Normal};
// use rand_core::RngCore;

use super::connection::{self, Connection, ConnectionError};
use super::neuron::Neuron;
use super::spike_train::SpikeTrainError;
use crate::core::{FIRING_THRESHOLD, REFRACTORY_PERIOD};
use crate::simulator::simulator::{SimulationError, SimulationProgram};
use crate::simulator::TIME_STEP;

/// Represents a spiking neural network.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Network {
    neurons: Vec<Neuron>,
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
            .map(|id| Neuron::new(id))
            .collect::<Vec<Neuron>>();

        // for connection in &connections {
        //     neurons
        //         .iter_mut()
        //         .find(|neuron| neuron.id() == connection.target_id())
        //         .unwrap()
        //         .add_input(connection)
        //         .unwrap();
        // }

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

    /// Add a neuron with the given id to the network.
    /// If such a neuron already exists, do nothing.
    pub fn add_neuron(&mut self, id: usize) {
        match self.neurons.iter().find(|neuron| neuron.id() == id) {
            Some(_) => (),
            None => self.neurons.push(Neuron::new(id)),
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
    ) -> Result<(), ConnectionError> {
        let connection = connection::Connection::build(source_id, target_id, weight, delay)?;

        self.add_neuron(source_id);
        self.add_neuron(target_id);

        // self.neurons
        //     .iter_mut()
        //     .find(|neuron| neuron.id() == target_id)
        //     .unwrap()
        //     .add_input(&connection)?;

        self.connections.push(connection);

        Ok(())
    }

    /// Add a firing time to each inputs whose source id is equal to the given id.
    pub fn add_input_firing_time(&mut self, id: usize, t: f64) {
        for connection in self.connections.iter().filter(|c| c.source_id() == id) {
            self.neurons
                .iter_mut()
                .find(|neuron| neuron.id() == connection.target_id())
                .unwrap()
                .add_input(id, connection.weight(), t + connection.delay());
        }
    }

    /// Read-only access to the neurons in the network.
    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }

    /// Read-only access to the connections in the network.
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Add a firing time to the neuron with the given id.
    pub fn add_firing_time(&mut self, id: usize, t: f64) -> Result<(), SpikeTrainError> {
        self.neurons
            .iter_mut()
            .find(|neuron| neuron.id() == id)
            .unwrap()
            .add_firing_time(t)?;

        Ok(())
    }

    /// Returns the number of neurons in the network.
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Returns the number of connections in the network.
    pub fn num_connections(&self) -> usize {
        self.connections.len()
    }

    /// Returns the number of connections with the specified target id.
    pub fn num_inputs(&self, target_id: usize) -> usize {
        self.connections
            .iter()
            .filter(move |c| c.target_id() == target_id)
            .count()
    }

    /// Returns the number of connections with the specified source id.
    pub fn num_outputs(&self, source_id: usize) -> usize {
        self.connections
            .iter()
            .filter(move |c| c.source_id() == source_id)
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

        let mut time = program.start();

        while time < program.end() {
            let min_item = self
                .neurons
                .iter()
                .filter_map(|neuron| (neuron.step(&time, &program.end())))
                .min_by(|(_, t1), (_, t2)| t1.partial_cmp(t2).unwrap());

            match min_item {
                Some((id, t)) => {
                    if t < program.end() {
                        self.neurons
                            .iter_mut()
                            .find(|neuron| neuron.id() == id)
                            .unwrap()
                            .fires(t, normal.sample(rng))?;

                        
                        for connection in self.connections.iter().filter(|c| c.source_id() == id) {
                            self.neurons
                                .iter_mut()
                                .find(|neuron| neuron.id() == connection.target_id())
                                .unwrap()
                                .add_input(id, connection.weight(), t + connection.delay());
                        }

                        time = t;
                    } else {
                        return Ok(());
                    }
                }
                None => return Ok(()),
            }
        }
        Ok(())
    }
}

/// Run the network asynchronously with the provided simulation details.
///
/// Neurons communicate using a broadcast channel (see [`tokio::sync::broadcast`](https://docs.rs/tokio/latest/tokio/sync/broadcast/index.html)).
/// When a neuron spikes, it sends a message containing its firing time and id through the broadcast channel.
/// Messages are received every `REFRACTORY_PERIOD`, and neurons update their inputs accordingly.
///
/// Neurons are simulated in parallel and synchronized every `min_delay` using a barrier (see [`tokio::sync::Barrier`](https://docs.rs/tokio/latest/tokio/sync/struct.Barrier.html)).
///
/// For reproducibility, the simulation uses a different random number generator for each neuron, seeded with the provided seed and the neuron id.
///
/// Returns an error if the simulation fails.
// pub async fn run(
//     &mut self,
//     program: &SimulationProgram,
//     seed: u64,
// ) -> Result<(), SimulationError> {
//     let num_neurons = self.num_neurons();

//     if num_neurons == 0 {
//         return Ok(());
//     }

//     // Set up neuron control
//     for neuron in self.neurons.iter_mut() {
//         if let Some(firing_times) = program.neuron_control(neuron.id()) {
//             if let Err(e) = neuron.extend_firing_times(firing_times) {
//                 eprintln!(
//                     "Failed to extend firing times of neuron {} with {:#?}: {}",
//                     neuron.id(),
//                     firing_times,
//                     e
//                 );
//                 return Err(SimulationError::InvalidControl);
//             }
//         }
//     }

//     let min_delay = match self
//         .connections
//         .iter()
//         .map(|c| c.delay())
//         .min_by(|a, b| a.partial_cmp(&b).unwrap_or_else(|| panic!("Comparison failed: NaN values should have been caught earlier")))
//     {
//         Some(delay) => delay,
//         None => {
//             println!("Discrete network: neurons cannot communicate with each other");
//             return Ok(())
//         },
//     };

//     // take ownership of neurons
//     let neurons = std::mem::take(&mut self.neurons);

//     // set up broadcast channel for communication and barrier for synchronization
//     let (tx, _) = broadcast::channel::<(usize, f64)>(num_neurons);
//     let barrier = Arc::new(tokio::sync::Barrier::new(num_neurons));

//     let mut handles = vec![];

//     for mut neuron in neurons.into_iter() {
//         println!("Setting up neuron {}", neuron.id());

//         // setup transmitter and receiver for communication
//         let tx = tx.clone();
//         let mut rx = tx.subscribe();

//         // setup barrier for synchronization
//         let barrier = barrier.clone();

//         // simulation interval
//         let start = program.start();
//         let end = program.end();

//         // setup rng for threshold noise
//         let mut rng = ChaChaRng::seed_from_u64(seed + neuron.id() as u64);
//         let normal = Normal::new(0.0, program.threshold_noise()).unwrap();

//         // neuron inputs
//         for id in 0..num_neurons {
//             if let Some(firing_times) = program.neuron_control(id) {
//                 neuron.extend_inputs_firing_times(id, firing_times);
//             }
//         }

//         println!("Neuron {} has been set up", neuron.id());

//         let handle = task::spawn(async move {
//             let mut t = start;
//             let mut last_recv = start;
//             let mut last_sync = start;

//             while t < end {
//                 // Synchronize neurons every min_delay
//                 if t - last_sync > min_delay {
//                     barrier.wait().await;
//                     last_sync = t;
//                 }

//                 // Receive messages from the last REFRACTORY_PERIOD
//                 if t - last_recv > REFRACTORY_PERIOD {
//                     loop {
//                         match rx.try_recv() {
//                             Ok((source_id, firing_time)) => {
//                                 println!(
//                                     "Neuron {} received new message from neuron {}: {}",
//                                     neuron.id(),
//                                     source_id,
//                                     firing_time
//                                 );
//                                 neuron.add_inputs_firing_time(source_id, firing_time);
//                             }
//                             Err(broadcast::error::TryRecvError::Empty) => break,
//                             Err(broadcast::error::TryRecvError::Closed) => return neuron,
//                             Err(broadcast::error::TryRecvError::Lagged(_)) => {
//                                 println!("Neuron {} lagged", neuron.id());
//                             }
//                         };
//                     }
//                     last_recv = t;
//                 }

//                 // Compute neuron activity between t and t + TIME_STEP
//                 if let Some(firing_time) = neuron.step(t, TIME_STEP) {
//                     if let Err(e) = neuron.add_firing_time(firing_time) {
//                         eprintln!("Failed to add firing time to neuron {}: {}", neuron.id(), e);
//                         return neuron;
//                     }
//                     if let Err(e) = tx.send((neuron.id(), firing_time)) {
//                         eprintln!("Failed to send message from neuron {}: {}", neuron.id(), e);
//                         return neuron;
//                     }

//                     neuron.set_threshold(normal.sample(&mut rng) + FIRING_THRESHOLD);
//                     println!("Neuron {} sent new message: {}", neuron.id(), firing_time);
//                 }

//                 t += TIME_STEP;
//             }
//             neuron
//         });

//         handles.push(handle);
//     }

//     for handle in handles {
//         match handle.await {
//             Ok(neuron) => self.neurons.push(neuron),
//             Err(e) => {
//                 eprintln!("Neuron task failed: {}", e);
//                 // Handle the error appropriately
//                 return Err(SimulationError::NeuronTaskFailed);
//             }
//         }
//     }

//     Ok(())
// }
// }

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use super::*;

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

    // #[tokio::test]
    // async fn test_run_with_empty_network() {
    //     let mut network = Network::new(vec![]);

    //     let program = SimulationProgram::build(0.0, 1.0, 0.0, vec![]).unwrap();
    //     let seed = 42;

    //     let result = network.run(&program, seed).await;
    //     println!("{:?}", result);
    //     assert!(result.is_ok());
    // }

    // #[tokio::test]
    // async fn test_run_with_disconnected_network() {
    //     let mut network = Network::new(vec![]);
    //     for i in 0..10 {
    //         network.add_neuron(i);
    //     }

    //     let program = SimulationProgram::build(0.0, 1.0, 0.0, vec![]).unwrap();
    //     let seed = 42;

    //     let result = network.run(&program, seed).await;
    //     println!("{:?}", result);
    //     assert!(result.is_ok());
    // }
}
