//! Network module with utilities for instantiating and managing networks of neurons.

use itertools::{enumerate, Itertools};
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;


use tokio::sync::broadcast;
use tokio::task;
use tokio::time::{sleep, Duration};
use std::sync::Arc;

// use std::sync::mpsc;
// use std::sync::{Arc, Barrier};
// use std::thread;

use super::error::NetworkError;
use super::neuron::{Neuron, Message};

/// Represents a connection between two neurons in a network.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Connection {
    source_id: usize,
    target_id: usize,
    weight: f64,
    delay: f64,
}

impl Connection {
    /// Create a new connection with the specified parameters.
    pub fn new(source_id: usize, target_id: usize, weight: f64, delay: f64) -> Self {
        Connection {
            source_id,
            target_id,
            weight,
            delay,
        }
    }

    /// Returns the id of the source neuron.
    pub fn source_id(&self) -> usize {
        self.source_id
    }

    /// Returns the id of the target neuron.
    pub fn target_id(&self) -> usize {
        self.target_id
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

/// Represents a spiking neural networks.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Network {
    neurons: Vec<Neuron>,
    connections: Vec<Connection>,
}

impl Network {
    /// Create a new network from lists of neurons and connections.
    ///
    /// # Example
    /// ```rust
    /// use rusty_snn::network::network::{Network, Connection};
    /// use rusty_snn::network::neuron::Neuron;
    ///
    /// let neurons = vec![Neuron::new(0, 1.0), Neuron::new(1, 1.0), Neuron::new(2, 1.0)];
    /// let connections = vec![Connection::new(0, 1, 1.0, 1.0), Connection::new(1, 2, -1.0, 2.0), Connection::new(0, 2, 0.25, 0.5), Connection::new(0, 0, -0.125, 0.25)];
    /// let network = Network::new(neurons, connections);
    ///
    /// assert_eq!(network.num_neurons(), 3);
    /// assert_eq!(network.num_connections(), 4);
    /// ```
    pub fn new(neurons: Vec<Neuron>, connections: Vec<Connection>) -> Self {
        Network {
            neurons,
            connections,
        }
    }

    /// Create a new network from neurons and connections between them.
    /// The function returns an error for invalid neurons or connections.
    /// Because it handles possible errors, it is recommended to use this function instead of the `new` function.
    ///
    /// # Example
    /// ```rust
    /// use rusty_snn::network::network::{Network, Connection};
    /// use rusty_snn::network::neuron::Neuron;
    ///
    /// let neurons = vec![Neuron::new(0, 1.0), Neuron::new(1, 1.0), Neuron::new(2, 1.0)];
    /// let connections = vec![Connection::new(0, 1, 1.0, 1.0), Connection::new(1, 2, -1.0, 2.0), Connection::new(0, 2, 0.25, 0.5), Connection::new(0, 0, -0.125, 0.25)];
    /// let network = Network::build(neurons, connections).unwrap();
    ///
    /// assert_eq!(network.num_neurons(), 3);
    /// assert_eq!(network.num_connections(), 4);
    /// ```
    pub fn build(neurons: Vec<Neuron>, connections: Vec<Connection>) -> Result<Self, NetworkError> {
        for (id, neuron) in enumerate(&neurons) {
            if neuron.id() != id {
                return Err(NetworkError::InvalidNeuronId);
            }
        }

        for connection in &connections {
            if connection.source_id() >= neurons.len() {
                return Err(NetworkError::InvalidSourceId);
            }
            if connection.target_id() >= neurons.len() {
                return Err(NetworkError::InvalidTargetId);
            }
            if connection.delay() < 0.0 {
                return Err(NetworkError::InvalidDelay);
            }
        }
        
        Ok(Network { neurons, connections })
    }

    /// Save the network to a file.
    ///
    /// # Example
    /// ```rust
    /// use std::path::Path;
    /// use rusty_snn::network::network::{Network, Connection};
    /// use rusty_snn::network::neuron::Neuron;
    ///
    /// let neurons = vec![Neuron::new(0, 1.0), Neuron::new(1, 1.0), Neuron::new(2, 1.0)];
    /// let connections = vec![Connection::new(0, 1, 1.0, 1.0), Connection::new(1, 2, -1.0, 2.0), Connection::new(0, 2, 0.25, 0.5), Connection::new(0, 0, -0.125, 0.25)];
    /// let network = Network::new(neurons, connections);
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
    /// The function returns an error for invalid connection.
    pub fn add_connection(&mut self, source_id:usize, target_id:usize, weight:f64, delay:f64) -> Result<(), NetworkError> {
        match self.neurons.get(source_id) {
            None => return Err(NetworkError::InvalidSourceId),
            Some(_) => (),
        }

        match self.neurons.get(target_id) {
            None => return Err(NetworkError::InvalidTargetId),
            Some(_) => (),
        }

        if delay < 0.0 {
            return Err(NetworkError::InvalidDelay);
        }

        self.connections.push(Connection::new(source_id, target_id, weight, delay));
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

    /// Returns a collection of connections with the specified target id.
    fn transmit_to(&self, neuron_id: usize) -> impl Iterator<Item = &Connection> {
        self.connections
            .iter()
            .filter(move |c| c.target_id() == neuron_id)
    }

    /// Returns the number of connections with the specified target id.
    pub fn num_inputs(&self, neuron_id: usize) -> usize {
        self.transmit_to(neuron_id)
            .count()
    }

    /// Returns a collection of connections with the specified source id.
    fn transmit_from(&self, neuron_id: usize) -> impl Iterator<Item = &Connection> {
        self.connections
            .iter()
            .filter(move |c| c.source_id() == neuron_id)
    }

    /// Returns the number of connections with the specified source id.
    pub fn num_outputs(&self, neuron_id: usize) -> usize {
        self.transmit_from(neuron_id)
            .count()
    }

    fn init_neurons(&mut self) {
        for connection in self.connections.iter() {
            self.neurons[connection.target_id()].add_input(
                connection.source_id(),
                connection.weight(),
                connection.delay(),
            );
        }
    }

    // fn init_mpsc_channels(&mut self) -> Result<(), NetworkError> {
    //     // Is it better to use broadcast channels instead?
    //     // A broadcast channel is use to send many values from many producers to many consumers
    //     for neuron in &mut self.neurons {
    //         neuron.reset_senders();
    //     }

    //     for id in 0..self.num_neurons() {
    //         let (tx, rx) = mpsc::channel();
    //         self.neurons[id].set_receiver(rx);
    //         // iterate over all connections whose target has id, and drop duplicated source ids

    //         for connection in self
    //             .connections
    //             .iter()
    //             .filter(|c| c.target_id() == id)
    //             .unique_by(|c| c.source_id())
    //         {
    //             self.neurons[connection.source_id].add_sender(tx.clone());
    //         }
    //     }

    //     Ok(())
    // }

    /// Schedule contains the simulation program with simulation interval, measurement intervals, excitation intervals, ...
    #[tokio::main]
    pub async fn run(&mut self, start: f64, end: f64) -> Result<(), NetworkError> {

        let (tx, _) = broadcast::channel::<Message>(self.num_neurons());

        let barrier = Arc::new(tokio::sync::Barrier::new(self.num_neurons()));

        let mut handles = vec![];
        let dt = 0.001;

        for neuron in &mut self.neurons {
            let tx = tx.clone();
            let mut rx = tx.subscribe();
            let barrier = barrier.clone();

            let handle = task::spawn(async move {
                let mut t = start;
                let mut last_check = start;
                while t < end {
                    if t - last_check > 1.0 {
                        loop {
                            match rx.try_recv() {
                                Ok(msg) => {
                                    println!("Neuron {} received new message: {:?}", neuron.id(), msg);
                                    neuron.receive_and_process(msg);
                                }
                                Err(broadcast::error::TryRecvError::Empty) => break,
                                Err(broadcast::error::TryRecvError::Closed) => return,
                                Err(broadcast::error::TryRecvError::Lagged(_)) => {
                                    println!("Neuron {} lagged", neuron.id());
                                }
                            };
                        }
                        last_check = t;
                    }
                    println!("Neuron {} is doing computation", neuron.id());
                    if let Some(msg) = neuron.step(t) {
                        tx.send(msg).unwrap();
                    }
                    
                    t += dt;
                    barrier.wait().await;
                }
                neuron
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }





        // // self.init_mpsc_channels()?;

        // let barrier = Arc::new(Barrier::new(self.num_neurons()));
        // let mut handles = vec![];
        // let mut neurons = std::mem::take(&mut self.neurons);

        // let dt = 0.001;
        // for mut neuron in neurons.into_iter() {
        //     let barrier = barrier.clone();
        //     let handle = thread::spawn(move || {
        //         let mut time = start;
        //         while time < end {
        //             neuron.process_and_send(time, dt);
        //             barrier.wait();
        //             neuron.receive_and_process();
        //             barrier.wait();
        //             time += dt;
        //         }
        //         neuron
        //     });

        //     handles.push(handle);
        // }

        // self.neurons = handles
        //     .into_iter()
        //     .map(|handle| handle.join().unwrap())
        //     .collect();
        // // handle.join().unwrap();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    // #[test]
    // fn test_build_network_invalid_id() {
    //     let neurons = (0..3).map(|id| Neuron::new(id, 1.0, vec![])).collect();
    //     let connections = vec![Connection::build(0, 999, 1.0, 1.0).unwrap()];
    //     let result = Network::build(neurons, connections);
    //     assert_eq!(result, Err(NetworkError::InvalidNeuronId));

    //     let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
    //     let connections = vec![Connection::build(999, 0, 1.0, 1.0).unwrap()];
    //     let result = Network::build(neurons, connections);
    //     assert_eq!(result, Err(NetworkError::InvalidNeuronId));
    // }

    // #[test]
    // fn test_add_connection_invalid_id() {
    //     let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
    //     let mut network = Network::build(neurons, vec![]).unwrap();
    //     let connection = Connection::build(0, 999, 1.0, 1.0).unwrap();
    //     assert_eq!(
    //         network.add_connection(connection),
    //         Err(NetworkError::InvalidNeuronId)
    //     );
    // }


    #[test]
    fn test_num_neurons() {
        let neurons = vec![];
        let connections = vec![];
        let network = Network::new(neurons, connections);
        assert_eq!(network.num_neurons(), 0);        

        let neurons = (0..42).map(|id| Neuron::new(id, 1.0)).collect();
        let connections = vec![];
        let network = Network::new(neurons, connections);
        assert_eq!(network.num_neurons(), 42);

    }

    #[test]
    fn test_num_connections() {
        let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
        let connections = vec![];
        let network = Network::new(neurons, connections);
        assert_eq!(network.num_connections(), 0);

        let neurons = vec![Neuron::new(0, 1.0), Neuron::new(1, 1.0)];
        let connections = vec![Connection::new(0, 1, 1.0, 1.0); 42];
        let network = Network::new(neurons, connections);
        assert_eq!(network.num_connections(), 42);
    }

    #[test]
    fn test_save_load() {
        // Create a network
        let neurons = (0..3).map(|id| Neuron::new(id, 1.0)).collect();
        let connections = vec![
            Connection::new(0, 1, 1.0, 1.0),
            Connection::new(1, 2, -1.0, 2.0),
            Connection::new(0, 2, 0.25, 0.5),
        ];
        let network = Network::new(neurons, connections);

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();

        // Save network to the temporary file
        network.save_to(temp_file.path()).unwrap();

        // Load the network from the temporary file
        let loaded_network = Network::load_from(temp_file.path()).unwrap();

        assert_eq!(network, loaded_network);
    }

    #[test]
    fn test_num_inputs_outputs() {
        let neurons = vec![Neuron::new(0, 1.0), Neuron::new(1, 1.0), Neuron::new(2, 1.0)];
        let connections = vec![Connection::new(0, 1, 1.0, 1.0); 42];
        let network = Network::new(neurons, connections);

        assert_eq!(network.num_inputs(0), 0);
        assert_eq!(network.num_inputs(1), 42);
        assert_eq!(network.num_inputs(2), 0);

        assert_eq!(network.num_outputs(0), 42);
        assert_eq!(network.num_outputs(1), 0);
        assert_eq!(network.num_outputs(2), 0);
    }
}
