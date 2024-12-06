//! Core module defining the main components of the Rusty SNN library.
//!
//! This module provides the fundamental building blocks for creating and simulating
//! spiking neural networks. It consists of the following components:
//!
//! - [`spike_train`]: Handles the temporal sequence of neuron spikes
//! - [`network`]: Manages the structure and simulation of the neural network
//! - [`neuron`]: Implements individual neuron behavior and dynamics
//! - [`connection`]: Defines synaptic connections between neurons
//!
//! # Examples
//!
//! ```
//! use rusty_snn::core::{network::Network, neuron::Neuron, connection::Connection};
//!
//! 
//! // Create a list of connections between neurons
//! let connections = vec![Connection::build(0, 1, 1.0, 1.0).unwrap(), Connection::build(1, 2, -1.0, 2.0).unwrap()];
//! 
//! // Create a network with the specified connections
//! let mut network = Network::new(connections);
//! 
//! // Add a new neuron 42 to the network
//! network.add_neuron(42);
//! 
//! // Add a connection from neuron 0 to (a new) neuron 7
//! network.add_connection(0, 7, 1.0, 1.0).unwrap();
//! 
//! // Check the number of neurons in the network
//! assert_eq!(network.num_neurons(), 5);
//! 
//! // Check the number of connections in the network
//! assert_eq!(network.num_connections(), 3);
//! 
//! // Check the number of outputs of neuron 0
//! assert_eq!(network.num_outputs(0), 2);
//! ```
pub mod spike_train;
pub mod network;
pub mod neuron;
pub mod connection;
pub mod error;

/// The minimum time between spikes. Can be seen as the default unit of time of a neuron.
pub const REFRACTORY_PERIOD: f64 = 1.0;
/// The nominal threshold for a neuron to fire.
pub const FIRING_THRESHOLD: f64 = 1.0;
/// Minimum number of neurons to consider parallel processing.
pub const MIN_PARALLEL_NEURONS: usize = 100;