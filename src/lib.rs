//! This crate provides tools for simulating spiking neural networks (SNNs) in Rust.
//!
//! # Creating Networks
//!
//! ## From Scratch
//!
//! ```rust
//! use rusty_snn::network::Network;
//!
//! // Init an empty network
//! let mut network = Network::new();
//!
//! // Add connections to the network
//! network.add_connection(0, 1, 1.0, 0.25).unwrap();
//! network.add_connection(0, 1, -0.5, 0.5).unwrap();
//! network.add_connection(2, 3, -0.75, 1.0).unwrap();
//! network.add_connection(3, 3, 0.25, 0.125).unwrap();
//! network.add_connection(1, 2, 0.25, 2.0).unwrap();
//!
//! // Check the number of neurons and connections
//! assert_eq!(network.num_neurons(), 4);
//! assert_eq!(network.num_connections(), 5);
//! ```
//!
//! ## From a File
//!
//! ## At Random
//!
//! ```rust
//! use rusty_snn::network::{Network, Topology};
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! // Create a random network with 277 neurons and 769 connections
//! let mut rng = StdRng::seed_from_u64(42);
//! let network = Network::rand(277, 769, (-0.1, 0.1), (0.1, 10.0), Topology::Random, &mut rng).unwrap();
//!
//! assert_eq!(network.num_neurons(), 277);
//! assert_eq!(network.num_connections(), 769);
//! ```
//!
//! # Simulating Networks
//!
//!
//! # Optimizing Networks
//!
//! ```rust
//! use rusty_snn::network::{Network, Topology};
//! use rusty_snn::spike_train::SpikeTrain;
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! // Create a random network
//! let mut rng = StdRng::seed_from_u64(42);
//! let mut network = Network::rand(200, 200 * 500, (-0.2, 0.2), (0.1, 10.0), Topology::Random, &mut rng).unwrap();
//! let spike_trains = SpikeTrain::rand(200, 50.0, 0.2, &mut rng).unwrap();
//!
//! // Optimize the network, i.e., find the optimal weights to reproduce the spike trains
//! // network.memorize_periodic_spike_trains(&spike_trains, 100.0, (-0.2, 0.2), 0.0, 0.2, 0.2).unwrap();
//! ```

pub mod connection;
pub mod error;
pub mod network;
pub mod neuron;
pub mod optim;
// pub mod simulator;
pub mod spike_train;
pub mod utils;

/// The tolerance for a potential value to be considered negligible (relative to the number of inputs).
pub const POTENTIAL_TOLERANCE: f64 = 1e-9;
/// The minimum time between spikes. Can be seen as the default unit of time of a neuron.
pub const REFRACTORY_PERIOD: f64 = 1.0;
/// The nominal threshold for a neuron to fire.
pub const FIRING_THRESHOLD: f64 = 1.0;
