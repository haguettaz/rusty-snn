//! This crate provides tools for simulating spiking neural networks (SNNs) in Rust.
//!
//! # Creating Networks
//!
//! ## At Random
//!
//! ## From Data
//!
//! ## From Scratch
//!
//! ### Examples
//!
//! ```rust
//! use rusty_snn::network::Network;
//!
//! // Init an empty network and add a few neurons and connections
//! let mut network = Network::new();
//! network.add_connection(0, 1, 1.0, 0.25).unwrap();
//! network.add_connection(0, 1, -0.5, 0.5).unwrap();
//! network.add_connection(2, 3, -0.75, 1.0).unwrap();
//! network.add_connection(3, 3, 0.25, 0.125).unwrap();
//! network.add_connection(1, 2, 0.25, 2.0).unwrap();
//!
//! assert_eq!(network.num_neurons(), 4);
//! assert_eq!(network.num_connections(), 5);
//! ```
//!
//! # Simulating Networks
//!
//!
//! # Optimizing Networks
//!
//!

// pub mod connection;
pub mod error;
pub mod network;
pub mod neuron;
pub mod optimizer;
pub mod simulator;
pub mod spike_train;
pub mod utils;
