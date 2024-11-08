//! Core module defining the main components of the library.
pub mod spike_train;
pub mod network;
pub mod neuron;
pub mod connection;

pub const REFRACTORY_PERIOD: f64 = 1.0;
pub const FIRING_THRESHOLD: f64 = 1.0;