//! Core module defining the main abstract components of the Rusty SNN library.
pub mod neuron;
pub mod optim;
pub mod spike;
pub mod utils;
pub mod metrics;
pub mod network;

/// The minimum time period between spikes.
pub const REFRACTORY_PERIOD: f64 = 1.0;
/// The nominal threshold for a neuron to fire.
pub const FIRING_THRESHOLD: f64 = 1.0;
