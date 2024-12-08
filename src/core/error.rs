use std::fmt;
use std::error::Error;

use crate::core::REFRACTORY_PERIOD;

/// Error types for the spike train module.
#[derive(Debug, PartialEq)]
pub enum CoreError {
    /// Error for invalid firing times, e.g., NaN or infinite values.
    InvalidFiringTimes,
    /// Error for refractory period violation, e.g., two consecutive spikes are too close.
    RefractoryPeriodViolation {t1: f64, t2: f64},
    /// Error for incompatible spike trains, e.g., different duration/period or number of channels.
    IncompatibleSpikeTrains,
    /// Error for neuron not found in the network.
    NeuronNotFound,
    /// Error for incompatible topology, e.g., the number of connections and neurons do not fit.
    IncompatibleTopology,
    /// Error for invalid delay values, e.g., negative values.
    InvalidDelay
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CoreError::InvalidFiringTimes => write!(f, "Invalid firing times detected"),
            CoreError::RefractoryPeriodViolation {t1, t2} => write!(f, "Violation of the refractory period: {} and {} are less than {} apart", t1, t2, REFRACTORY_PERIOD),
            CoreError::IncompatibleSpikeTrains => write!(f, "Incompatible spike trains"),
            CoreError::NeuronNotFound => {
                write!(f, "Neuron not found in the network")
            }
            CoreError::InvalidDelay => write!(f, "Invalid delay value: must be non-negative"),
            CoreError::IncompatibleTopology => write!(f, "The connectivity topology is not compatible with the number of connections and neurons"),
        }
    }
}

impl Error for CoreError {}