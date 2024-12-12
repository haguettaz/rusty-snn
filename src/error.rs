use std::fmt;
use std::error::Error;

use super::neuron::REFRACTORY_PERIOD;

/// Error types for the spike train module.
#[derive(Debug, PartialEq)]
pub enum SNNError {
    /// Error for invalid firing times, e.g., NaN or infinite values.
    InvalidFiringTimes,
    /// Error for refractory period violation, e.g., two consecutive spikes are too close.
    RefractoryPeriodViolation {t1: f64, t2: f64},
    /// Error for incompatible spike trains, e.g., different duration/period or number of channels.
    IncompatibleSpikeTrains,
    /// Error for neuron not found in the network.
    NeuronNotFound,
    /// Error for incompatible topology, e.g., the number of connections and neurons do not fit.
    IncompatibleTopology {num_neurons: usize, num_connections: usize},
    /// Error for invalid delay values, e.g., negative values.
    InvalidDelay
}

impl fmt::Display for SNNError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SNNError::InvalidFiringTimes => write!(f, "Invalid firing times detected"),
            SNNError::RefractoryPeriodViolation {t1, t2} => write!(f, "Violation of the refractory period: {} and {} are less than {} apart", t1, t2, REFRACTORY_PERIOD),
            SNNError::IncompatibleSpikeTrains => write!(f, "Incompatible spike trains"),
            SNNError::NeuronNotFound => {
                write!(f, "Neuron not found in the network")
            }
            SNNError::InvalidDelay => write!(f, "Invalid delay value: must be non-negative"),
            SNNError::IncompatibleTopology {num_neurons, num_connections} => write!(f, "Number of connections ({}) must be divisible by number of neurons ({}) for the selected topology", num_connections, num_neurons),
        }
    }
}

impl Error for SNNError {}