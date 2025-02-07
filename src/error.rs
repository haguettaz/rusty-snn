use std::error::Error;
use std::fmt;

use crate::core::REFRACTORY_PERIOD;

/// Error types for the spike train module.
#[derive(Debug, PartialEq)]
pub enum SNNError {
    /// Error for refractory period violation, e.g., two consecutive spikes are too close.
    RefractoryPeriodViolation {
        t1: f64,
        t2: f64,
    },
    /// Error for incompatible spike trains, e.g., different duration/period or number of channels.
    IncompatibleSpikeTrains,
    // Error while computing the number of spikes probabilities.
    InvalidNumSpikeWeights(String),
    /// Error for out of bounds access, e.g., neuron not found.
    OutOfBounds(String),
    /// Error for incompatible topology, e.g., the number of connections and neurons do not fit.
    IncompatibleTopology {
        num_neurons: usize,
        num_connections: usize,
    },
    /// Error for invalid operation.
    InvalidOperation(String),
    /// Error while memorizing the spike trains
    InfeasibleMemorization(String),
    /// External error from Gurobi
    OptimizationError(String),
    /// Convergence error from iterative algorithms
    ConvergenceError(String),
    /// Error for invalid parameters
    InvalidParameters(String),
    /// Error for error while computing the basis of the dominant eigenspace of the jitter linear propagation using Gram-Schmidt orthogonalization.
    GramSchmidtError(String),
    /// Error for invalid channel.
    InvalidChannel(String),
    /// Error for I/O operations.
    IOError(String),
}

impl fmt::Display for SNNError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SNNError::RefractoryPeriodViolation {t1, t2} => write!(f, "Violation of the refractory period: {} and {} are less than {} apart", t1, t2, REFRACTORY_PERIOD),
            SNNError::IncompatibleSpikeTrains => write!(f, "Incompatible spike trains"),
            SNNError::InvalidNumSpikeWeights(e) => write!(f, "Error while computing the number of spikes distribution: {}", e),
            SNNError::OutOfBounds(e) => {
                write!(f, "Index out of bounds: {}", e)
            }
            SNNError::IncompatibleTopology {num_neurons, num_connections} => write!(f, "Number of connections ({}) must be divisible by number of neurons ({}) for the selected topology", num_connections, num_neurons),
            SNNError::InfeasibleMemorization(e) => write!(f, "Infeasible memorization: {}", e),
            SNNError::OptimizationError(e) => write!(f, "Error with the Gurobi solver: {}", e),
            SNNError::ConvergenceError(e) => write!(f, "Convergence error: {}", e),
            SNNError::InvalidParameters(e) => write!(f, "Invalid parameters: {}", e),
            SNNError::GramSchmidtError(e) => write!(f, "Gram-Schmidt orthogonalization error: {}", e),
            SNNError::InvalidOperation(e) => write!(f, "Invalid operation: {}", e),
            SNNError::InvalidChannel(e) => write!(f, "Invalid channel: {}", e),
            SNNError::IOError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl Error for SNNError {}
