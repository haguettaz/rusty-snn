//! Error module for the Rusty SNN library.
use std::error::Error;
use std::fmt;

use crate::core::REFRACTORY_PERIOD;

/// Error types for the library.
#[derive(Debug, PartialEq)]
pub enum SNNError {
    /// Error for refractory period violation, e.g., two consecutive spikes are too close.
    RefractoryPeriodViolation {
        t1: f64,
        t2: f64,
    },
    /// Error for incompatible spike trains, e.g., different duration/period or number of channels.
    IncompatibleSpikeTrains(String),
    // Error while computing the number of spikes probabilities.
    InvalidNumSpikeWeights(String),
    /// Error for out of bounds access, e.g., neuron not found.
    OutOfBounds(String),
    /// Not implemented operation.
    NotImplemented(String),
    /// Error for invalid operation.
    InvalidOperation(String),
    /// Optimization error, e.g., failure while using the Gurobi solver.
    OptimizationError(String),
    /// Convergence error from iterative algorithms
    ConvergenceError(String),
    /// Error for invalid parameters
    InvalidParameter(String),
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
            SNNError::IncompatibleSpikeTrains(e) => write!(f, "Incompatible spike trains: {}", e),
            SNNError::InvalidNumSpikeWeights(e) => write!(f, "Error while computing the number of spikes distribution: {}", e),
            SNNError::OutOfBounds(e) => {
                write!(f, "Index out of bounds: {}", e)
            }
            SNNError::OptimizationError(e) => write!(f, "Optimization error: {}", e),
            SNNError::NotImplemented(e) => write!(f, "Not implemented: {}", e),
            SNNError::ConvergenceError(e) => write!(f, "Convergence error: {}", e),
            SNNError::InvalidParameter(e) => write!(f, "Invalid parameters: {}", e),
            SNNError::GramSchmidtError(e) => write!(f, "Gram-Schmidt orthogonalization error: {}", e),
            SNNError::InvalidOperation(e) => write!(f, "Invalid operation: {}", e),
            SNNError::InvalidChannel(e) => write!(f, "Invalid channel: {}", e),
            SNNError::IOError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl Error for SNNError {}
