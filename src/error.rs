use std::fmt;
use std::error::Error;

use super::REFRACTORY_PERIOD;


// SpikeTrainError: period, firing_rate, refractory period
// NetworkError: num_neurons, num_connections, topology, weight, delay
// SimulationError: ...
// MemorizationError: GurobiAPI, infeasible 

/// Error types for the spike train module.
#[derive(Debug, PartialEq)]
pub enum SNNError {
    /// Error for invalid firing times, e.g., NaN or infinite values.
    // InvalidFiring/Times,
    /// Error for refractory period violation, e.g., two consecutive spikes are too close.
    RefractoryPeriodViolation {t1: f64, t2: f64},
    /// Error for incompatible spike trains, e.g., different duration/period or number of channels.
    IncompatibleSpikeTrains,
    // Error for invalid spike train period, e.g., negative values.
    // InvalidPeriod,
    // Error for invalid firing rate, e.g., non-positive values.
    // InvalidFiringRate,
    // Error while computing the number of spikes probabilities.
    InvalidNumSpikeWeights(String),
    /// Error for neuron not found in the network.
    NeuronNotFound,
    /// Error for incompatible topology, e.g., the number of connections and neurons do not fit.
    IncompatibleTopology {num_neurons: usize, num_connections: usize},
    /// Error for invalid delay values, e.g., negative values or NaN.
    // InvalidDelay,
    /// Error for invalid weight values, e.g., NaN or infinite values.
    // InvalidWeight,
    /// Error for invalid interval 
    // InvalidInterval,
    /// Error while memorizing the spike trains
    InfeasibleMemorization,
    /// External error from Gurobi
    OptimizationError(String),
    /// Convergence error from iterative algorithms
    ConvergenceError(String),
    /// Error for invalid parameters
    InvalidParameters(String),
    /// Error for error while computing the basis of the dominant eigenspace of the jitter linear propagation using Gram-Schmidt orthogonalization.
    JitterGramSchmidtError(String),
    /// Error for normalization error
    JitterNormalizationError(String),
}

impl fmt::Display for SNNError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SNNError::RefractoryPeriodViolation {t1, t2} => write!(f, "Violation of the refractory period: {} and {} are less than {} apart", t1, t2, REFRACTORY_PERIOD),
            SNNError::IncompatibleSpikeTrains => write!(f, "Incompatible spike trains"),
            SNNError::InvalidNumSpikeWeights(e) => write!(f, "Error while computing the number of spikes distribution: {}", e),
            SNNError::NeuronNotFound => {
                write!(f, "Neuron not found in the network")
            }
            SNNError::IncompatibleTopology {num_neurons, num_connections} => write!(f, "Number of connections ({}) must be divisible by number of neurons ({}) for the selected topology", num_connections, num_neurons),
            SNNError::InfeasibleMemorization => write!(f, "Error while memorizing the spike trains"),
            SNNError::OptimizationError(e) => write!(f, "Error with the Gurobi solver: {}", e),
            SNNError::ConvergenceError(e) => write!(f, "Convergence error: {}", e),
            SNNError::InvalidParameters(e) => write!(f, "Invalid parameters: {}", e),
            SNNError::JitterGramSchmidtError(e) => write!(f, "Gram-Schmidt orthogonalization error: {}", e),
            SNNError::JitterNormalizationError(e) => write!(f, "Normalization error: {}", e),
        }
    }
}

impl Error for SNNError {}