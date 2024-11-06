//! Error types for the network module.

// #[derive(Debug, PartialEq)]
// pub enum NetworkError {
//     /// Error for invalid neuron id.
//     InvalidNeuronId,
//     /// Error for invalid delay value.
//     InvalidDelay,
//     /// Error for incompatibility of the topology with the number of connections and neurons.
//     IncompatibleTopology,
// }

// impl std::fmt::Display for NetworkError {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         match self {
//             NetworkError::InvalidNeuronId => write!(f, "Invalid neuron id: out of bounds"),
//             NetworkError::InvalidDelay => write!(f, "Invalid delay value: must be non-negative"),
//         }
//     }
// }

#[derive(Debug, PartialEq)]
pub enum NetworkError {
    /// Error for invalid neuron id.
    InvalidNeuronId,
    /// Error for invalid source neuron id.
    InvalidSourceId,
    /// Error for invalid target neuron id.
    InvalidTargetId,
    /// Error for invalid delay value.
    InvalidDelay,
    /// Error for incompatibility between the topology and the number of connections and neurons.
    IncompatibleTopology,
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NetworkError::InvalidNeuronId => write!(f, "Invalid neuron id: out of bounds"),
            NetworkError::InvalidSourceId => write!(f, "Invalid source neuron id: out of bounds"),
            NetworkError::InvalidTargetId => write!(f, "Invalid target neuron id: out of bounds"),
            NetworkError::InvalidDelay => write!(f, "Invalid delay value: must be non-negative"),
            NetworkError::IncompatibleTopology => write!(f, "The connectivity topology is not compatible with the number of connections and neurons"),
        }
    }
}