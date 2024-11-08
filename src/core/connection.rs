//! Module implementing the concept of connections in a network.

use serde::{Deserialize, Serialize};

/// Represents a connection between two neurons in a network.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Connection {
    source_id: usize,
    target_id: usize,
    weight: f64,
    delay: f64,
}

impl Connection {
    /// Create a new connection with the specified parameters.
    /// Returns an error if the delay is negative.
    pub fn build(source_id: usize, target_id: usize, weight: f64, delay: f64) -> Result<Self, ConnectionError> {
        if delay < 0.0 {
            return Err(ConnectionError::InvalidDelay);
        }

        Ok(Connection {
            source_id,
            target_id,
            weight,
            delay,
        })
    }

    /// Returns the ID of the source neuron.
    pub fn source_id(&self) -> usize {
        self.source_id
    }

    /// Returns the ID of the target neuron.
    pub fn target_id(&self) -> usize {
        self.target_id
    }

    /// Returns the weight of the connection.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Returns the delay of the connection.
    pub fn delay(&self) -> f64 {
        self.delay
    }
}

/// Error type related to network operations.
#[derive(Debug, PartialEq)]
pub enum ConnectionError {
    /// Error for invalid delay value.
    InvalidDelay,
    /// Error for invalid target ID
    InvalidTargetID,
    /// Error for invalid source ID
    InvalidSourceID,
}

impl std::fmt::Display for ConnectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ConnectionError::InvalidDelay => write!(f, "Invalid delay value: must be non-negative"),
            ConnectionError::InvalidTargetID => write!(f, "Invalid target ID"),
            ConnectionError::InvalidSourceID => write!(f, "Invalid source ID"),
        }
    }
}