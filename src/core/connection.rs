//! Module implementing the concept of connections in a network.
//!
//! A connection represents a synaptic link between two neurons, characterized by:
//! - source_id: The ID of the presynaptic neuron
//! - target_id: The ID of the postsynaptic neuron
//! - weight: The connection strength
//! - delay: The connection delay (must be non-negative)
//!
//! # Example
//! ```
//! use rusty_snn::core::connection::Connection;
//! 
//! let connection = Connection::build(0, 1, 0.5, 1.0).unwrap();
//! ```

use std::fmt;
use std::error::Error;
use serde::{Deserialize, Serialize};

/// Represents a connection between two neurons in a network.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Connection {
    /// ID of the source (presynaptic) neuron
    source_id: usize,
    /// ID of the target (postsynaptic) neuron
    target_id: usize,
    /// Connection weight
    weight: f64,
    /// Connection delay (must be non-negative)
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

impl fmt::Display for ConnectionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConnectionError::InvalidDelay => write!(f, "Invalid delay value: must be non-negative"),
            ConnectionError::InvalidTargetID => write!(f, "Invalid target ID"),
            ConnectionError::InvalidSourceID => write!(f, "Invalid source ID"),
        }
    }
}

impl Error for ConnectionError {}