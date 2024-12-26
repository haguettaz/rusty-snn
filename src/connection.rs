//! Module implementing the concept of connections in a network.

use serde::{Deserialize, Serialize};

use super::error::SNNError;

/// Represents an input to a neuron.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Input {
    /// The unique input ID
    id: usize,
    /// The source neuron ID
    source_id: usize,
    /// The input weight
    weight: f64,
    /// The input_delay
    delay: f64,
}

impl Input {
    /// Create a new input with the specified parameters.
    pub fn new(id: usize, source_id: usize, weight: f64, delay: f64) -> Self {
        Input {
            id,
            source_id,
            weight,
            delay,
        }
    }

    /// Returns the weight of the connection.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Returns the delay of the connection.
    pub fn delay(&self) -> f64 {
        self.delay
    }

    /// Returns the ID of the source neuron.
    pub fn source_id(&self) -> usize {
        self.source_id
    }

    /// Returns the ID of the input.
    /// The ID is unique within the neuron.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Update the weight of the connection.
    pub fn update_weight(&mut self, weight: f64) {
        self.weight = weight;
    }
}

/// Represents a connection between two neurons in a network.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Connection {
    /// Connection ID
    id: usize,
    /// Source ID
    source_id: usize,
    /// Target ID
    target_id: usize,
    /// Connection weight
    weight: f64,
    /// Connection delay (must be non-negative)
    delay: f64,
}

impl Connection {
    /// Create a new connection with the specified parameters.
    /// Returns an error if the delay is negative.
    pub fn build(
        id: usize,
        source_id: usize,
        target_id: usize,
        weight: f64,
        delay: f64,
    ) -> Result<Self, SNNError> {
        if delay < 0.0 {
            return Err(SNNError::InvalidDelay);
        }

        Ok(Connection {
            id,
            source_id,
            target_id,
            weight,
            delay,
        })
    }

    /// Returns the ID of the connection.
    /// The ID is unique within the network.
    pub fn id(&self) -> usize {
        self.id
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

    /// Set the weight of the connection.
    pub fn update_weight(&mut self, weight: f64) {
        self.weight = weight;
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_connection_build() {
//         let connection = Connection::build(0.5, 1.0).unwrap();
//         assert_eq!(connection.weight, 0.5);
//         assert_eq!(connection.delay, 1.0);
//     }

//     #[test]
//     fn test_connection_build_invalid_delay() {
//         let connection = Connection::build(0.5, -1.0);
//         assert_eq!(connection, Err(SNNError::InvalidDelay));
//     }
// }
