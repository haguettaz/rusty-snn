//! Module implementing the concept of connections in a network.

use serde::{Deserialize, Serialize};

use super::error::SNNError;

/// Represents a connection between two neurons in a network.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Connection {
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
        // id: usize,
        source_id: usize,
        target_id: usize,
        weight: f64,
        delay: f64,
    ) -> Result<Self, SNNError> {
        if delay < 0.0 {
            return Err(SNNError::InvalidParameters(
                "Connection delay must be non-negative".to_string(),
            ));
        }

        Ok(Connection {
            source_id,
            target_id,
            weight,
            delay,
        })
    }

    // /// Returns the ID of the connection.
    // /// The ID is unique within the network.
    // pub fn id(&self) -> usize {
    //     self.id
    // }

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
    pub fn set_weight(&mut self, weight: f64) {
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
