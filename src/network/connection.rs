//! Connection module with utilities for creating and managing inputs to neurons.

use serde::{Deserialize, Serialize};

use super::error::NetworkError;

/// Represents an input to a neuron.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Connection {
    /// ID of the sending neuron
    source_id: usize,
    /// ID of the receiving neuron
    target_id: usize,
    /// Weight of the input
    weight: f64,
    /// Delay of the input
    delay: f64,
    /// Times at which the sending neuron fired
    firing_times: Vec<f64>,
}

impl Connection {
    /// Create a new input with the specified parameters.
    pub fn build(source_id: usize, target_id: usize, weight: f64, delay: f64) -> Result<Self, NetworkError> {
        if delay < 0.0 {
            return Err(NetworkError::InvalidDelay);
        }
        Ok(Connection {
            source_id,
            target_id,
            weight,
            delay,
            firing_times: Vec::new(),
        })
    }

    /// Add a firing time to the input train.
    pub fn add_firing_time(&mut self, t: f64) {
        self.firing_times.push(t);
    }

    /// Evaluate the connection signal at the output at a given time.
    pub fn output(&self, t: f64) -> f64 {
        self.firing_times
            .iter()
            .map(|ft| t - ft - self.delay)
            .filter_map(|dt| {
                if dt > 0. {
                    Some(2_f64 * dt * (-dt).exp())
                } else {
                    None
                }
            })
            .sum()
    }

    /// Get the weight of the connection.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Get the delay of the connection.
    pub fn delay(&self) -> f64 {
        self.delay
    }

    /// Get the id of the sending neuron.
    pub fn source_id(&self) -> usize {
        self.source_id
    }

    /// Get the id of the receiving neuron.
    pub fn target_id(&self) -> usize {
        self.target_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    #[test]
    fn test_invalid_delay() {
        let result = Connection::build(0, 1, 1.0, -1.0);
        assert_eq!(result, Err(NetworkError::InvalidDelay));
    }

    #[test]
    fn test_output() {
        let mut connection = Connection::build(0, 1, 1.0, 1.0).unwrap();
        connection.add_firing_time(0.0);
        assert_eq!(connection.output(0.0), 0.0);
        assert_eq!(connection.output(1.0), 0.0);
        assert_eq!(connection.output(2.0), 2.0 / E);
    }
}
