//! Channel module with utilities for creating and managing communication channels between neurons.

use serde::{Deserialize, Serialize};

/// Trait representing a communication channel (axon + synapse + dentrite) between two neurons.
pub trait Channel {
    /// Evaluate the signal sent by the source neuron at a given time.
    fn eval_source(&self, t: f64) -> f64;

    /// Evaluate the signal received by the target neuron at a given time.
    fn eval_target(&self, t: f64) -> f64;

    /// Get the source id of the channel.
    fn source_id(&self) -> usize;

    /// Get the target id of the channel.
    fn target_id(&self) -> usize;
}

/// implement the channel trait for a input struct 
impl Channel for Input {
    fn eval_source(&self, t: f64) -> f64 {
        self.eval(t)
    }

    fn eval_target(&self, t: f64) -> f64 {
        self.eval(t) * self.weight
    }

    fn source_id(&self) -> usize {
        self.source_id()
    }

    fn target_id(&self) -> usize {
        // Assuming target_id is the same as source_id for Input
        self.source_id()
    }
}

/// Represents an input to a neuron.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Input {
    /// Unique identifier for the source of the input
    source_id: usize,
    /// Weight of the input
    weight: f64,
    /// Delay of the input
    delay: f64,
    /// Times at which the input fired
    firing_times: Vec<f64>,
}

impl Input {
    /// Create a new input with the specified parameters.
    /// 
    /// # Arguments
    /// 
    /// * `source_id` - Unique identifier for the source of the input
    /// * `weight` - Weight of the input
    /// * `delay` - Delay of the input
    pub fn new(source_id: usize, weight: f64, delay: f64) -> Self {
        Input {
            source_id,
            weight,
            delay,
            firing_times: vec![],
        }
    }

    /// Add a firing time to the input.
    /// 
    /// # Arguments
    /// 
    /// * `firing_time` - Time at which the input fired
    pub fn add_firing_time(&mut self, firing_time: f64) {
        self.firing_times.push(firing_time);
    }

    /// Evaluate the input at a given time.
    /// 
    /// # Arguments
    /// 
    /// * `t` - Time at which to evaluate the input
    pub fn eval(&self, t: f64) -> f64 {
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

    /// Borrow the firing times of the input.
    pub fn firing_times(&self) -> &[f64] {
        &self.firing_times
    }

    /// Get the weight of the input.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Get the delay of the input.
    pub fn delay(&self) -> f64 {
        self.delay
    }

    /// Get the id of the input neuron.
    pub fn source_id(&self) -> usize {
        self.source_id
    }
}

impl Wire for Input {
    fn eval(&self, t: f64) -> f64 {
        self.eval(t)
    }

    fn source_id(&self) -> usize {
        self.source_id()
    }

    fn target_id(&self) -> usize {
        // Assuming target_id is the same as source_id for Input
        self.source_id()
    }
}

impl Channel for Input {
    fn eval_source(&self, t: f64) -> f64 {
        self.eval(t)
    }

    fn eval_target(&self, t: f64) -> f64 {
        self.eval(t) * self.weight
    }

    fn source_id(&self) -> usize {
        self.source_id()
    }

    fn target_id(&self) -> usize {
        // Assuming target_id is the same as source_id for Input
        self.source_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    #[test]
    fn test_input_apply() {
        let mut input = Input::new(0, 1.0, 1.0);
        input.add_firing_time(0.0);
        input.add_firing_time(1.0);
        assert!((input.eval(0.0)).abs() < 1e-10);
        assert!((input.eval(1.0)).abs() < 1e-10);
        assert!((input.eval(2.0) - 2.0 / E).abs() < 1e-10);
    }
}
