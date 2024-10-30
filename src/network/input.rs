//! Input module with utilities for creating and managing inputs to neurons.

use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq)]
pub enum InputError {
    DelaysError(String),
    // WeightsError(String),
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
    pub fn new(source_id: usize, weight: f64, delay: f64) -> Self {
        Input {
            source_id,
            weight,
            delay,
            firing_times: Vec::new(),
        }
    }

    pub fn build(
        source_id: usize,
        weight: f64,
        delay: f64,
        // order: i32,
        // beta: f64,
    ) -> Result<Self, InputError> {
        if delay < 0.0 {
            return Err(InputError::DelaysError("Delays must be non-negative.".into()));
        }

        Ok(Input {
            source_id,
            weight,
            delay,
            firing_times: Vec::new(),
        })
    }

    pub fn add_firing_time(&mut self, ft: f64) {
        self.firing_times.push(ft);
    }

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

    pub fn firing_times(&self) -> &Vec<f64> {
        &self.firing_times
    }

    pub fn weight(&self) -> f64 {
        self.weight
    }

    pub fn delay(&self) -> f64 {
        self.delay
    }

    // pub fn kernel(&self) -> &Kernel {
    //     &self.kernel
    // }

    pub fn source_id(&self) -> usize {
        self.source_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    #[test]
    fn test_input() {
        assert_eq!(Input::build(42, 1.0, -1.0), Err(InputError::DelaysError("Delays must be non-negative.".into())));
    }

    #[test]
    fn test_input_apply() {
        let mut input = Input::build(0, 1.0, 1.0).unwrap();
        input.add_firing_time(0.0);
        input.add_firing_time(1.0);
        assert!((input.eval(0.0)).abs() < 1e-10);
        assert!((input.eval(1.0)).abs() < 1e-10);
        assert!((input.eval(2.0) - 2.0 / E).abs() < 1e-10);
    }

    // #[test]
    // fn test_input_clone() {
    //     let input = Input::build(0, 1.0, 1.0).unwrap();
    //     let cloned = input.clone();
    //     assert_eq!(input, cloned);
    // }
}
