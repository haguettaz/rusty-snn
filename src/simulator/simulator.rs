//! This module contains the simulation program and interval structures.
//!
use crate::core::spike_train::SpikeTrain;

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

/// Represents a time interval with a start and end time.
#[derive(Debug, PartialEq)]
pub struct TimeInterval {
    start: f64,
    end: f64,
}

impl PartialOrd for TimeInterval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.end <= other.start {
            Some(Ordering::Less)
        } else if self.start >= other.end {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}

impl TimeInterval {
    /// Create a time interval with the specified parameters.
    pub fn build(start: f64, end: f64) -> Result<Self, SimulationError> {
        if start >= end {
            return Err(SimulationError::EmptyTimeInterval);
        }

        if !(start.is_finite() && end.is_finite()) {
            return Err(SimulationError::HalfInfiniteTimeInterval);
        }

        Ok(TimeInterval { start, end })
    }

    /// Returns the start time of the time interval.
    pub fn start(&self) -> f64 {
        self.start
    }

    /// Returns the end time of the time interval.
    pub fn end(&self) -> f64 {
        self.end
    }
}

/// Represents a simulation interval with neuron control and disturbance.
#[derive(Debug, PartialEq)]
pub struct SimulationProgram {
    interval: TimeInterval,
    neuron_control: Vec<SpikeTrain>,
    threshold_noise: f64,
}

impl SimulationProgram {
    /// Create a simulation interval with the specified parameters.
    /// The function returns an error for invalid simulation intervals or control.
    pub fn build(
        start: f64,
        end: f64,
        threshold_noise: f64,
        neuron_control: Vec<SpikeTrain>,
    ) -> Result<Self, SimulationError> {
        let interval = TimeInterval::build(start, end)?;

        if threshold_noise < 0.0 {
            return Err(SimulationError::InvalidThresholdNoise);
        }

        let mut ids = neuron_control
            .iter()
            .map(|spike_train| spike_train.id())
            .collect::<Vec<_>>();
        ids.sort();
        if ids.windows(2).any(|w: &[usize]| w[0] == w[1]) {
            return Err(SimulationError::InvalidControl);
        }

        Ok(SimulationProgram {
            interval,
            neuron_control,
            threshold_noise,
        })
    }

    /// Returns the time interval of the simulation program.
    pub fn interval(&self) -> &TimeInterval {
        &self.interval
    }

    /// Returns the start time of the simulation interval.
    pub fn start(&self) -> f64 {
        self.interval.start()
    }

    /// Returns the end time of the simulation interval.
    pub fn end(&self) -> f64 {
        self.interval.end()
    }

    /// Returns the threshold noise (standard deviation) of the simulation interval.
    pub fn threshold_noise(&self) -> f64 {
        self.threshold_noise
    }

    /// Returns the control firing times for the specified neuron.
    pub fn neuron_control(&self, id: usize) -> Option<&[f64]> {
        let spike_train = self
            .neuron_control
            .iter()
            .find(|spike_train| spike_train.id() == id);
        match spike_train {
            Some(spike_train) => Some(spike_train.firing_times()),
            None => None,
        }
    }
}

/// Error types for the simulation program.
#[derive(Debug, PartialEq)]
pub enum SimulationError {
    /// Error for empty time interval.
    EmptyTimeInterval,
    /// Error for half-infinite time interval.
    HalfInfiniteTimeInterval,
    /// Error for invalid threshold noise.
    InvalidThresholdNoise,
    /// Error for invalid control.
    InvalidControl,
    /// Error for failed simulation.
    SimulationFailed,
}

impl fmt::Display for SimulationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SimulationError::EmptyTimeInterval => write!(f, "Start must be less than end"),
            SimulationError::HalfInfiniteTimeInterval => {
                write!(f, "Start and end must be finite")
            }
            SimulationError::InvalidThresholdNoise => write!(
                f,
                "Invalid threshold noise: standard deviation must be non-negative"
            ),
            SimulationError::InvalidControl => write!(f, "Invalid control"),
            SimulationError::SimulationFailed => write!(f, "Simulation failed"),
        }
    }
}

impl Error for SimulationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_ordering_time_interval() {
        let interval1 = TimeInterval::build(0.0, 1.0).unwrap();
        let interval2 = TimeInterval::build(1.0, 2.0).unwrap();
        let interval3 = TimeInterval::build(1.0, 3.0).unwrap();
        assert_eq!(interval1.partial_cmp(&interval2), Some(Ordering::Less));
        assert_eq!(interval2.partial_cmp(&interval1), Some(Ordering::Greater));
        assert_eq!(interval1.partial_cmp(&interval3), Some(Ordering::Less));
        assert_eq!(interval3.partial_cmp(&interval1), Some(Ordering::Greater));
        assert_eq!(interval2.partial_cmp(&interval3), None);
        assert_eq!(interval3.partial_cmp(&interval2), None);
    }

    #[test]
    fn test_invalid_simulation_interval() {
        assert_eq!(
            SimulationProgram::build(1.0, 0.0, 0.0, vec![]),
            Err(SimulationError::EmptyTimeInterval)
        );
        assert_eq!(
            SimulationProgram::build(0.0, 0.0, 0.0, vec![]),
            Err(SimulationError::EmptyTimeInterval)
        );
        assert_eq!(
            SimulationProgram::build(0.0, std::f64::INFINITY, 0.0, vec![]),
            Err(SimulationError::HalfInfiniteTimeInterval)
        );
    }

    #[test]
    fn test_invalid_control() {
        let spike_train = SpikeTrain::build(1, &[0.0, 1.5]).unwrap();
        let spike_train_duplicate = SpikeTrain::build(1, &[0.1, 1.3]).unwrap();
        assert_eq!(
            SimulationProgram::build(0.0, 3.0, 0.0, vec![spike_train, spike_train_duplicate]),
            Err(SimulationError::InvalidControl)
        );
    }

    #[test]
    fn test_invalid_noise() {
        assert_eq!(
            SimulationProgram::build(0.0, 1.0, -1.0, vec![]),
            Err(SimulationError::InvalidThresholdNoise)
        );
    }
}
