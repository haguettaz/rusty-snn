//! Module implementing the concept of a spike train.

use crate::core::REFRACTORY_PERIOD;
use std::fmt;
use std::error::Error;

/// Error types for the spike train module.
#[derive(Debug, PartialEq)]
pub enum SpikeTrainError {
    /// Error for invalid firing times, e.g., NaN or infinite values.
    InvalidTimes,
    /// Error for refractory period violation, e.g., two consecutive spikes are too close.
    RefractoryPeriodViolation {t1: f64, t2: f64},
    /// Error for incompatible spike trains, e.g., different duration/period or number of channels.
    IncompatibleSpikeTrains,
}

impl fmt::Display for SpikeTrainError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SpikeTrainError::InvalidTimes => write!(f, "Invalid firing times detected"),
            SpikeTrainError::RefractoryPeriodViolation {t1, t2} => write!(f, "Violation of the refractory period: {} and {} are less than {} apart", t1, t2, REFRACTORY_PERIOD),
            SpikeTrainError::IncompatibleSpikeTrains => write!(f, "Incompatible spike trains"),
        }
    }
}

impl Error for SpikeTrainError {}

/// Represents a spike train associated with a specific neuron.
#[derive(Debug, PartialEq, Clone)]
pub struct SpikeTrain {
    id: usize,
    firing_times: Vec<f64>,
}

impl SpikeTrain {
    /// Create a spike train with the specified parameters.
    /// If necessary, the firing times are sorted.
    /// The function returns an error for invalid firing times.
    pub fn build(id: usize, firing_times: &[f64]) -> Result<Self, SpikeTrainError> {
        for t in firing_times {
            if !t.is_finite() {
                return Err(SpikeTrainError::InvalidTimes);
            }
        }

        let mut firing_times = firing_times.to_vec();
        firing_times.sort_by(|t1, t2| {
            t1.partial_cmp(t2).unwrap_or_else(|| panic!("Comparison failed: NaN values should have been caught earlier"))
        });

        for ts in firing_times.windows(2) {
            if ts[1] - ts[0] <= REFRACTORY_PERIOD {
                return Err(SpikeTrainError::RefractoryPeriodViolation {t1: ts[0], t2: ts[1]});
            }
        }    

        Ok(SpikeTrain { id, firing_times: firing_times})
    }

    /// Returns the ID of the neuron associated with the spike train.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the firing times of the spike train.
    pub fn firing_times(&self) -> &[f64] {
        &self.firing_times[..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_train_build() {
        // Test valid spike trains with unsorted firing times
        let spike_train = SpikeTrain::build(0, &[0.0, 2.0, 5.0]).unwrap();
        assert_eq!(spike_train.firing_times(), &[0.0, 2.0, 5.0]);

        // Test valid spike trains with unsorted firing times
        let spike_train = SpikeTrain::build(0, &[0.0, 5.0, 2.0]).unwrap();
        assert_eq!(spike_train.firing_times(), &[0.0, 2.0, 5.0]);
        
        // Test empty spike train
        let spike_train = SpikeTrain::build(0, &[]).unwrap();
        assert_eq!(spike_train.firing_times(), &[] as &[f64]);

        // Test invalid spike train (NaN values)
        let spike_train = SpikeTrain::build(0, &[0.0, 5.0, f64::NAN]);
        assert_eq!(spike_train, Err(SpikeTrainError::InvalidTimes));

        // Test invalid spike train (refractory period violation)
        let spike_train = SpikeTrain::build(0, &[0.0, 1.0]);
        assert_eq!(spike_train, Err(SpikeTrainError::RefractoryPeriodViolation {t1: 0.0, t2: 1.0}));

        // Test invalid spike train (strict refractory period violation)
        let spike_train = SpikeTrain::build(0, &[0.0, 5.0, 3.0, 4.5]);
        assert_eq!(spike_train, Err(SpikeTrainError::RefractoryPeriodViolation {t1: 4.5, t2: 5.0}));
    }
}