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
    RefractoryPeriodViolation,
    /// Error for incompatible spike trains, e.g., different duration/period or number of channels.
    IncompatibleSpikeTrains,
}

impl fmt::Display for SpikeTrainError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SpikeTrainError::InvalidTimes => write!(f, "Invalid firing times"),
            SpikeTrainError::RefractoryPeriodViolation => write!(f, "Violation of the refractory period"),
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
        if firing_times.iter().any(|t| !t.is_finite()) {
            return Err(SpikeTrainError::InvalidTimes);
        }

        let mut firing_times = firing_times.to_vec();

        firing_times.sort_by(|t1, t2| {
            t1.partial_cmp(t2).unwrap()
        });

        if firing_times.windows(2).any(|w| w[1] - w[0] <= REFRACTORY_PERIOD) {
            return Err(SpikeTrainError::RefractoryPeriodViolation);
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
    use std::f64::NAN;

    use super::*;

    #[test]
    fn test_spike_train_build() {
        let spike_train = SpikeTrain::build(0, &[0.0, 5.0, 2.0]).unwrap();
        assert_eq!(spike_train.firing_times(), &[0.0, 2.0, 5.0]);

        let spike_train = SpikeTrain::build(0, &[0.0, 5.0, NAN]);
        assert_eq!(spike_train, Err(SpikeTrainError::InvalidTimes));

        let spike_train = SpikeTrain::build(0, &[0.0, 1.0]);
        assert_eq!(spike_train, Err(SpikeTrainError::RefractoryPeriodViolation));

        let spike_train = SpikeTrain::build(0, &[0.0, 5.0, 3.0, 4.5]);
        assert_eq!(spike_train, Err(SpikeTrainError::RefractoryPeriodViolation));
    }
}