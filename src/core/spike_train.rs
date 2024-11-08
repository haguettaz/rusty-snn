//! Module implementing the concept of a spike train.

use crate::core::REFRACTORY_PERIOD;

/// Error types for the spike train module.
#[derive(Debug, PartialEq)]
pub enum SpikeTrainError {
    /// Error for refractory period violation, e.g., two consecutive spikes are too close.
    RefractoryPeriodViolation,
    /// Error for incompatible spike trains, e.g., different duration/period or number of channels.
    IncompatibleSpikeTrains,
}

impl std::fmt::Display for SpikeTrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SpikeTrainError::RefractoryPeriodViolation => write!(f, "Violation of the refractory period"),
            SpikeTrainError::IncompatibleSpikeTrains => write!(f, "Incompatible spike trains"),
        }
    }
}

/// Represents a spike train associated with a specific neuron.
#[derive(Debug, PartialEq)]
pub struct SpikeTrain {
    id: usize,
    firing_times: Vec<f64>,
}

impl SpikeTrain {
    /// Create a spike train with the specified parameters.
    /// If necessary, the firing times are sorted.
    /// The function returns an error for invalid firing times.
    pub fn build(id: usize, firing_times: &[f64]) -> Result<Self, SpikeTrainError> {    
        let mut firing_times = firing_times.to_vec();

        firing_times.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("A problem occured while sorting the provided firing times.")
        });

        if firing_times.windows(2).any(|w| w[0] >= w[1] + REFRACTORY_PERIOD) {
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