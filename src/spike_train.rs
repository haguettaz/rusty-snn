//! Module implementing the concept of a spike train.

use super::neuron::REFRACTORY_PERIOD;
use super::error::SNNError;

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
    pub fn build(id: usize, firing_times: &[f64]) -> Result<Self, SNNError> {
        for t in firing_times {
            if !t.is_finite() {
                return Err(SNNError::InvalidFiringTimes);
            }
        }

        let mut firing_times = firing_times.to_vec();
        firing_times.sort_by(|t1, t2| {
            t1.partial_cmp(t2).unwrap_or_else(|| panic!("Comparison failed: NaN values should have been caught earlier"))
        });

        for ts in firing_times.windows(2) {
            if ts[1] - ts[0] <= REFRACTORY_PERIOD {
                return Err(SNNError::RefractoryPeriodViolation {t1: ts[0], t2: ts[1]});
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
        assert_eq!(spike_train, Err(SNNError::InvalidFiringTimes));

        // Test invalid spike train (refractory period violation)
        let spike_train = SpikeTrain::build(0, &[0.0, 1.0]);
        assert_eq!(spike_train, Err(SNNError::RefractoryPeriodViolation {t1: 0.0, t2: 1.0}));

        // Test invalid spike train (strict refractory period violation)
        let spike_train = SpikeTrain::build(0, &[0.0, 5.0, 3.0, 4.5]);
        assert_eq!(spike_train, Err(SNNError::RefractoryPeriodViolation {t1: 4.5, t2: 5.0}));
    }
}