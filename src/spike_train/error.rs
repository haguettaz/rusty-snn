//! Error types for the spike train module.

#[derive(Debug, PartialEq)]
pub enum SpikeTrainError {
    /// Error for refractory period violation, e.g., two consecutive spikes are too close.
    RefractoryPeriodViolation,
}

impl std::fmt::Display for SpikeTrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SpikeTrainError::RefractoryPeriodViolation => write!(f, "Violation of the refractory period"),
        }
    }
}