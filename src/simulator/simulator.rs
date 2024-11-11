//! This module contains the simulation program and interval structures.
//! 
use crate::core::spike_train::SpikeTrain;

/// Represents a simulation interval with neuron control and disturbance.
#[derive(Debug, PartialEq)]
pub struct SimulationInterval {
    start: f64,
    end: f64,
    neuron_control: Vec<SpikeTrain>,
    threshold_noise: f64,
}

impl SimulationInterval {
    /// Create a simulation interval with the specified parameters.
    /// The function returns an error for invalid simulation intervals or control.
    pub fn build(start: f64, end: f64, threshold_noise:f64, neuron_control: Vec<SpikeTrain>) -> Result<Self, SimulationError> {
        if (start >= end) || !(start.is_finite() && end.is_finite()) {
            return Err(SimulationError::InvalidSimulationInterval);
        }

        if neuron_control.iter().flat_map(|spike_train| spike_train.firing_times().iter()).any(|&t| t < start || t >= end) {
            return Err(SimulationError::InvalidControl);
        }
        
        if threshold_noise < 0.0 {
            return Err(SimulationError::InvalidThresholdNoise);
        }

        let mut ids = neuron_control.iter().map(|spike_train| spike_train.id()).collect::<Vec<_>>();
        ids.sort();
        if ids.windows(2).any(|w| w[0] == w[1]) {
            return Err(SimulationError::InvalidControl);
        }

        Ok(SimulationInterval { start, end, neuron_control, threshold_noise })
    }

    /// Returns the start time of the simulation interval.
    pub fn start(&self) -> f64 {
        self.start
    }

    /// Returns the end time of the simulation interval.
    pub fn end(&self) -> f64 {
        self.end
    }

    /// Returns the threshold noise (standard deviation) of the simulation interval.
    pub fn threshold_noise(&self) -> f64 {
        self.threshold_noise
    }

    /// Returns the control times for the specified neuron.
    pub fn neuron_control(&self, id:usize) -> &[f64] {
        let spike_train = self.neuron_control.iter().find(|spike_train| spike_train.id() == id);
        match spike_train {
            Some(spike_train) => spike_train.firing_times(),
            None => &[],
        }
    }
}

/// Represents a simulation program consisting of contiguous simulation intervals.
#[derive(Debug, PartialEq)]
pub struct SimulationProgram {
    intervals: Vec<SimulationInterval>,
}

impl SimulationProgram {
    /// Create a simulation program with the specified intervals.
    /// If necessary, the intervals are sorted.
    /// The function returns an error for empty or non-contiguous simulation programs.
    pub fn build(mut intervals: Vec<SimulationInterval>) -> Result<Self, SimulationError> {
        if intervals.is_empty() {
            return Err(SimulationError::EmptySimulationProgram);
        }

        intervals.sort_by(|i1, i2| i1.start.partial_cmp(&i2.start).expect("A problem occured while sorting the simulation intervals."));
        if intervals.windows(2).any(|w| w[0].end != w[1].start) {
            return Err(SimulationError::NonContiguousSimulationProgram);
        }

        Ok(SimulationProgram { intervals })
    }

    /// Returns the simulation intervals of the simulation program.
    pub fn intervals(&self) -> &[SimulationInterval] {
        &self.intervals[..]
    }

    /// Returns the duration of the simulation program.
    pub fn duration(&self) -> f64 {
        match self.intervals.is_empty() {
            true => 0.0,
            false => self.intervals.last().unwrap().end - self.intervals.first().unwrap().start,
        }
    }
}


#[derive(Debug, PartialEq)]
pub enum SimulationError {
    /// Error for invalid simulation interval.
    InvalidSimulationInterval,
    /// Error for empty simulation program.
    EmptySimulationProgram,
    /// Error for non-contiguous simulation program.
    NonContiguousSimulationProgram,
    /// Error for invalid threshold noise.
    InvalidThresholdNoise,
    /// Error for invalid control.
    InvalidControl,
    /// Error for failed simulation.
    SimulationFailed,
}

impl std::fmt::Display for SimulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SimulationError::InvalidSimulationInterval => write!(f, "Invalid simulation interval: start must be less than end"),
            SimulationError::EmptySimulationProgram => write!(f, "Empty simulation program"),
            SimulationError::NonContiguousSimulationProgram => write!(f, "Non-contiguous simulation program"),
            SimulationError::InvalidThresholdNoise => write!(f, "Invalid threshold noise: standard deviation must be non-negative"),
            SimulationError::InvalidControl => write!(f, "Invalid control"),
            SimulationError::SimulationFailed => write!(f, "Simulation failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_simulation_interval() {
        assert_eq!(SimulationInterval::build(1.0, 0.0, 0.0, vec![]), Err(SimulationError::InvalidSimulationInterval));
        assert_eq!(SimulationInterval::build(0.0, 0.0, 0.0, vec![]), Err(SimulationError::InvalidSimulationInterval));
        assert_eq!(SimulationInterval::build(0.0, std::f64::INFINITY, 0.0, vec![]), Err(SimulationError::InvalidSimulationInterval));
    }

    #[test]
    fn test_invalid_control() {
        let spike_train = SpikeTrain::build(1, &[0.0, 1.5]).unwrap();
        assert_eq!(SimulationInterval::build(0.0, 1.0, 0.0, vec![spike_train]), Err(SimulationError::InvalidControl));
    }

    #[test]
    fn test_invalid_noise() {
        assert_eq!(SimulationInterval::build(0.0, 1.0, -1.0, vec![]), Err(SimulationError::InvalidThresholdNoise));
    }

    #[test]
    fn test_simulation_program() {
        let interval1 = SimulationInterval::build(0.0, 1.0, 0.0, vec![]).unwrap();
        let interval2 = SimulationInterval::build(1.0, 2.0, 0.0, vec![]).unwrap();
        let interval3 = SimulationInterval::build(2.0, 3.0, 0.0, vec![]).unwrap();
        let program = SimulationProgram::build(vec![interval3, interval1, interval2]).unwrap();
        assert_eq!(program.duration(), 3.0);
        assert_eq!(program.intervals[0].start, 0.0);
        assert_eq!(program.intervals[0].end, 1.0);
        assert_eq!(program.intervals[1].start, 1.0);
        assert_eq!(program.intervals[1].end, 2.0);
        assert_eq!(program.intervals[2].start, 2.0);
        assert_eq!(program.intervals[2].end, 3.0);
    }

    #[test]
    fn test_non_contiguous_simulation_program() {
        let interval1 = SimulationInterval::build(0.0, 1.0, 0.0, vec![]).unwrap();
        let interval2 = SimulationInterval::build(0.5, 2.0, 0.0, vec![]).unwrap();
        assert_eq!(SimulationProgram::build(vec![interval1, interval2]), Err(SimulationError::NonContiguousSimulationProgram));

        let interval1 = SimulationInterval::build(0.0, 1.0, 0.0, vec![]).unwrap();
        let interval2 = SimulationInterval::build(1.5, 2.0, 0.0, vec![]).unwrap();
        assert_eq!(SimulationProgram::build(vec![interval1, interval2]), Err(SimulationError::NonContiguousSimulationProgram));
    }
}