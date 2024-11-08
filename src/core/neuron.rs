//! Module implementing the spiking neurons.

use embed_doc_image::embed_doc_image;
use serde::{Deserialize, Serialize};

use crate::core::{REFRACTORY_PERIOD, FIRING_THRESHOLD};
use crate::simulator::TIME_RESOLUTION;

use super::connection::{Connection, ConnectionError};
use super::spike_train::SpikeTrainError;

/// Represents an input to a neuron.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Input {
    /// ID of the sending neuron
    source_id: usize,
    /// Weight of the input
    weight: f64,
    /// Delay of the input
    delay: f64,
    /// Times at which the sending neuron fired
    firing_times: Vec<f64>,
}

impl Input {
    /// Create a new input with the specified parameters.
    /// Note that the function cannot check if the source id is valid; this check must be done at the network level.
    pub fn from_connection(connection: &Connection) -> Self {
        Input {
            source_id: connection.source_id(),
            weight: connection.weight(),
            delay: connection.delay(),
            firing_times: vec![],
        }
    }

    /// Add a firing time to the input.
    /// The function does not check the refractory period: in principle, the input can fire without any restriction.
    pub fn add_firing_time(&mut self, t: f64) {
        self.firing_times.push(t + self.delay);
    }

    /// Extend the firing times of the input.
    /// The function does not check the refractory period: in principle, the input can fire without any restriction.
    pub fn extend_firing_times(&mut self, firing_times: &[f64]) {
        self.firing_times.extend(firing_times.iter().map(|ft| ft + self.delay));
    }

    /// Evaluate the input signal at a given time.
    pub fn eval(&self, t: f64) -> f64 {
        self.firing_times
            .iter()
            .map(|ft| t - ft)
            .filter_map(|dt| {
                if dt > 0. {
                    Some(2_f64 * dt * (-dt).exp())
                } else {
                    None
                }
            })
            .sum()
    }

    /// Returns the weight of the connection.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Returns the delay of the connection.
    pub fn delay(&self) -> f64 {
        self.delay
    }

    /// Returns the ID of the sending neuron.
    pub fn source_id(&self) -> usize {
        self.source_id
    }
}

/// Represents a spiking neuron.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Neuron {
    id: usize,
    threshold: f64,
    firing_times: Vec<f64>,
    inputs: Vec<Input>,
}

impl Neuron {
    pub fn new(id: usize) -> Self {
        Neuron {
            id,
            threshold: FIRING_THRESHOLD,
            firing_times: vec![],
            inputs: vec![],
        }
    }

    /// Extend the neuron's firing times with new ones.
    /// If necessary, the provided firing times are sorted before being added.
    /// The function returns an error if the refractory period is violated.
    pub fn extend_firing_times(&mut self, firing_times: &[f64]) -> Result<(), SpikeTrainError> {
        let mut firing_times = firing_times.to_vec();
        firing_times.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("A problem occured while sorting the provided firing times.")
        });
        if firing_times
            .windows(2)
            .map(|w| (w[1] - w[0]))
            .any(|dt| dt <= REFRACTORY_PERIOD)
        {
            return Err(SpikeTrainError::RefractoryPeriodViolation);
        }
        match (firing_times.first(), self.firing_times.last()) {
            (Some(&first), Some(&last)) => {
                if first <= last + REFRACTORY_PERIOD {
                    return Err(SpikeTrainError::RefractoryPeriodViolation);
                }
            }
            _ => {}
        }
        self.firing_times.extend(firing_times);
        Ok(())
    }

    /// Add a firing time to the neuron's firing times.
    /// The function returns an error if the refractory period is violated.
    pub fn add_firing_time(&mut self, t: f64) -> Result<(), SpikeTrainError> {
        match self.firing_times.last() {
            Some(&last) if t <= last + REFRACTORY_PERIOD => Err(SpikeTrainError::RefractoryPeriodViolation),
            _ => {
                self.firing_times.push(t);
                Ok(())
            }
        }
    }

    /// Add firing time to the neuron's inputs whose source id matches the provided one.
    pub fn add_inputs_firing_time(&mut self, source_id: usize, firing_time: f64) {
        for input in self.inputs.iter_mut().filter(|input| input.source_id() == source_id) {
            input.add_firing_time(firing_time);
        }
    }

    /// Extend the firing times of the neuron's inputs whose source id matches the provided ones.
    pub fn extend_inputs_firing_times(&mut self, source_id:usize, firing_times: &[f64]) {
        for input in self.inputs.iter_mut().filter(|input| input.source_id() == source_id) {
            input.extend_firing_times(firing_times);
        }
    }

    /// Returns the ID of the neuron.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the threshold of the neuron.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Returns the number of inputs of the neuron.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }
    
    /// Add an input to the neuron with the specified parameters.
    /// If the target neuron of the connection is not the neuron, returns an error.
    pub fn add_input(&mut self, connection: &Connection) -> Result<(), ConnectionError> {
        if connection.target_id() != self.id {
            return Err(ConnectionError::InvalidTargetID);
        }
        self.inputs.push(Input::from_connection(connection));
        Ok(())
    }

    /// Returns a slice of inputs of the neuron.
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    /// Returns a slice of firing times of the neuron.
    pub fn firing_times(&self) -> &[f64] {
        &self.firing_times[..]
    }

    /// Calculates the neuron's potential at a given time by summing the contributions from its inputs.
    /// When the potential exceeds the threshold, the neuron fires.
    ///
    /// ![A Foobaring][neuron]
    ///
    #[embed_doc_image("neuron", "images/neuron.svg")]
    pub fn potential(&self, t: f64) -> f64 {
        self.inputs.iter().map(|input| input.eval(t)).sum()
    }

    // Returns the crossing time of the threshold by the neuron's potential.
    // The function uses a binary search algorithm to find the crossing time.
    // It assumes that the potential is a continuous function of time on the interval [start, end] and that potential(start) < threshold < potential(end).
    fn solve_threshold_crossing(&self, start: f64, end: f64) -> f64 {
        let dt = end - start;
        let mid = (start + end) / 2.0;
        if dt <= TIME_RESOLUTION {
            return mid;
        }

        match self.potential(mid) >= self.threshold() {
            true => self.solve_threshold_crossing(start, mid),
            false => self.solve_threshold_crossing(mid, end),
        }
    }

    /// Simulate the neuron's dynamic on the interval [start, start + dt).
    /// The function returns the unique time at which the neuron fires during the interval, if any.
    pub fn step(&mut self, start: f64, dt: f64) -> Option<f64> {
        let mut start = start;
        let end = start + dt;

        if let Some(&last) = self.firing_times().last() {
            // If the neuron is in refractory period at the end of the interval, it cannot fire during the interval.
            if end <= last + REFRACTORY_PERIOD {
                return None;
            }

            // Otherwise, make sure to start the search after the refractory period.
            start = start.max(last + REFRACTORY_PERIOD + TIME_RESOLUTION);
        }

        match (self.potential(start) >= self.threshold(), self.potential(end) >= self.threshold()) {
            // If the potential is above the threshold at the start of the interval, the neuron fires immediately.
            (true, _) => Some(start),
            // If the potential cross the threshold during the interval, find the crossing time.
            (false, true) => Some(self.solve_threshold_crossing(start, end)),
            // Otherwise, the neuron does not fire during the interval.
            (false, false) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::E;

    // #[test]
    // fn test_input_invalid_delay() {
    //     assert_eq!(
    //         Input::build(0, 1.0, -1.0),
    //         Err(NetworkError::InvalidDelay)
    //     );
    // }

    #[test]
    fn test_input_eval() {
        let mut input = Input::from_connection(&Connection::build(0, 1, 1.0, 1.0).unwrap());
        input.add_firing_time(0.0);
        assert_eq!(input.eval(0.0), 0.0);
        assert_eq!(input.eval(1.0), 0.0);
        assert_eq!(input.eval(2.0), 2.0 / E);
    }

    #[test]
    fn test_input_add_firing_time() {
        let mut input = Input::from_connection(&Connection::build(0, 1, 1.0, 1.0).unwrap());
        input.add_firing_time(0.0);
        assert_eq!(input.firing_times, [1.0]);
        input.add_firing_time(7.0);
        assert_eq!(input.firing_times, [1.0, 8.0]);
    }

    #[test]
    fn test_input_extend_firing_times() {
        let mut input = Input::from_connection(&Connection::build(0, 1, 1.0, 1.0).unwrap());
        input.extend_firing_times(&[0.0, 3.0, 7.0]);
        assert_eq!(input.firing_times, [1.0, 4.0, 8.0]);
        input.extend_firing_times(&[10.0]);
        
    }

    #[test]
    fn test_neuron_extend_firing_times() {
        let mut neuron = Neuron::new(0);
        assert_eq!(neuron.extend_firing_times(&[0.0, 3.0, 7.0]), Ok(()));
        assert_eq!(neuron.firing_times, [0.0, 3.0, 7.0]);
        assert_eq!(
            neuron.extend_firing_times(&[6.0]),
            Err(SpikeTrainError::RefractoryPeriodViolation)
        );
        assert_eq!(neuron.firing_times, [0.0, 3.0, 7.0]);
        assert_eq!(neuron.extend_firing_times(&[10.0, 12.0]), Ok(()));
        assert_eq!(neuron.firing_times, [0.0, 3.0, 7.0, 10.0, 12.0]);
    }

    #[test]
    fn test_neuron_add_firing_time() {
        let mut neuron = Neuron::new(0);
        assert_eq!(neuron.add_firing_time(0.0), Ok(()));
        assert_eq!(neuron.firing_times, [0.0]);
        assert_eq!(neuron.add_firing_time(7.0), Ok(()));
        assert_eq!(neuron.firing_times, [0.0, 7.0]);
        assert_eq!(
            neuron.add_firing_time(5.0),
            Err(SpikeTrainError::RefractoryPeriodViolation)
        );
        assert_eq!(neuron.firing_times, [0.0, 7.0]);
    }

    #[test]
    fn test_neuron_add_inputs_firing_time() {
        let mut neuron = Neuron::new(0);
        neuron.add_input(&Connection::build(7, 0, 1.0, 1.0).unwrap()).unwrap();
        neuron.add_input(&Connection::build(42, 0, 1.0, 1.0).unwrap()).unwrap();
        neuron.add_input(&Connection::build(7, 0, -0.5, 2.0).unwrap()).unwrap();

        neuron.add_inputs_firing_time(7, 0.0);
        assert_eq!(neuron.inputs[0].firing_times, [1.0]);
        assert_eq!(neuron.inputs[1].firing_times, Vec::<f64>::new());
        assert_eq!(neuron.inputs[2].firing_times, [2.0]);

        neuron.add_inputs_firing_time(42, 7.0);
        assert_eq!(neuron.inputs[0].firing_times, [1.0]);
        assert_eq!(neuron.inputs[1].firing_times, [8.0]);
        assert_eq!(neuron.inputs[2].firing_times, [2.0]);

        neuron.add_inputs_firing_time(9, 12.0);
        assert_eq!(neuron.inputs[0].firing_times, [1.0]);
        assert_eq!(neuron.inputs[1].firing_times, [8.0]);
        assert_eq!(neuron.inputs[2].firing_times, [2.0]);

        neuron.add_inputs_firing_time(42, 12.5);
        assert_eq!(neuron.inputs[0].firing_times, [1.0]);
        assert_eq!(neuron.inputs[1].firing_times, [8.0, 13.5]);
        assert_eq!(neuron.inputs[2].firing_times, [2.0]);
    }
}
