//! Module implementing the spiking neurons.

use super::connection::Connection;
use crate::spike_train::error::SpikeTrainError;
use embed_doc_image::embed_doc_image;
use serde::{Deserialize, Serialize};

// #[derive(Debug, PartialEq)]
// pub enum NeuronError {
//     InputError,
//     /// Error for invalid firing time, e.g., violated refractory period.
//     InvalidFiringTime,
// }

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Neuron {
    /// Unique identifier for the neuron
    id: usize,
    /// Minimum potential required for the neuron to fire
    threshold: f64,
    /// Historical record of times when the neuron fired
    firing_times: Vec<f64>,
    // /// Collection of inputs connected to this neuron
    // inputs: Vec<Connection>,
}

impl Neuron {
    /// Creates a new neuron with a (unique) id and (nominal) firing threshold.
    pub fn new(id: usize, threshold: f64) -> Self {
        Neuron {
            id,
            threshold,
            firing_times: Vec::new(),
        }
    }

    /// Creates a new neuron with a (unique) id and (nominal) firing threshold.
    pub fn build(
        id: usize,
        threshold: f64,
        mut firing_times: Vec<f64>,
    ) -> Result<Self, SpikeTrainError> {
        firing_times.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("A problem occured while sorting the provided firing times.")
        });

        if firing_times
            .windows(2)
            .map(|w| (w[1] - w[0]))
            .any(|dt| dt <= 1.0)
        {
            return Err(SpikeTrainError::RefractoryPeriodViolation);
        }

        Ok(Neuron {
            id,
            threshold,
            firing_times,
        })
    }

    pub fn id(&self) -> usize {
        self.id
    }

    // pub fn add_input(&mut self, source_id: usize, weight: f64, delay: f64) {
    //     self.inputs
    //         .push(Connection::new(source_id, self.id, weight, delay));
    // }

    // pub fn inputs(&self) -> &[Connection] {
    //     &self.inputs
    // }

    pub fn firing_times(&self) -> &[f64] {
        &self.firing_times[..]
    }

    /// Calculates the neuron's potential at a given time by summing the contributions from its inputs.
    /// When the potential exceeds the threshold, the neuron fires.
    ///
    /// ![A Foobaring][neuron]
    ///
    #[embed_doc_image("neuron", "images/neuron.svg")]
    pub fn potential(&self, t: f64, connections: &[Connection]) -> f64 {
        connections
            .iter()
            .map(|connection| connection.output(t))
            .sum()
    }

    pub fn add_firing_time(&mut self, t: f64) -> Result<(), SpikeTrainError> {
        match self.firing_times.last() {
            Some(&last) if t - 1.0 <= last => Err(SpikeTrainError::RefractoryPeriodViolation),
            _ => {
                self.firing_times.push(t);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_firing_times() {
        let neuron = Neuron::build(0, 1.0, vec![0.0, 0.5]);
        assert_eq!(neuron, Err(SpikeTrainError::RefractoryPeriodViolation));

        let neuron = Neuron::build(0, 1.0, vec![0.0, 7.0, 3.0]).unwrap();
        assert_eq!(neuron.firing_times, vec![0.0, 3.0, 7.0]);
    }

    #[test]
    fn test_add_firing_time() {
        let mut neuron = Neuron::new(0, 1.0);
        assert_eq!(neuron.add_firing_time(0.0), Ok(()));
        assert_eq!(neuron.firing_times, vec![0.0]);
        assert_eq!(neuron.add_firing_time(7.0), Ok(()));
        assert_eq!(neuron.firing_times, vec![0.0, 7.0]);
    }
}
