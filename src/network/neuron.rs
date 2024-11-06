//! Module implementing the spiking neurons.

use embed_doc_image::embed_doc_image;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{Receiver, Sender};

use super::error::NetworkError;
use crate::spike_train::error::SpikeTrainError;

// #[derive(Debug, PartialEq)]
// pub enum NeuronError {
//     InputError,
//     /// Error for invalid firing time, e.g., violated refractory period.
//     InvalidFiringTime,
// }

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
    pub fn new(source_id: usize, weight: f64, delay: f64) -> Self {
        Input {
            source_id,
            weight,
            delay,
            firing_times: Vec::new(),
        }
    }

    /// Create a new input with the specified parameters.
    /// Returns an error if the delay is negative.
    /// Note that the function cannot check if the source id is valid; this check must be done at the network level.
    pub fn build(source_id: usize, weight: f64, delay: f64) -> Result<Self, NetworkError> {
        if delay < 0.0 {
            return Err(NetworkError::InvalidDelay);
        }
        Ok(Input {
            source_id,
            weight,
            delay,
            firing_times: Vec::new(),
        })
    }

    /// Add a firing time to the input train.
    /// The function returns an error if the refractory period is violated.
    pub fn add_firing_time(&mut self, t: f64) -> Result<(), SpikeTrainError> {
        let t = t + self.delay;
        match self.firing_times.last() {
            Some(&last) if t - 1.0 <= last => Err(SpikeTrainError::RefractoryPeriodViolation),
            _ => {
                self.firing_times.push(t);
                Ok(())
            }
        }
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

    /// Returns the id of the sending neuron.
    pub fn source_id(&self) -> usize {
        self.source_id
    }
}

/// Represents a message sent by a neuron.
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Message {
    firing_time: f64,
    source_id: usize,
}

impl Message {
    /// Create a new message to be sent.
    pub fn new(firing_time: f64, source_id: usize) -> Self {
        Message { firing_time, source_id }
    }
}

/// Represents a spiking neuron.
#[derive(Serialize, Deserialize)]
pub struct Neuron {
    id: usize,
    threshold: f64,
    firing_times: Vec<f64>,
    inputs: Vec<Input>,
    #[serde(skip)]
    rx: Option<Receiver<Message>>,
    #[serde(skip)]
    txs: Vec<Sender<Message>>,
}

impl PartialEq for Neuron {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.threshold == other.threshold
            && self.firing_times == other.firing_times
            && self.inputs == other.inputs
    }
}

impl std::fmt::Debug for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Neuron")
            .field("id", &self.id)
            .field("threshold", &self.threshold)
            .field("firing_times", &self.firing_times)
            .field("inputs", &self.inputs)
            .finish()
    }
}

impl Clone for Neuron {
    fn clone(&self) -> Self {
        Neuron {
            id: self.id,
            threshold: self.threshold,
            firing_times: self.firing_times.clone(),
            inputs: self.inputs.clone(),
            rx: None,
            txs: Vec::new(),
        }
    }
}

impl Neuron {
    pub fn new(id: usize, threshold: f64) -> Self {
        Neuron {
            id,
            threshold,
            firing_times: Vec::new(),
            inputs: Vec::new(),
            rx: None,
            txs: Vec::new(),
        }
    }

    /// Extend the neuron's firing times with new ones.
    /// If necessary, the provided firing times are sorted before being added.
    /// The function returns an error if the refractory period is violated.
    pub fn extend_firing_times(&mut self, firing_times: Vec<f64>) -> Result<(), SpikeTrainError> {
        let mut firing_times = firing_times.clone();
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
        match (firing_times.first(), self.firing_times.last()) {
            (Some(&first), Some(&last)) => {
                if first <= last + 1.0 {
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
            Some(&last) if t - 1.0 <= last => Err(SpikeTrainError::RefractoryPeriodViolation),
            _ => {
                self.firing_times.push(t);
                Ok(())
            }
        }
    }

    pub fn reset_senders(&mut self) {
        self.txs = Vec::new();
    }

    pub fn add_sender(&mut self, tx: Sender<Message>) {
        self.txs.push(tx);
    }

    pub fn set_receiver(&mut self, rx: Receiver<Message>) {
        self.rx = Some(rx);
    }

    pub fn reset_receiver(&mut self) {
        self.rx = None;
    }

    /// Simulate the neuron's behavior at a given time for a short duration dt.
    /// If the neuron fires, it sends the spike location throught its txs.
    pub fn process_and_send(&mut self, t: f64, dt: f64) {
        todo!()
    }

    fn update_inputs(&mut self, source_id: usize, time: f64) -> Result<(), SpikeTrainError> {
        for input in self.inputs.iter_mut() {
            if input.source_id() == source_id {
                input.add_firing_time(time + input.delay())?;
            }
        }
        Ok(())
    }

    pub fn receive_and_process(&mut self) -> Result<(), SpikeTrainError> {
        match self.rx {
            Some(ref rx) => {
                let messages: Vec<Message> = rx.try_iter().collect();
                for msg in messages {
                    self.update_inputs(msg.source_id, msg.firing_time)?;
                }
                Ok(())
            }
            None => Ok(()),
        }
    }
    // pub struct Neuron {
    //     /// Unique identifier for the neuron
    //     id: usize,
    //     /// Minimum potential required for the neuron to fire
    //     threshold: f64,
    //     /// Historical record of times when the neuron fired
    //     firing_times: Vec<f64>,
    //     /// Collection of inputs connected to this neuron
    //     inputs: Vec<Input>,
    // }

    // impl Neuron {
    //     /// Creates a new neuron with a (unique) id and (nominal) firing threshold.
    //     pub fn new(id: usize, threshold: f64) -> Self {
    //         Neuron {
    //             id,
    //             threshold,
    //             firing_times: Vec::new(),
    //         }
    //     }

    //     /// Creates a new neuron with a (unique) id and (nominal) firing threshold.
    //     pub fn build(
    //         id: usize,
    //         threshold: f64,
    //         mut firing_times: Vec<f64>,
    //     ) -> Result<Self, SpikeTrainError> {
    //         firing_times.sort_by(|a, b| {
    //             a.partial_cmp(b)
    //                 .expect("A problem occured while sorting the provided firing times.")
    //         });

    //         if firing_times
    //             .windows(2)
    //             .map(|w| (w[1] - w[0]))
    //             .any(|dt| dt <= 1.0)
    //         {
    //             return Err(SpikeTrainError::RefractoryPeriodViolation);
    //         }

    //         Ok(Neuron {
    //             id,
    //             threshold,
    //             firing_times,
    //         })
    //     }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn add_input(&mut self, source_id: usize, weight: f64, delay: f64) {
        self.inputs.push(Input::new(source_id, weight, delay));
    }

    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

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

    // pub fn simulate(&self, t: f64, connections: &[Input]) -> bool {
    //     let potential = self.potential(t, connections);
    //     if potential >= self.threshold {
    //         self.add_firing_time(t).is_ok()
    //     } else {
    //         false
    //     }
    // }
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
        let mut input = Input::new(0, 1.0, 1.0);
        input.add_firing_time(0.0);
        assert_eq!(input.eval(0.0), 0.0);
        assert_eq!(input.eval(1.0), 0.0);
        assert_eq!(input.eval(2.0), 2.0 / E);
    }

    #[test]
    fn test_input_add_firing_time() {
        let mut input = Input::new(42, 1.0, 1.0);
        assert_eq!(input.add_firing_time(0.0), Ok(()));
        assert_eq!(input.firing_times, vec![1.0]);
        assert_eq!(input.add_firing_time(7.0), Ok(()));
        assert_eq!(input.firing_times, vec![1.0, 8.0]);
        assert_eq!(
            input.add_firing_time(5.0),
            Err(SpikeTrainError::RefractoryPeriodViolation)
        );
        assert_eq!(input.firing_times, vec![1.0, 8.0]);
    }

    #[test]
    fn test_neuron_extend_firing_times() {
        let mut neuron = Neuron::new(0, 1.0);
        assert_eq!(neuron.extend_firing_times(vec![0.0, 3.0, 7.0]), Ok(()));
        assert_eq!(neuron.firing_times, vec![0.0, 3.0, 7.0]);
        assert_eq!(
            neuron.extend_firing_times(vec![6.0]),
            Err(SpikeTrainError::RefractoryPeriodViolation)
        );
        assert_eq!(neuron.firing_times, vec![0.0, 3.0, 7.0]);
        assert_eq!(neuron.extend_firing_times(vec![10.0, 12.0]), Ok(()));
        assert_eq!(neuron.firing_times, vec![0.0, 3.0, 7.0, 10.0, 12.0]);
    }

    #[test]
    fn test_neuron_add_firing_time() {
        let mut neuron = Neuron::new(0, 1.0);
        assert_eq!(neuron.add_firing_time(0.0), Ok(()));
        assert_eq!(neuron.firing_times, vec![0.0]);
        assert_eq!(neuron.add_firing_time(7.0), Ok(()));
        assert_eq!(neuron.firing_times, vec![0.0, 7.0]);
        assert_eq!(
            neuron.add_firing_time(5.0),
            Err(SpikeTrainError::RefractoryPeriodViolation)
        );
        assert_eq!(neuron.firing_times, vec![0.0, 7.0]);
    }
}
