// Neurons are the basic building blocks of the network.
// They are connected to each other through Inputs.
// Neurons have a threshold, which is the value that their potential must reach in order to fire.
// Neurons have a potential, which is the value that is accumulated over time.
// Should it be a struct or a trait object?

// mod super::Input::Input;
use super::connection::Input;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Neuron {
    id: usize,
    threshold: f64,
    firing_times: Vec<f64>,
    inputs: Vec<Input> // or slice?
}

impl Neuron {
    ///
    pub fn new(id:usize, threshold: f64, inputs: Vec<Input>) -> Neuron {
        Neuron {
            id: id,
            threshold: threshold,
            firing_times: Vec::new(),
            inputs: inputs,
        }
    }

    pub fn firing_times(&self) -> &Vec<f64> {
        &self.firing_times
    }

    pub fn fire(&mut self, time: f64) {
        self.firing_times.push(time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron() {
        let neuron = Neuron::new(0, 1.0, Vec::new());
        assert_eq!(neuron.id, 0);
        assert_eq!(neuron.threshold, 1.0);
        assert_eq!(neuron.firing_times.len(), 0);
    }

    #[test]
    fn test_fire() {
        let mut neuron = Neuron::new(0, 1.0, Vec::new());
        neuron.fire(0.0);
        assert_eq!(neuron.firing_times.len(), 1);
        neuron.fire(1.45);
        assert_eq!(neuron.firing_times.len(), 2);
    }
}