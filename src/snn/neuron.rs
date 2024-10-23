// Neurons are the basic building blocks of the network.
// They are connected to each other through Inputs.
// Neurons have a threshold, which is the value that their potential must reach in order to fire.
// Neurons have a potential, which is the value that is accumulated over time.
// Should it be a struct or a trait object?

// mod super::Input::Input;
use super::connection::Input;

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

    // pub fn add_input(&mut self, input: &Input) {
    //     self.inputs.push(input);
    // }

    // pub fn add_output(&mut self, output: &Input) {
    //     self.outputs.push(output);
    // }

    pub fn firing_times(&self) -> &Vec<f64> {
        &self.firing_times
    }

    pub fn fire(&mut self, time: f64) {
        self.firing_times.push(time);
    }
}