use std::sync::mpsc;

use crate::network::neuron::Neuron;
use crate::network::network::Network;
enum SimulationError {}

impl Network {
    pub fn init_mpsc_channels(&mut self) -> Result<(), SimulationError> {
        // set up the channels for simulation...
        // for every neuron, create a new communication channel and set the receiver to the neuron.
        // clone the sender and add it to all neurons that are connected to the neuron.
        for neuron in &mut self.neurons {
            neuron.reset_sender();
        }
        
        for id in 0..self.num_neurons() {
            let (tx, rx) = mpsc::channel();
            self.neurons[l].set_receiver(rx);
            for source_id in self.connections.iter().filter(|c| c.target_id() == id).map(|c| c.source_id()).unique() {
                self.neurons[source_id].add_sender(tx.clone());
            }
        }

        Ok(())
    }
}