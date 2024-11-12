use rand::rngs::StdRng;
use rand::SeedableRng;

use rusty_snn::sampler::network::{NetworkSampler, Topology};
use rusty_snn::simulator::simulator::SimulationProgram;
use rusty_snn::core::spike_train::SpikeTrain;

static SEED: u64 = 42;

#[tokio::main]
async fn main() {
    // Create a network sampler to generate networks with 3 neurons, 12 connections, weights in the range (-0.1, 0.1), delays in the range (0.1, 10.0), and fixed-in-degree topology.
    let sampler = NetworkSampler::build(3, 12, (-1.0, 1.0), (0.1, 10.0), Topology::Fin).unwrap();

    // Sample a network from the distribution
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut network = sampler.sample(&mut rng);

    let neuron_control = vec![SpikeTrain::build(0, &[0.0, 2.0]).unwrap(), SpikeTrain::build(1, &[0.5, 2.5]).unwrap(), SpikeTrain::build(2, &[1.0, 2.5, 4.0]).unwrap()];
    let program = SimulationProgram::build(0.0, 10.0, 0.1, neuron_control).unwrap();

    network.run(&program, SEED).await.unwrap();
}
