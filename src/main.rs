use rand::rngs::StdRng;
use rand::SeedableRng;

use rusty_snn::sampler::network::{NetworkSampler, Topology};
use rusty_snn::sampler::spike_train::PeriodicSpikeTrainSampler;
use rusty_snn::simulator::simulator::SimulationProgram;

static SEED: u64 = 42;


fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);

    // Randomly generate a network
    let network_sampler = NetworkSampler::build(10, 100, (-1.0, 1.0), (0.1, 10.0), Topology::Fin).unwrap();
    let mut network = network_sampler.sample(&mut rng);

    // Randomly generate a spike train
    let spike_train_sampler = PeriodicSpikeTrainSampler::build(10.0, 0.2).unwrap();
    let spike_trains = spike_train_sampler.sample(5, &mut rng);

    let program = SimulationProgram::build(0.0, 100.0, 0.1, spike_trains).unwrap();

    network.run(&program, &mut rng).unwrap();
}
