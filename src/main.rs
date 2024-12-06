use rand::rngs::StdRng;
use rand::SeedableRng;

use rusty_snn::sampler::network::{NetworkSampler, Topology};
use rusty_snn::sampler::spike_train::PeriodicSpikeTrainSampler;
use rusty_snn::simulator::simulator::SimulationProgram;

const SEED: u64 = 42;
const NUM_NEURONS: usize = 200;
const NUM_CONNECTIONS: usize = 1000;
const WEIGHT_RANGE: (f64, f64) = (-1.0, 1.0);
const DELAY_RANGE: (f64, f64) = (0.1, 10.0);
const PERIOD: f64 = 100.0;
const FIRING_RATE: f64 = 0.2;
const TOPOLOGY: Topology = Topology::Fin;

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);

    // Randomly generate a network
    let network_sampler = NetworkSampler::build(
        NUM_NEURONS,
        NUM_CONNECTIONS,
        WEIGHT_RANGE,
        DELAY_RANGE,
        TOPOLOGY,
    )
    .unwrap();
    let mut network = network_sampler.sample(&mut rng);

    // Randomly generate a spike train
    let spike_train_sampler = PeriodicSpikeTrainSampler::build(PERIOD, FIRING_RATE).unwrap();
    let spike_trains = spike_train_sampler.sample(NUM_NEURONS, &mut rng);

    let program = SimulationProgram::build(100.0, 1000.0, 0.1, &spike_trains).unwrap();

    if let Err(e) = network.run(&program, &mut rng) {
        eprintln!("Simulation failed: {}", e);
    }
}
