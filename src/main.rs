use rand::rngs::StdRng;
use rand::SeedableRng;
use env_logger;

// use grb::prelude::*;
// use grb::VarType::Continuous;
// use grb::{Model, Var, Expr};
// use grb::{c, add_ctsvar};
// use grb::ModelSense::Minimize;
// use grb::expr::QuadExpr;

use rusty_snn::spike_train::SpikeTrain;
// use rusty_snn::simulator::SimulationProgram;
use rusty_snn::network::{Network, Topology};
use rusty_snn::optim::Objective;

const SEED: u64 = 42;
const NUM_NEURONS: usize = 200;
const NUM_CONNECTIONS: usize = 200*500;
const TOPOLOGY: Topology = Topology::Fin;
const LIM_WEIGHTS: (f64, f64) = (-0.2, 0.2);
const LIM_DELAYS: (f64, f64) = (0.1, 10.0);
const PERIOD: f64 = 50.0;
const FIRING_RATE: f64 = 0.2;
const MAX_LEVEL: f64 = 0.0;
const MIN_SLOPE: f64 = 2.0;
const HALF_WIDTH: f64 = 0.2;

fn main() {
    env_logger::init();

    let mut rng = StdRng::seed_from_u64(SEED);
    let mut network = Network::rand(
        NUM_NEURONS,
        NUM_CONNECTIONS,
        LIM_WEIGHTS,
        LIM_DELAYS,
        TOPOLOGY,
        &mut rng,
    )
    .unwrap();
    let spike_trains = SpikeTrain::rand(NUM_NEURONS, PERIOD, FIRING_RATE, &mut rng).unwrap();

    network
        .memorize_periodic_spike_trains(
            &spike_trains,
            PERIOD,
            LIM_WEIGHTS,
            MAX_LEVEL,
            MIN_SLOPE,
            HALF_WIDTH,
            Objective::L2Norm
        )
        .unwrap();
}
