use env_logger;
use rand::rngs::StdRng;
use rand::SeedableRng;

use grb::prelude::*;

use rusty_snn::spike_train::SpikeTrain;
// use rusty_snn::simulator::SimulationProgram;
use rusty_snn::network::{Network, Topology};
use rusty_snn::optim::Objective;

const SEED: u64 = 42;
const NUM_NEURONS: usize = 200;
const NUM_CONNECTIONS: usize = 200 * 500;
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

    let mut model = grb::Model::new("test").unwrap();
    let weights: Vec<Var> = (0..3)
        .map(|_| add_ctsvar!(model, bounds: -1..1).unwrap())
        .collect();
    let mut slacks = Vec::new();
    let mut obj_expr = grb::expr::LinExpr::new();
    for (i, &weight) in weights.iter().enumerate() {
        let slack = add_ctsvar!(model).unwrap();
        obj_expr.add_term(1.0, slack);
        model.add_constr("", c!(weight >= -slack)).unwrap();
        model.add_constr("", c!(weight <= slack)).unwrap();
        // model
        //     .add_constrs(vec![
        //         (&format!("min_slack_{}", i).as_str(), c!(weight >= -slack)),
        //         (&format!("max_slack_{}", i).as_str(), c!(weight <= slack)),
        //     ])
        //     .unwrap();
        slacks.push(slack);
    }
    model.set_objective(obj_expr, ModelSense::Minimize).unwrap();
    model
        .add_constr("cstr", c!(weights[0] + weights[1] + weights[2] == 1.0))
        .unwrap();

    model.optimize().unwrap();

    let num_constrs = model
        .get_attr(grb::attribute::ModelIntAttr::NumConstrs)
        .unwrap();
    let obj_val = model
        .get_attr(grb::attribute::ModelDoubleAttr::ObjVal)
        .unwrap();

    println!("Number of constraints: {}", num_constrs);
    println!("Objective value: {}", obj_val);
    // collect all slack values
    let slack_val: Vec<f64> = (0..3)
        .map(|i| {
            model
                .get_obj_attr(grb::attribute::VarDoubleAttr::X, &slacks[i])
                .unwrap()
        })
        .collect();
    println!("Slack values: {:?}", slack_val);
    let weights_val: Vec<f64> = (0..3)
        .map(|i| {
            model
                .get_obj_attr(grb::attribute::VarDoubleAttr::X, &weights[i])
                .unwrap()
        })
        .collect();
    println!("Weights values: {:?}", weights_val);

    // let mut rng = StdRng::seed_from_u64(SEED);
    // let mut network = Network::rand(
    //     NUM_NEURONS,
    //     NUM_CONNECTIONS,
    //     LIM_WEIGHTS,
    //     LIM_DELAYS,
    //     TOPOLOGY,
    //     &mut rng,
    // )
    // .unwrap();
    // let spike_trains = SpikeTrain::rand(NUM_NEURONS, PERIOD, FIRING_RATE, &mut rng).unwrap();

    // network
    //     .memorize_periodic_spike_trains(
    //         &spike_trains,
    //         PERIOD,
    //         LIM_WEIGHTS,
    //         MAX_LEVEL,
    //         MIN_SLOPE,
    //         HALF_WIDTH,
    //         Objective::L2Norm
    //     )
    //     .unwrap();
}
