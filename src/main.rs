// use rand::rngs::StdRng;
// use rand::SeedableRng;
// use grb::prelude::*;
// use grb::VarType::Continuous;
use grb::{Model, Var, Expr};
use grb::{c, add_ctsvar};
use grb::ModelSense::Minimize;
use grb::expr::QuadExpr;

// use rusty_snn::spike_train::PeriodicSpikeTrainSampler;
// use rusty_snn::simulator::SimulationProgram;
// use rusty_snn::network::{Network, Topology};

// const SEED: u64 = 42;
// const NUM_NEURONS: usize = 1000;
// const NUM_CONNECTIONS: usize = 1000;
// const WEIGHT_RANGE: (f64, f64) = (0.5, 0.75);
// const DELAY_RANGE: (f64, f64) = (0.1, 10.0);
// const PERIOD: f64 = 100.0;
// const FIRING_RATE: f64 = 0.2;
// const TOPOLOGY: Topology = Topology::Fin;

fn main() -> Result<(), grb::Error> {
    println!("1 < INFINITY: {:?}", 1.0 < f64::INFINITY);
    println!("1 < 2 + INFINITY: {:?}", 1.0 < 2.0 + f64::INFINITY);

    // let mut model = Model::new("neuron_1")?;
    
    // // add decision variables with no bounds
    // let w: Vec<Var> = (0..1000).map(|k| add_ctsvar!(model, name: &format!("x{}", k), bounds: ..).unwrap()).collect();
    // println!("{:?}", w);

    // let mut objective = QuadExpr::new();
    // for k in 0..1000 {
    //     objective.add_qterm(1.0, w[k], w[k]);
    // }
    // model.set_objective(objective, Minimize)?;
    
    // // add linear constraints
    // let constraints = model.add_constr("c0", c!(w[0] + w[2] >= 1.0))?;
    
    // // // model is lazily updated by default
    // // assert_eq!(model.get_obj_attr(attr::VarName, &x1).unwrap_err(), grb::Error::ModelObjectPending);
    // // assert_eq!(model.get_attr(attr::IsMIP)?, 0);
    
    // // // set the objective function, which updates the model objects (variables and constraints).
    // // // One could also call `model.update()`
    // // assert_eq!(model.get_obj_attr(attr::VarName, &x1)?, "x1");
    // // assert_eq!(model.get_attr(attr::IsMIP)?, 1);
    
    // // // write model to the file.
    // // model.write("model.lp")?;
    
    // // optimize the model
    // model.optimize()?;
    // println!("model status: {:?}", model.status().unwrap());
    
    // // // Querying a model attribute
    // // assert_eq!(model.get_attr(attr::ObjVal)? , 59.0);
    
    // // // Querying a model object attributes
    // // assert_eq!(model.get_obj_attr(attr::Slack, &c0)?, -34.5);
    // // let x1_name = model.get_obj_attr(attr::VarName, &x1)?;
    
    // // // Querying an attribute for multiple model objects
    // // let val = model.get_obj_attr_batch(attr::X, vec![x1, x2])?;
    // // assert_eq!(val, [6.5, 7.0]);
    
    // // // Querying variables by name
    // // assert_eq!(model.get_var_by_name(&x1_name)?, Some(x1));

    Ok(())
    

    // let mut rng = StdRng::seed_from_u64(SEED);

    // // Randomly generate a network
    // let mut network = Network::rand(NUM_NEURONS,
    //     NUM_CONNECTIONS,
    //     WEIGHT_RANGE,
    //     DELAY_RANGE,
    //     TOPOLOGY, &mut rng).unwrap();

    // // Randomly generate a spike train
    // let spike_train_sampler = PeriodicSpikeTrainSampler::build(PERIOD, FIRING_RATE).unwrap();
    // let spike_trains = spike_train_sampler.sample(NUM_NEURONS, &mut rng);

    // let program = SimulationProgram::build(100.0, 1000.0, 0.1, &spike_trains).unwrap();

    // if let Err(e) = network.run(&program, &mut rng) {
    //     eprintln!("Simulation failed: {}", e);
    // }
}
