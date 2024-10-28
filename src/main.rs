use rusty_snn::spike_train::spike_train::{PeriodicSpikeTrain, SpikeTrain};

use rand::distributions::WeightedIndex;
use rand::prelude::*; // for thread_rng()

// use rusty_snn::spike_train::utils::prob_num_spikes;

fn main() {
    let spike_train = SpikeTrain::build(vec![vec![0.0, 1.0, 2.0], vec![10.5, 2.5, 1.5]], 0.0, 3.0);

    // let mut rng = StdRng::seed_from_u64(42);

    // let spike_train = PeriodicSpikeTrain::new_random(10.0, 1000.0, 10, &mut rng).unwrap();
    // println!("{:?}", spike_train.firing_times(0));
    // println!("{:?}", spike_train.firing_times(1));
    // println!("{:#?}", spike_train);

}
