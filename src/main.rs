use clap::Parser;
use env_logger;
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
// use arpack_ng::{self, Error, Which};

use rusty_snn::comparator::Comparator;
use rusty_snn::error::SNNError;
use rusty_snn::jitter::jitter_spectral_radius;
use rusty_snn::network::{Network, Topology};
use rusty_snn::optim::Objective;
use rusty_snn::spike_train::{rand_spike_train, spike_trains_to_firing_times};

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser, Debug)]
struct Args {
    /// The experiment seed
    #[arg(short, long)]
    seed: u64,
    /// The number of neurons
    #[arg(short = 'L', long, default_value = "200")]
    /// The number of connections
    num_neurons: usize,
    #[arg(short = 'K', long, default_value = "100000")]
    num_connections: usize,
    /// The network topology, must be one of: fin, fout, finfout, or rand
    #[arg(long, default_value = "fin")]
    topology: String,
    /// The minimum weight
    #[arg(long, default_value = "-0.2")]
    min_weight: f64,
    /// The maximum weight
    #[arg(long, default_value = "0.2")]
    max_weight: f64,
    /// The minimum delay
    #[arg(long, default_value = "0.1")]
    min_delay: f64,
    /// The maximum delay
    #[arg(long, default_value = "10.0")]
    max_delay: f64,
    /// The args.period
    #[arg(short = 'T', long, default_value = "50.0")]
    period: f64,
    /// The firing rate
    #[arg(short = 'f', long, default_value = "0.2")]
    firing_rate: f64,
    /// The maximum level
    #[arg(long, default_value = "0.0")]
    max_level: f64,
    /// The minimum slope
    #[arg(long, default_value = "0.0")]
    min_slope: f64,
    /// The half width
    #[arg(long, default_value = "0.2")]
    half_width: f64,
    /// The objective function, must be one of: none, l2norm, l1norm
    #[arg(long, default_value = "l2norm")]
    objective: String,
}

fn main() -> Result<(), SNNError> {
    env_logger::init();

    let args = Args::parse();

    info!("{:?}", args);

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut network = Network::rand(
        args.num_neurons,
        args.num_connections,
        (args.min_weight, args.max_weight),
        (args.min_delay, args.max_delay),
        Topology::from_str(&args.topology).unwrap(),
        &mut rng,
    )
    .unwrap();
    let spike_trains =
        rand_spike_train(args.num_neurons, args.period, args.firing_rate, &mut rng).unwrap();

    network.memorize_periodic_spike_train(
        &spike_trains,
        args.period,
        (args.min_weight, args.max_weight),
        args.max_level,
        args.min_slope,
        args.half_width,
        Objective::from_str(&args.objective).unwrap(),
    )?;
    network.save_to("network.json").unwrap();
    // // let mut network = Network::load_from("network.json").unwrap();

    let phi = jitter_spectral_radius(&network.connections(), &spike_trains, args.period, &mut rng)?;
    info!("Spectral radius: {}", phi);

    let firing_times_r = spike_trains_to_firing_times(&spike_trains, args.num_neurons);
    let comparator = Comparator::new(firing_times_r, args.period);

    network.clear_all_firing_times();
    network.extend_all_firing_times_from_spike_train(&spike_trains)?;
    network.clear_all_inspikes_from_spike_train();
    network.extend_all_inspikes_from_spike_train(&spike_trains);
    for i in 0..10 {
        network.run(
            i as f64 * args.period,
            (i + 1) as f64 * args.period,
            0.02,
            &mut rng,
        )?;
        let firing_times = network.firing_times();
        let precision = comparator.precision(&firing_times, (i + 1) as f64 * args.period)?;
        let recall = comparator.recall(&firing_times, (i + 1) as f64 * args.period)?;
        info!(
            "From {} to {}: precision: {} and recall: {} (w/ 2% threshold noise)",
            i as f64 * args.period,
            (i + 1) as f64 * args.period,
            precision,
            recall
        );
    }

    network.clear_all_firing_times();
    network.extend_all_firing_times_from_spike_train(&spike_trains)?;
    network.clear_all_inspikes_from_spike_train();
    network.extend_all_inspikes_from_spike_train(&spike_trains);
    for i in 0..10 {
        network.run(
            i as f64 * args.period,
            (i + 1) as f64 * args.period,
            0.05,
            &mut rng,
        )?;
        let firing_times = network.firing_times();
        let precision = comparator.precision(&firing_times, (i + 1) as f64 * args.period)?;
        let recall = comparator.recall(&firing_times, (i + 1) as f64 * args.period)?;
        info!(
            "From {} to {}: precision: {} and recall: {} (w/ 5% threshold noise)",
            i as f64 * args.period,
            (i + 1) as f64 * args.period,
            precision,
            recall
        );
    }

    network.clear_all_firing_times();
    network.extend_all_firing_times_from_spike_train(&spike_trains)?;
    network.clear_all_inspikes_from_spike_train();
    network.extend_all_inspikes_from_spike_train(&spike_trains);
    for i in 0..10 {
        network.run(
            i as f64 * args.period,
            (i + 1) as f64 * args.period,
            0.1,
            &mut rng,
        )?;
        let firing_times = network.firing_times();
        let precision = comparator.precision(&firing_times, (i + 1) as f64 * args.period)?;
        let recall = comparator.recall(&firing_times, (i + 1) as f64 * args.period)?;
        info!(
            "From {} to {}: precision: {} and recall: {} (w/ 10% threshold noise)",
            i as f64 * args.period,
            (i + 1) as f64 * args.period,
            precision,
            recall
        );
    }

    network.clear_all_firing_times();
    network.extend_all_firing_times_from_spike_train(&spike_trains)?;
    network.clear_all_inspikes_from_spike_train();
    network.extend_all_inspikes_from_spike_train(&spike_trains);
    for i in 0..10 {
        network.run(
            i as f64 * args.period,
            (i + 1) as f64 * args.period,
            0.2,
            &mut rng,
        )?;
        let firing_times = network.firing_times();
        let precision = comparator.precision(&firing_times, (i + 1) as f64 * args.period)?;
        let recall = comparator.recall(&firing_times, (i + 1) as f64 * args.period)?;
        info!(
            "From {} to {}: precision: {} and recall: {} (w/ 20% threshold noise)",
            i as f64 * args.period,
            (i + 1) as f64 * args.period,
            precision,
            recall
        );
    }

    Ok(())
}
