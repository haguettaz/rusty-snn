use clap::Parser;
use log;
use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use sha2::{Digest, Sha256};
use std::path::Path;

use rusty_snn::alpha::metrics::AlphaLinearJitterPropagator;
use rusty_snn::alpha::network::AlphaNetwork;
use rusty_snn::core::metrics::{RealLinearOperator, Similarity};
use rusty_snn::core::network::Network;
use rusty_snn::core::optim::Objective;
use rusty_snn::core::spike;
use rusty_snn::core::utils::TimeInterval;
use rusty_snn::error::SNNError;

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser, Debug)]
struct Args {
    /// The seed used for network sampling, spike train sampling and memorization
    #[arg(long)]
    seed: u64,
    /// The number of neurons
    #[arg(short = 'L', long, default_value = "200")]
    num_neurons: usize,
    /// The number of connections
    #[arg(short = 'K', long, default_value = "500")]
    num_inputs: usize,
    /// The maximum weight magnitude
    #[arg(long, default_value = "0.2")]
    lim_weight: f64,
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
    #[arg(long, default_value = "0.5")]
    firing_rate: f64,
    /// The maximum level
    #[arg(long, default_value = "0.0")]
    max_level: f64,
    /// The minimum slope
    #[arg(long, default_value = "2.0")]
    min_slope: f64,
    /// The half width
    #[arg(long, default_value = "0.2")]
    half_width: f64,
    /// The objective function, must be one of: none, l2 (or linf), l1 (or l0???)
    #[arg(long, default_value = "l2")]
    objective: String,
    /// The seed used for network simulation
    #[arg(long, default_value = "0")]
    sim_seed: u64,
    /// The number of simulation cycles
    #[arg(long, default_value = "50")]
    num_cycles: usize,
    /// The threshold noise level
    #[arg(long, default_value = "0.1")]
    std_threshold: f64,
}

fn main() -> Result<(), SNNError> {
    let args = Args::parse();

    let mut hasher = Sha256::new();
    hasher.update(format!("{:?}", args));
    let hash = hasher.finalize();
    let output_path = format!("log/{:x}.log", hash);

    if Path::new(&output_path).exists() {
        log::info!("Logfile already exists at {}", output_path);
        return Ok(());
    }

    let logfile = FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new("{l} - {m}\n")))
        .build(output_path)
        .unwrap();

    let config = Config::builder()
        .appender(Appender::builder().build("logfile", Box::new(logfile)))
        .build(Root::builder().appender("logfile").build(LevelFilter::Info))
        .unwrap();

    log4rs::init_config(config).unwrap();

    log::info!("{:?}", args);

    let mut hasher = Sha256::new();
    hasher.update(format!(
        "{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}{:?}",
        args.seed,
        args.num_neurons,
        args.num_inputs,
        args.lim_weight,
        (args.min_delay, args.max_delay),
        args.period,
        args.firing_rate,
        args.max_level,
        args.min_slope,
        args.half_width,
        args.objective
    ));
    let hash = hasher.finalize();
    let network_path = format!("network/{:x}.json", hash);

    let rftimes = spike::rand_cyclic(args.num_neurons, args.period, args.firing_rate, args.seed)?;
    log::info!("Spike train sampling done!");

    let mut network;
    match Path::new(&network_path).exists() {
        true => {
            network = AlphaNetwork::load_from(&network_path)?;
            log::info!("Network loading: done! Loaded from {}", network_path);
        }
        false => {
            network = AlphaNetwork::rand_fin(
                args.num_neurons,
                args.num_inputs,
                (args.min_delay, args.max_delay),
                args.seed,
            )?;
            log::info!("Network sampling: done!");

            network.memorize_cyclic(
                &vec![&rftimes],
                &vec![args.period],
                (-args.lim_weight, args.lim_weight),
                args.max_level,
                args.min_slope,
                args.half_width,
                Objective::from_str(&args.objective).unwrap(),
            )?;
            log::info!("Network optimization: done!");

            network.save_to(&network_path)?;
            log::info!("Network saving: done! Saved to {}", network_path);
        }
    };

    let jitter_propagator =
        AlphaLinearJitterPropagator::new(network.connections_ref(), &rftimes, args.period);
    let phi = jitter_propagator.spectral_radius(args.seed)?;
    log::info!(
        "Spectral radius: done! Value: {} (for {} spikes in total)",
        phi,
        jitter_propagator.dim()
    );

    let comparator = Similarity::new(rftimes.clone(), args.period)?;
    let channels: Vec<usize> = (0..args.num_neurons).collect();

    network.init_ftimes(&rftimes);
    network.run(
        &TimeInterval::new(0.0, (args.num_cycles + 1) as f64 * args.period),
        args.std_threshold,
        args.sim_seed,
    )?;
    for i in 0..=args.num_cycles {
        let (precision, recall) =
            comparator.measure(&network, &channels[..], i as f64 * args.period)?;
        log::info!(
            "Cycle {}: precision is {:.3} (with lag {:.3}) and recall is {:.3} (with lag {:.3})",
            i,
            precision.value,
            precision.time,
            recall.value,
            recall.time,
        );
    }
    Ok(())
}