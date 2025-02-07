use clap::Parser;
use log;
use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use rand::rngs::StdRng;
use rand::SeedableRng;
use sha2::{Digest, Sha256};
use std::path::Path;

use rusty_snn::alpha::network::AlphaNetwork;
use rusty_snn::alpha::utils::AlphaLinerJitterPropagator;
use rusty_snn::core::network::Network;
use rusty_snn::core::optim::Objective;
use rusty_snn::core::spikes::{MultiChannelCyclicSpikeTrain, MultiChannelSpikeTrain};
use rusty_snn::core::utils::RealLinearOperator;
use rusty_snn::core::utils::{Comparator, TimeInterval};
use rusty_snn::core::REFRACTORY_PERIOD;
use rusty_snn::error::SNNError;

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser, Debug)]
struct Args {
    /// The experiment seed
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
    /// The objective function, must be one of: none, l2, l1
    #[arg(long, default_value = "l2")]
    objective: String,
}

fn main() -> Result<(), SNNError> {
    let args = Args::parse();

    let mut hasher = Sha256::new();
    hasher.update(format!("{:?}", args));
    let hash = hasher.finalize();

    let output_path = format!("log/{:x}.log", hash);
    let network_path = format!("network/{:x}.json", hash);

    if Path::new(&output_path).exists() {
        return Ok(());
    }

    let logfile = FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new("{l} - {m}\n")))
        .build(output_path).unwrap();

    let config = Config::builder()
        .appender(Appender::builder().build("logfile", Box::new(logfile)))
        .build(Root::builder()
                   .appender("logfile")
                   .build(LevelFilter::Info)).unwrap();

    log4rs::init_config(config).unwrap();

    // env_logger::init();

    log::info!("{:?}", args);

    let mut rng = StdRng::seed_from_u64(args.seed);
    log::info!("Network sampling...");
    let mut network = AlphaNetwork::rand_fin(
        args.num_neurons,
        args.num_inputs,
        (args.min_delay, args.max_delay),
        &mut rng,
    )?;

    log::info!("Spike train sampling...");
    let ref_spike_train = MultiChannelCyclicSpikeTrain::rand(
        args.num_neurons,
        args.period,
        args.firing_rate,
        &mut rng,
    )?;

    log::info!("Memorization...");
    network.memorize_cyclic(
        &vec![&ref_spike_train],
        (-args.lim_weight, args.lim_weight),
        args.max_level,
        args.min_slope,
        args.half_width,
        Objective::from_str(&args.objective).unwrap(),
    )?;
    log::info!("Memorization done!");

    network.save_to(&network_path)?;
    log::info!("Network saved to {}", network_path);
    // network = AlphaNetwork::load_from(&network_path).unwrap();
    // log::info!("Network loaded from {}", network_path);

    log::info!("Init jitter propagator...");
    let jitter_propagator =
        AlphaLinerJitterPropagator::new(network.connections(), &ref_spike_train);
    log::info!("Jitter propagator successfully initialized!");
    let phi = jitter_propagator.spectral_radius(&mut rng)?;
    log::info!(
        "Spectral radius: {} ({} per spikes and {} per period)",
        phi,
        phi.powf(1.0 / ref_spike_train.num_spikes() as f64),
        phi.powf(1.0 / ref_spike_train.period),
    );

    let comparator = Comparator::new(ref_spike_train.clone());

    // network.init_firing_times(&MultiChannelSpikeTrain {
    //     spike_train: ref_spike_train.spike_train.clone(),
    // });
    // network.run(&TimeInterval::new(0.0, 50.0 * args.period), 0.0)?;
    // let precision = comparator.precision(
    //     network.spike_train(),
    //     &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
    //     (49.0 * args.period - REFRACTORY_PERIOD, 49.0 * args.period + REFRACTORY_PERIOD)
    // )?;
    // let recall = comparator.recall(
    //     network.spike_train(),
    //     &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
    //     (49.0 * args.period - REFRACTORY_PERIOD, 49.0 * args.period + REFRACTORY_PERIOD)
    // )?;
    // log::info!(
    //     "After 50 cycles: precision and recall are {} and {} (w/ 0% threshold noise)",
    //     precision, recall
    // );

    // network.init_firing_times(&MultiChannelSpikeTrain {
    //     spike_train: ref_spike_train.spike_train.clone(),
    // });
    // network.run(&TimeInterval::new(0.0, 50.0 * args.period), 0.01)?;
    // let precision = comparator.precision(
    //     network.spike_train(),
    //     &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
    //     (49.0 * args.period - REFRACTORY_PERIOD, 49.0 * args.period + REFRACTORY_PERIOD)
    // )?;
    // let recall = comparator.recall(
    //     network.spike_train(),
    //     &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
    //     (49.0 * args.period - 10.0*REFRACTORY_PERIOD, 49.0 * args.period + 10.0*REFRACTORY_PERIOD)
    // )?;
    // log::info!(
    //     "After 50 cycles: precision and recall are {} and {} (w/ 1% threshold noise)",
    //     precision, recall
    // );

    network.init_firing_times(&MultiChannelSpikeTrain {
        spike_train: ref_spike_train.spike_train.clone(),
    });
    network.run(&TimeInterval::new(0.0, 50.0 * args.period), 0.02)?;
    let precision = comparator.precision(
        network.spike_train(),
        &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
        (
            49.0 * args.period - 10.0 * REFRACTORY_PERIOD,
            49.0 * args.period + 10.0 * REFRACTORY_PERIOD,
        ),
    )?;
    let recall = comparator.recall(
        network.spike_train(),
        &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
        (
            49.0 * args.period - 10.0 * REFRACTORY_PERIOD,
            49.0 * args.period + 10.0 * REFRACTORY_PERIOD,
        ),
    )?;
    log::info!(
        "After 50 cycles: precision and recall are {} and {} (w/ 2% threshold noise)",
        precision, recall
    );

    network.init_firing_times(&MultiChannelSpikeTrain {
        spike_train: ref_spike_train.spike_train.clone(),
    });
    network.run(&TimeInterval::new(0.0, 50.0 * args.period), 0.05)?;
    let precision = comparator.precision(
        network.spike_train(),
        &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
        (49.0 * args.period - REFRACTORY_PERIOD, 49.0 * args.period + REFRACTORY_PERIOD)
    )?;
    let recall = comparator.recall(
        network.spike_train(),
        &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
        (49.0 * args.period - 10.0*REFRACTORY_PERIOD, 49.0 * args.period + 10.0*REFRACTORY_PERIOD)
    )?;
    log::info!(
        "After 50 cycles: precision and recall are {} and {} (w/ 5% threshold noise)",
        precision, recall
    );

    network.init_firing_times(&MultiChannelSpikeTrain {
        spike_train: ref_spike_train.spike_train.clone(),
    });
    network.run(&TimeInterval::new(0.0, 50.0 * args.period), 0.1)?;
    let precision = comparator.precision(
        network.spike_train(),
        &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
        (49.0 * args.period - 10.0*REFRACTORY_PERIOD, 49.0 * args.period + 10.0*REFRACTORY_PERIOD)
    )?;
    let recall = comparator.recall(
        network.spike_train(),
        &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
        (49.0 * args.period - 10.0*REFRACTORY_PERIOD, 49.0 * args.period + 10.0*REFRACTORY_PERIOD)
    )?;
    log::info!(
        "After 50 cycles: precision and recall are {} and {} (w/ 10% threshold noise)",
        precision, recall
    );

    network.init_firing_times(&MultiChannelSpikeTrain {
        spike_train: ref_spike_train.spike_train.clone(),
    });
    network.run(&TimeInterval::new(0.0, 50.0 * args.period), 0.2)?;
    let precision = comparator.precision(
        network.spike_train(),
        &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
        (49.0 * args.period - 10.0*REFRACTORY_PERIOD, 49.0 * args.period + 10.0*REFRACTORY_PERIOD)
    )?;
    let recall = comparator.recall(
        network.spike_train(),
        &TimeInterval::new(49.0 * args.period - REFRACTORY_PERIOD, 50.0 * args.period),
        (49.0 * args.period - 10.0*REFRACTORY_PERIOD, 49.0 * args.period + 10.0*REFRACTORY_PERIOD)
    )?;
    log::info!(
        "After 50 cycles: precision and recall are {} and {} (w/ 20% threshold noise)",
        precision, recall
    );

    Ok(())
}
