use rand::distributions::Uniform;
use rusty_snn::snn::network::Network;
use std::ops::Range;
use std::fs;

const NUM_NEURONS: usize = 100;
const NUM_INPUTS: usize = 100;
const WEIGHT_RANGE: Range<f64> = -1.0..1.0;
const DELAY_RANGE: Range<f64> = 1.0..10.0;
const ORDER_RANGE: Range<i32> = 1..16;
const BETA_RANGE: Range<f64> = 0.5..2.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure data directory exists
    fs::create_dir_all("data")?;

    // Create a new random network
    let network = Network::new_random_fin(
        NUM_NEURONS,
        NUM_INPUTS,
        Uniform::from(WEIGHT_RANGE),
        Uniform::from(DELAY_RANGE),
        Uniform::from(ORDER_RANGE),
        Uniform::from(BETA_RANGE)
    );

    // Save network 
    network.save_to("data/network.json").map_err(|e| format!("Failed to save network: {}", e))?;

    // Load network
    let _loaded_network = Network::load_from("data/network.json").map_err(|e| format!("Failed to load network: {}", e))?;

    Ok(())
}