use rusty_snn::snn::network::{Network, BalancedType};
use std::ops::Range;
use std::fs;

const NUM_NEURONS: usize = 100;
const NUM_CONNECTIONS: usize = 10_000;
const WEIGHT_RANGE: Range<f64> = -1.0..1.0;
const DELAY_RANGE: Range<f64> = 1.0..10.0;
const ORDER_RANGE: Range<i32> = 1..16;
const BETA_RANGE: Range<f64> = 0.5..2.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure data directory exists
    fs::create_dir_all("data")?;

    // Create a new random network
    let network = Network::new_random(
        NUM_NEURONS,
        NUM_CONNECTIONS,
        WEIGHT_RANGE,
        DELAY_RANGE,
        ORDER_RANGE,
        BETA_RANGE,
        BalancedType::Unbalanced,
    );

    // Save network 
    network.save_to("data/network.json").map_err(|e| format!("Failed to save network: {}", e))?;

    // Load network
    let _loaded_network = Network::load_from("data/network.json").map_err(|e| format!("Failed to load network: {}", e))?;

    Ok(())
}