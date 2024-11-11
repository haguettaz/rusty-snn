//! Simulation module.
pub mod simulator;
pub mod comparator;

pub const TIME_RESOLUTION : f64 = 1e-12;
pub const TIME_STEP : f64 = 1e-3;
pub const SHIFT_FACTOR: f64 = 1_000_000.0;