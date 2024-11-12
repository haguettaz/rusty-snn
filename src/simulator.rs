//! Simulation framework for Spiking Neural Networks.
//! 
//! This module provides two main components:
//! - `simulator`: Provides utilities for simulation programs.
//! - `comparator`: Provides utilities for assessing simulations.
//! ```

pub mod simulator;
pub mod comparator;

/// The smallest time unit used in the simulation.
pub const TIME_RESOLUTION : f64 = 1e-12;
/// The time step used in the simulation.
pub const TIME_STEP : f64 = 1e-3;
/// The scaling factor for comparing spike trains.
pub const SHIFT_FACTOR: f64 = 1_000_000.0;