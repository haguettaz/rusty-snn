//! This module contains the simulation program and interval structures.
//!
use crate::core::spike_train::SpikeTrain;
use crate::core::error::SNNError;

// use std::cmp::Ordering;
// use std::collections::HashSet;
// use std::error::Error;
// use std::fmt;



// /// Represents a half-open time interval [start, end).
// #[derive(Debug, PartialEq)]
// pub struct TimeInterval {
//     start: f64,
//     end: f64,
// }

// /// Implement the partial ordering for time intervals.
// /// The ordering is based on the start and end times of the intervals.
// /// A time interval is less than another if it ends before the other starts, and vice versa.
// /// Overlapping intervals are not ordered.
// impl PartialOrd for TimeInterval {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         if self.end <= other.start {
//             Some(Ordering::Less)
//         } else if self.start >= other.end {
//             Some(Ordering::Greater)
//         } else {
//             None
//         }
//     }
// }

// impl TimeInterval {
//     /// Create a time interval with the specified parameters.
//     pub fn build(start: f64, end: f64) -> Result<Self, SimulationError> {
//         if !(start.is_finite() && end.is_finite()) {
//             return Err(SimulationError::HalfInfiniteTimeInterval);
//         }

//         if start >= end {
//             return Err(SimulationError::EmptyTimeInterval);
//         }

//         Ok(TimeInterval { start, end })
//     }

//     /// Returns the start time of the time interval.
//     pub fn start(&self) -> f64 {
//         self.start
//     }

//     /// Returns the end time of the time interval.
//     pub fn end(&self) -> f64 {
//         self.end
//     }
// }

/// Represents a simulation program with a time interval, neuron control, and threshold noise.
#[derive(Debug, PartialEq)]
pub struct MemorizationProgram<'a> {
    spike_trains: &'a [SpikeTrain],
    min_slope: f64,
    min_gap: f64,
    activivity_window: f64
}

impl MemorizationProgram<'_> {
    /// Create a simulation interval with the specified parameters.
    /// The function returns an error for invalid simulation intervals or control.
    pub fn build<'a>(
        spike_trains: &'a [SpikeTrain],
        min_slope: f64,
        min_gap: f64,
        activivity_window: f64
    ) -> Result<MemorizationProgram<'a>, SNNError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
