//! This module provides a comparator for spike trains.
//!
//! # Examples
//!
//! ```rust
//! use rusty_snn::simulator::comparator::Comparator;
//!
//! // Create a reference spike train
//! let ref_trains: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
//! let comparator = Comparator::new(&ref_trains, 100.0);
//!
//! // Create a simulated spike train
//! let trains: Vec<Vec<f64>> = vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
//!
//! // Compute precision
//! match comparator.precision(&trains) {
//!     Ok(precision) => println!("Precision: {}", precision),
//!     Err(e) => println!("Error: {:?}", e),
//! }
//!
//! // Compute recall
//! match comparator.recall(&trains) {
//!     Ok(recall) => println!("Recall: {}", recall),
//!     Err(e) => println!("Error: {:?}", e),
//! }
//! ```


use crate::core::REFRACTORY_PERIOD;
use crate::simulator::SHIFT_FACTOR;

use itertools::Itertools;
use std::error::Error;
use std::fmt;

/// Represents a comparator for spike trains.
#[derive(Debug, PartialEq)]
pub struct Comparator<'a> {
    /// The reference (periodic) spike train.
    ref_trains: &'a Vec<Vec<f64>>,
    /// The period of the reference spike train.
    period: f64,
    /// The number of channels in the reference spike train.
    num_channels: usize,
}

impl<'a> Comparator<'a> {
    /// Create a new `Comparator` instance with the given reference spike train and period.
    pub fn new(ref_trains: &'a Vec<Vec<f64>>, period: f64) -> Comparator<'a> {
        Comparator {
            ref_trains,
            period,
            num_channels: ref_trains.len(),
        }
    }

    /// Calculate the minimum distance between two points on a circular modulo space.
    fn mod_dist(x: f64, y: f64, modulo: f64) -> f64 {
        let diff1 = (x - y).rem_euclid(modulo);
        let diff2 = (y - x).rem_euclid(modulo);
        diff1.min(diff2)
    }

    /// Calculate the precision of a spike train with respect to the reference.
    /// Returns an error if the number of channels in the spike trains to compare don't match.
    pub fn precision(&self, trains: &Vec<Vec<f64>>) -> Result<f64, ComparatorError> {
        if trains.len() != self.num_channels {
            return Err(ComparatorError::IncompatibleSpikeTrains);
        }

        let shifts = self
            .ref_trains
            .iter()
            .zip_eq(trains.iter())
            .flat_map(|(ref_train, train)| {
                ref_train
                    .iter()
                    .cartesian_product(train.iter())
                    .map(|(ref_t, t)| t - ref_t)
            })
            .unique_by(|&x| (x.rem_euclid(self.period) * SHIFT_FACTOR) as i32);

        let best_shift = shifts.max_by_key(|shift| {
            (self.precision_objective(trains, *shift) * SHIFT_FACTOR) as i32
        });

        match best_shift {
            Some(best_shift) => {
                Ok(self.precision_objective(trains, best_shift) / self.num_channels as f64)
            }
            // If there are no shifts at all, count the number of channels which are pairs of empty spike trains
            None => Ok(self
                .ref_trains
                .iter()
                .zip_eq(trains.iter())
                .filter(|(ref_train, train)| ref_train.is_empty() && train.is_empty())
                .count() as f64
                / self.num_channels as f64),
        }
    }

    /// Calculate the recall of a spike train with respect to the reference.
    /// Returns an error if the number of channels in the spike trains to compare don't match.
    pub fn recall(&self, trains: &Vec<Vec<f64>>) -> Result<f64, ComparatorError> {
        if trains.len() != self.num_channels {
            return Err(ComparatorError::IncompatibleSpikeTrains);
        }

        // Create an iterator over all candidate shifts
        let shifts = self
            .ref_trains
            .iter()
            .zip_eq(trains.iter())
            .flat_map(|(ref_train, train)| {
                ref_train
                    .iter()
                    .cartesian_product(train.iter())
                    .map(|(ref_t, t)| t - ref_t)
            })
            // Reduce the shift precision before dropping the duplicated shift to improve performance
            .unique_by(|&x| (x.rem_euclid(self.period) * SHIFT_FACTOR) as i32);

        let best_shift = shifts.max_by_key(|shift| {
            (self.recall_objective(trains, *shift) * SHIFT_FACTOR) as i32
        });

        match best_shift {
            Some(best_shift) => {
                Ok(self.recall_objective(trains, best_shift) / self.num_channels as f64)
            }
            // If there is no shift at all, count the number of channels which are pairs of empty spike trains
            None => Ok(self
                .ref_trains
                .iter()
                .zip_eq(trains.iter())
                .filter(|(ref_train, train)| ref_train.is_empty() && train.is_empty())
                .count() as f64
                / self.num_channels as f64),
        }
    }

    /// Calculate the (unormalized) precision objective function of the shifted spike train with respect to the reference.
    fn precision_objective(&self, trains: &Vec<Vec<f64>>, shift: f64) -> f64 {
        let mut objective = 0.0;
        for (ref_train, train) in self.ref_trains.iter().zip_eq(trains.iter()) {
            objective += match train.is_empty() {
                true => 1.0,
                false => {
                    let tmp = ref_train
                        .iter()
                        .cartesian_product(train.iter())
                        .map(|(ref_t, t)| {
                            let mod_dist = Comparator::mod_dist(*ref_t, *t - shift, self.period);
                            if mod_dist < REFRACTORY_PERIOD / 2.0 {
                                1.0 - 2.0 * REFRACTORY_PERIOD * mod_dist
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    tmp / train.len() as f64
                }
            }
        }
        objective
    }

    /// Calculate the (unormalized) recall objective function of the shifted spike train with respect to the reference.    
    fn recall_objective(&self, trains: &Vec<Vec<f64>>, shift: f64) -> f64 {
        let mut objective = 0.0;
        for (ref_train, train) in self.ref_trains.iter().zip_eq(trains.iter()) {
            objective += match ref_train.is_empty() {
                true => 1.0,
                false => {
                    let tmp = ref_train
                        .iter()
                        .cartesian_product(train.iter())
                        .map(|(ref_t, t)| {
                            let mod_dist = Comparator::mod_dist(*ref_t, *t - shift, self.period);
                            if mod_dist < REFRACTORY_PERIOD / 2.0 {
                                1.0 - 2.0 * REFRACTORY_PERIOD * mod_dist
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    tmp / ref_train.len() as f64
                }
            }
        }
        objective
    }
}


#[derive(Debug, PartialEq)]
pub enum ComparatorError {
    /// Error for refractory period violation, e.g., two consecutive spikes are too close.
    RefractoryPeriodViolation,
    /// Error for incompatible spike trains, e.g., different duration/period or number of channels.
    IncompatibleSpikeTrains,
}

impl fmt::Display for ComparatorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ComparatorError::RefractoryPeriodViolation => write!(f, "Violation of the refractory period"),
            ComparatorError::IncompatibleSpikeTrains => write!(f, "Incompatible spike trains"),
        }
    }
}

impl Error for ComparatorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_dist() {
        assert_eq!(Comparator::mod_dist(0.5, 4.5, 5.0), 1.0);
        assert_eq!(Comparator::mod_dist(0.0, 5.0, 5.0), 0.0);
        assert_eq!(Comparator::mod_dist(25.0, 5.0, 5.0), 0.0);
        assert_eq!(Comparator::mod_dist(1.0, 4.0, 5.0), 2.0);
        assert_eq!(Comparator::mod_dist(4.0, 1.0, 5.0), 2.0);
    }

    #[test]
    fn test_objective_empty() {
        let ref_trains = vec![vec![]];
        let comparator = Comparator::new(&ref_trains, 5.0);
        let times = vec![vec![]];
        assert_eq!(comparator.precision_objective(&times, 0.0), 1.0);
        assert_eq!(comparator.recall_objective(&times, 0.0), 1.0);

        let ref_trains = vec![vec![]];
        let comparator = Comparator::new(&ref_trains, 5.0);
        let times = vec![vec![1.0, 2.25, 3.5, 4.75]];
        assert_eq!(comparator.precision_objective(&times, 0.0), 0.0);
        assert_eq!(comparator.recall_objective(&times, 0.0), 1.0);

        let ref_trains = vec![vec![1.0, 2.25, 3.5, 4.75]];
        let comparator = Comparator::new(&ref_trains, 5.0);
        let times = vec![vec![]];
        assert_eq!(comparator.precision_objective(&times, 0.0), 1.0);
        assert_eq!(comparator.recall_objective(&times, 0.0), 0.0);
    }

    #[test]
    fn test_precision_objective() {
        // single channel
        let ref_trains = vec![vec![0.75, 2.0, 3.25, 4.5]];
        let comparator = Comparator::new(&ref_trains, 6.0);

        let times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        assert_eq!(comparator.precision_objective(&times, 0.0), 1.0);

        let times = vec![vec![1.0, 2.25, 3.5, 4.75]];
        assert_eq!(comparator.precision_objective(&times, 0.25), 1.0);

        let times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        assert_eq!(comparator.precision_objective(&times, 0.25), 0.5);

        let times = vec![vec![0.75, 2.0, 3.25, 4.5, 5.75]];
        assert_eq!(comparator.precision_objective(&times, 0.0), 0.8);

        let times = vec![vec![0.75, 2.0, 4.5]];
        assert_eq!(comparator.precision_objective(&times, 0.0), 1.0);

        // multiple channels
        let ref_trains = vec![vec![1.0, 3.25, 2.0], vec![1.5, 3.0], vec![3.5, 1.25]];
        let comparator = Comparator::new(&ref_trains, 4.0);

        let times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.precision_objective(&times, 0.0), 1.5);

        let times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.precision_objective(&times, 0.25), 3.0);
    }

    #[test]
    fn test_recall_objective() {
        // single channel
        let ref_trains = vec![vec![0.75, 2.0, 3.25, 4.5]];
        let comparator = Comparator::new(&ref_trains, 6.0);

        let times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        assert_eq!(comparator.recall_objective(&times, 0.0), 1.0);

        let times = vec![vec![1.0, 2.25, 3.5, 4.75]];
        assert_eq!(comparator.recall_objective(&times, 0.25), 1.0);

        let times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        assert_eq!(comparator.recall_objective(&times, 0.25), 0.5);

        let times = vec![vec![0.75, 2.0, 3.25, 4.5, 5.75]];
        assert_eq!(comparator.recall_objective(&times, 0.0), 1.0);

        let times = vec![vec![0.75, 2.0, 4.5]];
        assert_eq!(comparator.recall_objective(&times, 0.0), 0.75);

        // multiple channels
        let ref_trains = vec![vec![1.0, 3.25, 2.0], vec![1.5, 3.0], vec![3.5, 1.25]];
        let comparator = Comparator::new(&ref_trains, 4.0);

        let times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.recall_objective(&times, 0.0), 1.5);

        let times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.recall_objective(&times, 0.25), 3.0);
    }

    #[test]
    fn test_precision() {
        let ref_trains: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 10];
        let comparator = Comparator::new(&ref_trains, 100.0);
        let times: Vec<Vec<f64>> =
            vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        assert_eq!(
            comparator.precision(&times),
            Err(ComparatorError::IncompatibleSpikeTrains)
        );

        let ref_trains: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
        let comparator = Comparator::new(&ref_trains, 100.0);
        let times: Vec<Vec<f64>> =
            vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        assert_eq!(comparator.precision(&times), Ok(1.0));
    }

    #[test]
    fn test_recall() {
        let ref_trains: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 10];
        let comparator = Comparator::new(&ref_trains, 100.0);
        let times: Vec<Vec<f64>> =
            vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        assert_eq!(
            comparator.precision(&times),
            Err(ComparatorError::IncompatibleSpikeTrains)
        );

        let ref_trains: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
        let comparator = Comparator::new(&ref_trains, 100.0);
        let times: Vec<Vec<f64>> =
            vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        assert_eq!(comparator.recall(&times), Ok(1.0));
    }
}
