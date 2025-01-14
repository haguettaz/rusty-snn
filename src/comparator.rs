//! This module provides a comparator for spike trains.
//!
//! # Examples
//!
//! ```rust
//! use rusty_snn::simulator::comparator::Comparator;
//!
//! // Create a reference spike times
//! let firing_times_r: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
//! let comparator = Comparator::new(firing_times_r, 100.0);
//!
//! // Create a simulated spike times
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

use crate::error::SNNError;
use crate::REFRACTORY_PERIOD;

use itertools::Itertools;

const SHIFT_FACTOR: f64 = 1e6;

/// Represents a comparator for spike trains.
#[derive(Debug, PartialEq)]
pub struct Comparator {
    /// The reference (periodic) spike times.
    firing_times_r: Vec<Vec<f64>>,
    /// The period of the reference spike times.
    period: f64,
    // / The number of channels in the reference spike times.
    // num_channels: usize,
}

impl Comparator {
    /// Create a new `Comparator` instance with the given reference spike times and period.
    pub fn new(firing_times_r: Vec<Vec<f64>>, period: f64) -> Comparator {
        Comparator {
            firing_times_r,
            // num_channels,
            period,
        }
    }

    /// Calculate the precision of a spike times with respect to the reference.
    /// Returns an error if the number of channels in the spike trains to compare don't match.
    pub fn precision(&self, firing_times_h: &Vec<Vec<f64>>, end: f64) -> Result<f64, SNNError> {
        if firing_times_h.len() != self.firing_times_r.len() {
            return Err(SNNError::IncompatibleSpikeTrains);
        }

        // Process the prescribed firing times: only keep the spikes in the last period of time and remove periodic extension, if any
        let mut firing_times = vec![vec![]; firing_times_h.len()];
        for l in 0..firing_times_h.len() {
            firing_times[l] = firing_times_h[l]
                .iter()
                .filter(|&t| *t >= end - self.period - REFRACTORY_PERIOD)
                .map(|t| *t)
                .collect::<Vec<f64>>();
            if firing_times[l].len() > 1 {
                let t_start = firing_times[l].first().unwrap();
                let t_end = firing_times[l].last().unwrap();
                if mod_dist(*t_start, *t_end, self.period) < REFRACTORY_PERIOD {
                    firing_times[l].pop();
                }
            }
        }

        let shifts = self
            .firing_times_r
            .iter()
            .zip_eq(firing_times.iter())
            .flat_map(|(times_r, times)| {
                times_r
                    .iter()
                    .cartesian_product(times.iter())
                    .map(|(ref_t, t)| t - ref_t)
            })
            .unique_by(|&x| (x.rem_euclid(self.period) * SHIFT_FACTOR) as i32);

        let best_shift = shifts.max_by_key(|shift| {
            (self.precision_objective(&firing_times, *shift) * SHIFT_FACTOR) as i32
        });

        match best_shift {
            Some(best_shift) => Ok(self.precision_objective(&firing_times, best_shift)
                / self.firing_times_r.len() as f64),
            // If there are no shifts at all, count the number of channels which are pairs of empty spike firing_times
            None => Ok(self
                .firing_times_r
                .iter()
                .zip_eq(firing_times.iter())
                .filter(|(times_r, times)| times_r.is_empty() && times.is_empty())
                .count() as f64
                / self.firing_times_r.len() as f64),
        }
    }

    /// Calculate the recall of a spike times with respect to the reference.
    /// Returns an error if the number of channels in the spike trains to compare don't match.
    pub fn recall(&self, firing_times_h: &Vec<Vec<f64>>, end: f64) -> Result<f64, SNNError> {
        if firing_times_h.len() != self.firing_times_r.len() {
            return Err(SNNError::IncompatibleSpikeTrains);
        }

        // Process the prescribed firing times: only keep the spikes in the last period of time and remove periodic extension, if any
        let mut firing_times = vec![vec![]; firing_times_h.len()];
        for l in 0..firing_times_h.len() {
            firing_times[l] = firing_times_h[l]
                .iter()
                .filter(|&t| *t >= end - self.period - REFRACTORY_PERIOD)
                .map(|t| *t)
                .collect::<Vec<f64>>();
            if firing_times[l].len() > 1 {
                let t_start = firing_times[l].first().unwrap();
                let t_end = firing_times[l].last().unwrap();
                if mod_dist(*t_start, *t_end, self.period) < REFRACTORY_PERIOD {
                    firing_times[l].pop();
                }
            }
        }

        // Create an iterator over all candidate shifts
        let shifts = self
            .firing_times_r
            .iter()
            .zip_eq(firing_times.iter())
            .flat_map(|(times_r, times)| {
                times_r
                    .iter()
                    .cartesian_product(times.iter())
                    .map(|(ref_t, t)| t - ref_t)
            })
            // Reduce the shift precision before dropping the duplicated shift to improve performance
            .unique_by(|&x| (x.rem_euclid(self.period) * SHIFT_FACTOR) as i32);

        let best_shift = shifts
            .max_by_key(|shift| (self.recall_objective(&firing_times, *shift) * SHIFT_FACTOR) as i32);

        match best_shift {
            Some(best_shift) => {
                Ok(self.recall_objective(&firing_times, best_shift) / self.firing_times_r.len() as f64)
            }
            // If there is no shift at all, count the number of channels which are pairs of empty spike firing_times
            None => Ok(self
                .firing_times_r
                .iter()
                .zip_eq(firing_times.iter())
                .filter(|(times_r, times)| times_r.is_empty() && times.is_empty())
                .count() as f64
                / self.firing_times_r.len() as f64),
        }
    }

    /// Calculate the (unormalized) precision objective function of the shifted spike times with respect to the reference.
    fn precision_objective(&self, firing_times: &Vec<Vec<f64>>, shift: f64) -> f64 {
        let mut objective = 0.0;
        for (times_r, times) in self.firing_times_r.iter().zip_eq(firing_times.iter()) {
            objective += match times.is_empty() {
                true => 1.0,
                false => {
                    let tmp = times_r
                        .iter()
                        .cartesian_product(times.iter())
                        .map(|(ref_t, t)| {
                            let mod_dist = mod_dist(*ref_t, *t - shift, self.period);
                            if mod_dist < REFRACTORY_PERIOD / 2.0 {
                                1.0 - 2.0 * REFRACTORY_PERIOD * mod_dist
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    tmp / times.len() as f64
                }
            }
        }
        objective
    }

    /// Calculate the (unormalized) recall objective function of the shifted spike times with respect to the reference.    
    fn recall_objective(&self, firing_times: &Vec<Vec<f64>>, shift: f64) -> f64 {
        let mut objective = 0.0;
        for (times_r, times) in self.firing_times_r.iter().zip_eq(firing_times.iter()) {
            objective += match times_r.is_empty() {
                true => 1.0,
                false => {
                    let tmp = times_r
                        .iter()
                        .cartesian_product(times.iter())
                        .map(|(ref_t, t)| {
                            let mod_dist = mod_dist(*ref_t, *t - shift, self.period);
                            if mod_dist < REFRACTORY_PERIOD / 2.0 {
                                1.0 - 2.0 * REFRACTORY_PERIOD * mod_dist
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    tmp / times_r.len() as f64
                }
            }
        }
        objective
    }
}

/// Calculate the minimum distance between two points on a circular modulo space.
fn mod_dist(t1: f64, t2: f64, modulo: f64) -> f64 {
    let dist_1 = (t1 - t2).rem_euclid(modulo);
    let dist_2 = (t2 - t1).rem_euclid(modulo);
    dist_1.min(dist_2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_dist() {
        assert_eq!(mod_dist(0.5, 4.5, 5.0), 1.0);
        assert_eq!(mod_dist(0.0, 5.0, 5.0), 0.0);
        assert_eq!(mod_dist(25.0, 5.0, 5.0), 0.0);
        assert_eq!(mod_dist(1.0, 4.0, 5.0), 2.0);
        assert_eq!(mod_dist(4.0, 1.0, 5.0), 2.0);
    }

    #[test]
    fn test_objective_empty() {
        let firing_times_r = vec![vec![]];
        let comparator = Comparator::new(firing_times_r, 5.0);
        let firing_times = vec![vec![]];
        assert_eq!(comparator.precision_objective(&firing_times, 0.0), 1.0);
        assert_eq!(comparator.recall_objective(&firing_times, 0.0), 1.0);

        let firing_times_r = vec![vec![]];
        let comparator = Comparator::new(firing_times_r, 5.0);
        let firing_times = vec![vec![1.0, 2.25, 3.5, 4.75]];
        assert_eq!(comparator.precision_objective(&firing_times, 0.0), 0.0);
        assert_eq!(comparator.recall_objective(&firing_times, 0.0), 1.0);

        let firing_times_r = vec![vec![1.0, 2.25, 3.5, 4.75]];
        let comparator = Comparator::new(firing_times_r, 5.0);
        let firing_times = vec![vec![]];
        assert_eq!(comparator.precision_objective(&firing_times, 0.0), 1.0);
        assert_eq!(comparator.recall_objective(&firing_times, 0.0), 0.0);
    }

    #[test]
    fn test_precision_objective() {
        // single channel
        let firing_times_r = vec![vec![0.75, 2.0, 3.25, 4.5]];
        let comparator = Comparator::new(firing_times_r, 6.0);

        let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        assert_eq!(comparator.precision_objective(&firing_times, 0.0), 1.0);

        let firing_times = vec![vec![1.0, 2.25, 3.5, 4.75]];
        assert_eq!(comparator.precision_objective(&firing_times, 0.25), 1.0);

        let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        assert_eq!(comparator.precision_objective(&firing_times, 0.25), 0.5);

        let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5, 5.75]];
        assert_eq!(comparator.precision_objective(&firing_times, 0.0), 0.8);

        let firing_times = vec![vec![0.75, 2.0, 4.5]];
        assert_eq!(comparator.precision_objective(&firing_times, 0.0), 1.0);

        // multiple channels
        let firing_times_r = vec![vec![1.0, 3.25, 2.0], vec![1.5, 3.0], vec![3.5, 1.25]];
        let comparator = Comparator::new(firing_times_r, 4.0);

        let firing_times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.precision_objective(&firing_times, 0.0), 1.5);

        let firing_times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.precision_objective(&firing_times, 0.25), 3.0);
    }

    #[test]
    fn test_recall_objective() {
        // single channel
        let firing_times_r = vec![vec![0.75, 2.0, 3.25, 4.5]];
        let comparator = Comparator::new(firing_times_r, 6.0);

        let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        assert_eq!(comparator.recall_objective(&firing_times, 0.0), 1.0);

        let firing_times = vec![vec![1.0, 2.25, 3.5, 4.75]];
        assert_eq!(comparator.recall_objective(&firing_times, 0.25), 1.0);

        let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        assert_eq!(comparator.recall_objective(&firing_times, 0.25), 0.5);

        let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5, 5.75]];
        assert_eq!(comparator.recall_objective(&firing_times, 0.0), 1.0);

        let firing_times = vec![vec![0.75, 2.0, 4.5]];
        assert_eq!(comparator.recall_objective(&firing_times, 0.0), 0.75);

        // multiple channels
        let firing_times_r = vec![vec![1.0, 3.25, 2.0], vec![1.5, 3.0], vec![3.5, 1.25]];
        let comparator = Comparator::new(firing_times_r, 4.0);

        let firing_times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.recall_objective(&firing_times, 0.0), 1.5);

        let firing_times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.recall_objective(&firing_times, 0.25), 3.0);
    }

    #[test]
    fn test_precision() {
        let firing_times_r: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 10];
        let comparator = Comparator::new(firing_times_r, 100.0);
        let firing_times: Vec<Vec<f64>> =
            vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        assert_eq!(
            comparator.precision(&firing_times, 100.0),
            Err(SNNError::IncompatibleSpikeTrains)
        );

        let firing_times_r: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
        let comparator = Comparator::new(firing_times_r, 100.0);
        let firing_times: Vec<Vec<f64>> =
            vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        assert_eq!(comparator.precision(&firing_times, 100.0), Ok(1.0));
    }

    #[test]
    fn test_recall() {
        let firing_times_r: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 10];
        let comparator = Comparator::new(firing_times_r, 100.0);
        let firing_times: Vec<Vec<f64>> =
            vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        assert_eq!(
            comparator.precision(&firing_times, 100.0),
            Err(SNNError::IncompatibleSpikeTrains)
        );

        let firing_times_r: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
        let comparator = Comparator::new(firing_times_r, 100.0);
        let firing_times: Vec<Vec<f64>> =
            vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        assert_eq!(comparator.recall(&firing_times, 100.0), Ok(1.0));
    }
}
