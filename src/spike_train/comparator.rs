//! This module provides a comparator for spike trains.
//! 
//! The comparator is used to compare a spike train with a (periodic) reference.
//! 
//! # Examples
//! 
//! ```
//! use rusty_snn::spike_train::comparator::Comparator;
//! 
//! // Create a reference spike train
//! let ref_times: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
//! let comparator = Comparator::new(&ref_times, 100.0);
//!
//! // Create a simulated spike train
//! let sim_times: Vec<Vec<f64>> = vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
//!
//! // Compute precision
//! match comparator.precision(&sim_times) {
//!     Ok(precision) => println!("Precision: {}", precision),
//!     Err(e) => println!("Error: {:?}", e),
//! }
//!
//! // Compute recall
//! match comparator.recall(&sim_times) {
//!     Ok(recall) => println!("Recall: {}", recall),
//!     Err(e) => println!("Error: {:?}", e),
//! }
//! ```
use itertools::Itertools;

/// Error type for the `Comparator` struct.
#[derive(Debug, PartialEq)]
pub enum ComparatorError {
    /// Returned when the numbers of channels in the spike trains to compare don't match.
    InvalidNumChannels(String),
}

/// Represents a comparator for spike trains.
#[derive(Debug, PartialEq)]
pub struct Comparator<'a> {
    /// The reference (periodic) spike train.
    ref_times: &'a Vec<Vec<f64>>,
    /// The period of the reference spike train.
    period: f64,
    /// The number of channels in the reference spike train.
    num_channels: usize,
}

impl<'a> Comparator<'a> {

    /// Create a new `Comparator` instance with the given reference spike train and period.
    /// 
    /// # Arguments
    /// 
    /// * `ref_times` - The reference spike train.
    /// * `period` - The period of the reference spike train.
    /// 
    /// # Returns
    /// 
    /// A new `Comparator` instance.
    pub fn new(
        ref_times: &'a Vec<Vec<f64>>,
        period: f64,
    ) -> Comparator<'a> {

        Comparator {
            ref_times,
            period,
            num_channels: ref_times.len(),
        }
    }

    /// Calculate the minimum distance between two points on a circular modulo space.
    /// 
    /// # Arguments
    /// 
    /// * `x` - The first point.
    /// * `y` - The second point.
    /// * `modulo` - The modulo value representing the circular space.
    /// 
    /// # Returns
    /// 
    /// The minimum distance between `x` and `y` in the modulo space.
    fn mod_dist(x: f64, y: f64, modulo: f64) -> f64 {
        let diff1 = (x - y).rem_euclid(modulo);
        let diff2 = (y - x).rem_euclid(modulo);
        diff1.min(diff2)
    }

    /// Calculate the precision of the simulated spike train with respect to the reference spike train.
    /// 
    /// # Arguments
    /// 
    /// * `sim_times` - The simulated spike train.
    /// 
    /// # Returns
    /// 
    /// The precision of the simulated spike train with respect to the reference spike train.
    pub fn precision(&self, sim_times: &Vec<Vec<f64>>) -> Result<f64, ComparatorError> {
        if sim_times.len() != self.num_channels {
            return Err(ComparatorError::InvalidNumChannels("Impossible to align the provided spike train with the reference.".to_string()));
        }

        let shifts = self
            .ref_times
            .iter()
            .zip_eq(sim_times.iter())
            .flat_map(|(rts, sts)| {
                rts.iter()
                    .cartesian_product(sts.iter())
                    .map(|(rt, st)| st - rt)
            })
            .unique_by(|&x| (x.rem_euclid(self.period) * 1_000_000.0_f64) as i64);

        let best_shift = shifts.max_by_key(|shift| {
            (self.precision_objective(sim_times, *shift) * 1_000_000.0_f64) as i64
        });

        match best_shift {
            Some(best_shift) => {
                Ok(self.precision_objective(sim_times, best_shift) / self.num_channels as f64)
            },
            // If there are no shifts at all, count the number of channels which are pairs of empty spike trains
            None => {
                Ok(self.ref_times
                    .iter()
                    .zip_eq(sim_times.iter())
                    .filter(|(rts, sts)| rts.is_empty() && sts.is_empty())
                    .count() as f64
                    / self.num_channels as f64)
            }
        }
    }

    /// Calculate the recall of the simulated spike train with respect to the reference spike train.
    /// 
    /// # Arguments
    /// 
    /// * `sim_times` - The simulated spike train.
    /// 
    /// # Returns
    /// 
    /// The recall of the simulated spike train with respect to the reference spike train.
    pub fn recall(&self, sim_times: &Vec<Vec<f64>>) -> Result<f64, ComparatorError> {
        if sim_times.len() != self.num_channels {
            return Err(ComparatorError::InvalidNumChannels("Impossible to align the provided spike train with the reference.".to_string()));
        }

        let shifts = self
            .ref_times
            .iter()
            .zip_eq(sim_times.iter())
            .flat_map(|(rts, sts)| {
                rts.iter()
                    .cartesian_product(sts.iter())
                    .map(|(rt, st)| st - rt)
            })
            .unique_by(|&x| (x.rem_euclid(self.period) * 1_000_000.0_f64) as i64);

        let best_shift = shifts.max_by_key(|shift| {
            (self.recall_objective(sim_times, *shift) * 1_000_000.0_f64) as i64
        });

        match best_shift {
            Some(best_shift) => {
                Ok(self.recall_objective(sim_times, best_shift) / self.num_channels as f64)
            },
            // If there are no shifts at all, count the number of channels which are pairs of empty spike trains
            None => {
                Ok(self.ref_times
                    .iter()
                    .zip_eq(sim_times.iter())
                    .filter(|(rts, sts)| rts.is_empty() && sts.is_empty())
                    .count() as f64
                    / self.num_channels as f64)
            }
        }
    }

    /// Calculate the (unormalized) precision objective function between for a given shift of the simulated spike train.
    /// 
    /// # Arguments
    /// 
    /// * `sim_times` - The simulated spike train.
    /// * `shift` - The shift of the simulated spike train.
    /// 
    /// # Returns
    /// 
    /// The (unormalized) precision objective function for the given shift.
    fn precision_objective(&self, sim_times: &Vec<Vec<f64>>, shift: f64) -> f64 {
        let mut objective = 0.0;
        for (rts, sts) in self.ref_times.iter().zip_eq(sim_times.iter()) {
            objective += match sts.is_empty() {
                true => 1.0,
                false => {
                    let tmp = rts
                        .iter()
                        .cartesian_product(sts.iter())
                        .map(|(rt, st)| {
                            let mod_dist = Comparator::mod_dist(*rt, *st - shift, self.period);
                            if mod_dist < 0.5 {
                                1.0 - 2.0 * mod_dist
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    tmp / sts.len() as f64
                }
            }
        }
        objective
    }

    /// Calculate the (unormalized) recall objective function between for a given shift of the simulated spike train.
    /// 
    /// # Arguments
    /// 
    /// * `sim_times` - The simulated spike train.
    /// * `shift` - The shift of the simulated spike train.
    /// 
    /// # Returns
    /// 
    /// The (unormalized) recall objective function for the given shift.
    fn recall_objective(&self, sim_times: &Vec<Vec<f64>>, shift: f64) -> f64 {
        let mut objective = 0.0;
        for (rts, sts) in self.ref_times.iter().zip_eq(sim_times.iter()) {
            objective += match rts.is_empty() {
                true => 1.0,
                false => {
                    let tmp = rts
                        .iter()
                        .cartesian_product(sts.iter())
                        .map(|(rt, st)| {
                            let mod_dist = Comparator::mod_dist(*rt, *st - shift, self.period);
                            if mod_dist < 0.5 {
                                1.0 - 2.0 * mod_dist
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    tmp / rts.len() as f64
                }
            }
        }
        objective
    }
}

/// Tests for the `Comparator` struct.
/// 
/// # Tests
/// 
/// * Test the `mod_dist` function.
/// * Test the `precision_objective` function with empty spike trains.
/// * Test the `precision_objective` function with non-empty spike trains.
/// * Test the `recall_objective` function with empty spike trains.
/// * Test the `recall_objective` function with non-empty spike trains.
/// * Test the `precision` function with empty spike trains.
/// * Test the `precision` function with non-empty spike trains.
/// * Test the `recall` function with empty spike trains.
/// * Test the `recall` function with non-empty spike trains.
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
        let ref_times = vec![vec![]];
        let comparator = Comparator::new(&ref_times, 5.0);
        let times = vec![vec![]];
        assert_eq!(comparator.precision_objective(&times, 0.0), 1.0);
        assert_eq!(comparator.recall_objective(&times, 0.0), 1.0);

        let ref_times = vec![vec![]];
        let comparator = Comparator::new(&ref_times, 5.0);
        let times = vec![vec![1.0, 2.25, 3.5, 4.75]];
        assert_eq!(comparator.precision_objective(&times, 0.0), 0.0);
        assert_eq!(comparator.recall_objective(&times, 0.0), 1.0);

        let ref_times = vec![vec![1.0, 2.25, 3.5, 4.75]];
        let comparator = Comparator::new(&ref_times, 5.0);
        let times = vec![vec![]];
        assert_eq!(comparator.precision_objective(&times, 0.0), 1.0);
        assert_eq!(comparator.recall_objective(&times, 0.0), 0.0);
    }

    #[test]
    fn test_precision_objective() {
        // single channel
        let ref_times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        let comparator = Comparator::new(&ref_times, 6.0);

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
        let ref_times = vec![vec![1.0, 3.25, 2.0], vec![1.5, 3.0], vec![3.5, 1.25]];
        let comparator = Comparator::new(&ref_times, 4.0);

        let times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.precision_objective(&times, 0.0), 1.5);

        let times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.precision_objective(&times, 0.25), 3.0);
    }

    #[test]
    fn test_recall_objective() {
        // single channel
        let ref_times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        let comparator = Comparator::new(&ref_times, 6.0);

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
        let ref_times = vec![vec![1.0, 3.25, 2.0], vec![1.5, 3.0], vec![3.5, 1.25]];
        let comparator = Comparator::new(&ref_times, 4.0);

        let times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.recall_objective(&times, 0.0), 1.5);

        let times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
        assert_eq!(comparator.recall_objective(&times, 0.25), 3.0);
    }
    
        #[test]
        fn test_precision() {
            let ref_times:Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 10];
            let comparator = Comparator::new(&ref_times, 100.0);
            let times:Vec<Vec<f64>> = vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
            assert_eq!(comparator.precision(&times), Err(ComparatorError::InvalidNumChannels("Impossible to align the provided spike train with the reference.".to_string())));
    
            let ref_times:Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
            let comparator = Comparator::new(&ref_times, 100.0);
            let times:Vec<Vec<f64>> = vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
            assert_eq!(comparator.precision(&times), Ok(1.0));
        }

    #[test]
    fn test_recall() {
        let ref_times:Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 10];
        let comparator = Comparator::new(&ref_times, 100.0);
        let times:Vec<Vec<f64>> = vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        assert_eq!(comparator.precision(&times), Err(ComparatorError::InvalidNumChannels("Impossible to align the provided spike train with the reference.".to_string())));

        let ref_times:Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
        let comparator = Comparator::new(&ref_times, 100.0);
        let times:Vec<Vec<f64>> = vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        assert_eq!(comparator.recall(&times), Ok(1.0));
    }
}
