//! Utility functions and types.
use core::f64;
use faer::Mat;
use itertools::Itertools;
use log;
use rand::Rng;
use std::cmp::Ordering;

use crate::core::spikes::{MultiChannelCyclicSpikeTrain, MultiChannelSpikeTrain};
use crate::core::REFRACTORY_PERIOD;
use crate::error::SNNError;

/// Tolerance for computing the precision and recall of a spike train with respect to a reference.
const SHIFT_TOL: f64 = 1e-6;
/// Relative tolerance for two floating point numbers to be considered equal.
const REL_TOL: f64 = 1e-5;
// /// TODO: Used to compute the precision and recall more efficiently...
// const SHIFT_FACTOR: f64 = 1e6;

/// A time-value pair to represent the value of a function at a given time.
#[derive(Debug, Clone)]
pub struct TimeValuePair<T: PartialOrd> {
    pub time: f64,
    pub value: T,
}

impl<T: PartialOrd> PartialEq for TimeValuePair<T> {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.value == other.value
    }
}

impl<T: PartialOrd> PartialOrd for TimeValuePair<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.value.partial_cmp(&other.value) {
            Some(Ordering::Equal) => self.time.partial_cmp(&other.time),
            other => other,
        }
    }
}

/// An open time interval.
#[derive(PartialEq, Debug, Clone)]
pub enum TimeInterval {
    /// A closed interval [start, end].
    Closed { start: f64, end: f64 },
    /// An empty time interval.
    Empty,
}

impl TimeInterval {
    pub fn new(start: f64, end: f64) -> Self {
        if start >= end {
            TimeInterval::Empty
        } else {
            TimeInterval::Closed { start, end }
        }
    }

    pub fn start(&self) -> Option<f64> {
        match self {
            TimeInterval::Closed { start, end: _ } => Some(*start),
            TimeInterval::Empty => None,
        }
    }

    pub fn end(&self) -> Option<f64> {
        match self {
            TimeInterval::Closed { start: _, end } => Some(*end),
            TimeInterval::Empty => None,
        }
    }

    pub fn contains(&self, time: f64) -> bool {
        match self {
            TimeInterval::Closed { start, end } => time >= *start && time <= *end,
            TimeInterval::Empty => false,
        }
    }

    pub fn length(&self) -> f64 {
        match self {
            TimeInterval::Closed { start, end } => end - start,
            TimeInterval::Empty => 0.0,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            TimeInterval::Empty => true,
            _ => false,
        }
    }

    pub fn intersect(&self, interval: TimeInterval) -> Self {
        match (self, interval) {
            (TimeInterval::Empty, _) => TimeInterval::Empty,
            (_, TimeInterval::Empty) => TimeInterval::Empty,
            (
                TimeInterval::Closed {
                    start: start1,
                    end: end1,
                },
                TimeInterval::Closed {
                    start: start2,
                    end: end2,
                },
            ) => TimeInterval::new(start1.max(start2), end1.min(end2)),
        }
    }
}

/// A union of open time intervals.
#[derive(PartialEq, Debug, Clone)]
pub enum TimeIntervalUnion {
    /// A union of right half-open intervals [start, end).
    ClosedUnion(Vec<TimeInterval>),
    /// An empty time interval.
    Empty,
}

impl TimeIntervalUnion {
    pub fn new_from(mut intervals: Vec<TimeInterval>) -> Self {
        intervals.sort_by(|interval_1, interval_2| match (interval_1, interval_2) {
            (
                TimeInterval::Closed {
                    start: start_0,
                    end: end_0,
                },
                TimeInterval::Closed {
                    start: start_1,
                    end: end_1,
                },
            ) => {
                if start_0 < start_1 {
                    Ordering::Less
                } else if start_0 > start_1 {
                    Ordering::Greater
                } else if end_0 < end_1 {
                    Ordering::Less
                } else if end_0 > end_1 {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }
            (TimeInterval::Closed { .. }, TimeInterval::Empty) => Ordering::Greater,
            (TimeInterval::Empty, TimeInterval::Closed { .. }) => Ordering::Less,
            (TimeInterval::Empty, TimeInterval::Empty) => Ordering::Equal,
        });

        let mut union_intervals = vec![];

        for interval in intervals {
            if let TimeInterval::Closed { start, end } = interval {
                if union_intervals.is_empty() {
                    union_intervals.push(interval);
                } else {
                    let last_interval = union_intervals.last().unwrap().clone();
                    if last_interval.contains(start) {
                        union_intervals.pop();
                        union_intervals.push(TimeInterval::new(
                            last_interval.start().unwrap(),
                            end.max(last_interval.end().unwrap()),
                        ));
                    } else {
                        union_intervals.push(interval);
                    }
                }
            }
        }

        if union_intervals.is_empty() {
            TimeIntervalUnion::Empty
        } else {
            TimeIntervalUnion::ClosedUnion(union_intervals)
        }
    }

    pub fn contains(&self, time: f64) -> bool {
        match self {
            TimeIntervalUnion::ClosedUnion(time_intervals) => time_intervals
                .iter()
                .any(|interval| interval.contains(time)),
            TimeIntervalUnion::Empty => false,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            TimeIntervalUnion::Empty => true,
            _ => false,
        }
    }

    pub fn iter(&self) -> std::slice::Iter<TimeInterval> {
        match self {
            TimeIntervalUnion::ClosedUnion(time_intervals) => time_intervals.iter(),
            TimeIntervalUnion::Empty => [].iter(),
        }
    }
}

/// Returns a random vector on the unit sphere.
pub fn rand_unit<R: Rng>(dim: usize, rng: &mut R) -> Vec<f64> {
    let mut z = (0..dim).map(|_| rng.random::<f64>()).collect::<Vec<f64>>();
    let z_norm = l2_norm(&z);
    mult_scalar_in(&mut z, 1.0 / z_norm);
    z
}

/// Subtract the vector y from the vector x (in-place).
pub fn sub_in(x: &mut [f64], y: &[f64]) {
    x.iter_mut().zip(y.iter()).for_each(|(xi, yi)| *xi -= yi);
}

/// Multiply the vector z by the scalar a (in-place).
pub fn mult_scalar_in(z: &mut [f64], a: f64) {
    z.iter_mut().for_each(|zi| *zi *= a);
}

/// Subtract the scalar a from the vector z (in-place).
pub fn sub_scalar_in(z: &mut [f64], a: f64) {
    z.iter_mut().for_each(|zi| *zi -= a);
}

/// Returns the mean of the vector z.
pub fn mean(z: &Vec<f64>) -> f64 {
    z.iter().sum::<f64>() / z.len() as f64
}

/// Returns the norm of the vector z.
pub fn l2_norm(z: &[f64]) -> f64 {
    z.iter().map(|zi| zi * zi).sum::<f64>().sqrt()
}

/// Returns the inner product of x and y.
pub fn inner(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(x_i, y_i)| x_i * y_i)
        .sum::<f64>()
}

/// Orthogonalization of y from x, inplace.
fn sub_mult_in(x: &mut [f64], y: &[f64], a: f64) {
    x.iter_mut()
        .zip(y.iter())
        .for_each(|(xi, yi)| *xi -= a * yi);
}

/// A trait for real linear operators.
/// In future version, this trait will be replaced by an adapted version of the [`faer` crate](https://faer-rs.github.io).
pub trait RealLinearOperator {
    /// Returns the dimension of the linear operator.
    fn dim(&self) -> usize;

    /// Apply the linear operator to the vector z (in-place).
    fn apply_in(&self, z: &mut Vec<f64>);

    /// Returns the spectral radius, i.e., the largest eigenvalue magnitude, by computing a compression via Arnoldi iterations.
    fn spectral_radius<R: Rng>(&self, rng: &mut R) -> Result<f64, SNNError> {
        // Init the previous spectral radius with NaN
        let mut spectral_radius = f64::NAN;
        let mut prev_spectral_radius = f64::NAN;

        // Init the vector z randomly on the unit sphere
        // Note: the algorithm might fail if the initial vector has zero components in the dominant eigenspace
        let mut z0: Vec<f64> = rand_unit(self.dim(), rng);

        // Init the Hessenberg matrix and the Arnoldi vectors
        let mut hessenberg_mat: Mat<f64> = Mat::zeros(self.dim(), self.dim());
        let mut q_vecs: Vec<Vec<f64>> = vec![];
        q_vecs.push(z0.clone());

        for k in 0..self.dim() {
            self.apply_in(&mut z0);
            for j in 0..=k {
                hessenberg_mat[(j, k)] = inner(&q_vecs[j], &z0);
                sub_mult_in(&mut z0, &q_vecs[j], hessenberg_mat[(j, k)]);
            }

            // Try to compute the eigenvalues of the current compression
            let eigvals = hessenberg_mat
                .submatrix(0, 0, k + 1, k + 1)
                .eigenvalues_from_real()
                .map_err(|e| {
                    SNNError::ConvergenceError(format!(
                        "Failed to compute the eigenvalues of the Hessenberg matrix: {:?}",
                        e
                    ))
                })?;
            spectral_radius = eigvals
                .iter()
                .map(|x| x.norm_sqr())
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap()
                .sqrt();
            log::trace!("Iter {}: spectral radius is {}", k, spectral_radius);

            if (spectral_radius - prev_spectral_radius).abs()
                / spectral_radius.max(prev_spectral_radius)
                <= REL_TOL
            {
                log::info!(
                    "Spectral radius determined! Value: {}. Number of Arnoldi iterations: {}",
                    spectral_radius, k
                );
                break;
            }
            prev_spectral_radius = spectral_radius;

            if k < self.dim() - 1 {
                hessenberg_mat[(k + 1, k)] = l2_norm(&z0);
                mult_scalar_in(&mut z0, 1.0 / hessenberg_mat[(k + 1, k)]);
                q_vecs.push(z0.clone());
            }
        }

        Ok(spectral_radius)
    }
}

/// A comparator to evaluate the precision and recall of a spike train with respect to a reference.
///
/// # Examples
///
/// ```rust
/// use rusty_snn::simulator::comparator::Comparator;
///
/// // Create a reference spike times
/// let firing_times_r: Vec<Vec<f64>> = vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
/// let comparator = Comparator::new(firing_times_r, 100.0);
///
/// // Create a simulated spike times
/// let trains: Vec<Vec<f64>> = vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
///
/// // Compute precision
/// match comparator.precision(&trains) {
///     Ok(precision) => println!("Precision: {}", precision),
///     Err(e) => println!("Error: {:?}", e),
/// }
///
/// // Compute recall
/// match comparator.recall(&trains) {
///     Ok(recall) => println!("Recall: {}", recall),
///     Err(e) => println!("Error: {:?}", e),
/// }
/// ```
#[derive(Debug, PartialEq)]
pub struct Comparator {
    /// The reference (periodic) spike train.
    ref_spike_train: MultiChannelCyclicSpikeTrain,
}

impl Comparator {
    /// Create a new `Comparator` instance with the given reference spike times and period.
    pub fn new(ref_spike_train: MultiChannelCyclicSpikeTrain) -> Comparator {
        Comparator { ref_spike_train }
    }

    /// Calculate the minimum distance between two points on a circular modulo space.
    fn dist(&self, t1: f64, t2: f64) -> f64 {
        let dist_1 = (t1 - t2).rem_euclid(self.period());
        let dist_2 = (t2 - t1).rem_euclid(self.period());
        dist_1.min(dist_2)
    }

    /// Returns the period of the reference cyclic spike train.
    fn num_channels(&self) -> usize {
        self.ref_spike_train.num_channels()
    }

    /// Returns the period of the reference cyclic spike train.
    fn period(&self) -> f64 {
        self.ref_spike_train.period
    }

    /// Calculate the precision of a spike times with respect to the reference.
    /// Returns an error if the number of channels in the spike trains to compare don't match.
    pub fn precision(
        &self,
        mut spike_train: MultiChannelSpikeTrain,
        window: &TimeInterval,
        shift_lim: (f64, f64),
    ) -> Result<f64, SNNError> {
        if spike_train.num_channels() != self.num_channels() {
            return Err(SNNError::IncompatibleSpikeTrains);
        }

        // Process the prescribed firing times: only keep the spikes in the last period of time and remove periodic extension, if any
        // let mut firing_times = vec![vec![]; firing_times_h.len()];
        spike_train.iter_mut().for_each(|spike_train| {
            spike_train.retain(|time| window.contains(*time));
            if let Some((last, first)) = spike_train.iter().circular_tuple_windows().last() {
                if self.dist(*first, *last) < REFRACTORY_PERIOD {
                    spike_train.pop();
                }
            }
        });

        let shifts = self
            .ref_spike_train
            .iter()
            .zip_eq(spike_train.iter())
            .flat_map(|(ref_spikes, spikes)| {
                ref_spikes
                    .iter()
                    .cartesian_product(spikes.iter())
                    .map(|(ref_time, time)| time - ref_time)
            })
            .filter(|&x| (x < shift_lim.1) && (x > shift_lim.0))
            .unique_by(|&x| (x / SHIFT_TOL).round() as i64);
        // .unique_by(|&x| (x.rem_euclid(self.period()) * SHIFT_FACTOR) as i32);

        // let best_shift = shifts.max_by(|shift_1, shift_2| {
        //     self.precision_objective(&spike_train, *shift_1)
        //         .partial_cmp(&self.precision_objective(&spike_train, *shift_2))
        //         .unwrap()
        // });
        // (self.precision_objective(&spike_train, *shift) * SHIFT_FACTOR) as i32

        match shifts.max_by(|shift_1, shift_2| {
            self.precision_objective(&spike_train, *shift_1)
                .partial_cmp(&self.precision_objective(&spike_train, *shift_2))
                .unwrap()
        }) {
            Some(best_shift) => {
                Ok(self.precision_objective(&spike_train, best_shift) / self.num_channels() as f64)
            }
            // If there are no shifts at all, count the number of channels which are pairs of empty spike firing_times
            None => Ok(self
                .ref_spike_train
                .iter()
                .zip_eq(spike_train.iter())
                .filter(|(ref_spikes, spikes)| ref_spikes.is_empty() && spikes.is_empty())
                .count() as f64
                / self.num_channels() as f64),
        }
    }

    /// Calculate the recall of a spike times with respect to the reference.
    /// Returns an error if the number of channels in the spike trains to compare don't match.
    pub fn recall(
        &self,
        mut spike_train: MultiChannelSpikeTrain,
        window: &TimeInterval,
        shift_lim: (f64,f64),
    ) -> Result<f64, SNNError> {
        if spike_train.num_channels() != self.num_channels() {
            return Err(SNNError::IncompatibleSpikeTrains);
        }

        // Process the prescribed firing times: only keep the spikes in the last period of time and remove periodic extension, if any
        // let mut firing_times = vec![vec![]; firing_times_h.len()];
        spike_train.iter_mut().for_each(|spike_train| {
            spike_train.retain(|time| window.contains(*time));
            if let Some((last, first)) = spike_train.iter().circular_tuple_windows().last() {
                if self.dist(*first, *last) < REFRACTORY_PERIOD {
                    spike_train.pop();
                }
            }
        });

        let shifts = self
            .ref_spike_train
            .iter()
            .zip_eq(spike_train.iter())
            .flat_map(|(ref_spikes, spikes)| {
                ref_spikes
                    .iter()
                    .cartesian_product(spikes.iter())
                    .map(|(ref_time, time)| time - ref_time)
            })
            .filter(|&x| (x < shift_lim.1) && (x > shift_lim.0))
            .unique_by(|&x| (x / SHIFT_TOL).round() as i64);
        // .unique_by(|&x| (x.rem_euclid(self.period()) * SHIFT_FACTOR) as i32);

        // let best_shift = shifts.max_by_key(|shift| {
        //     (self.recall_objective(&spike_train, *shift) * SHIFT_FACTOR) as i32
        // });

        match shifts.max_by(|shift_1, shift_2| {
            self.recall_objective(&spike_train, *shift_1)
                .partial_cmp(&self.recall_objective(&spike_train, *shift_2))
                .unwrap()
        }) {
            Some(best_shift) => {
                Ok(self.recall_objective(&spike_train, best_shift) / self.num_channels() as f64)
            }
            // If there are no shifts at all, count the number of channels which are pairs of empty spike firing_times
            None => Ok(self
                .ref_spike_train
                .iter()
                .zip_eq(spike_train.iter())
                .filter(|(ref_spikes, spikes)| ref_spikes.is_empty() && spikes.is_empty())
                .count() as f64
                / self.num_channels() as f64),
        }

        // match best_shift {
        //     Some(best_shift) => {
        //         Ok(self.precision_objective(&spike_train, best_shift) / self.num_channels() as f64)
        //     }
        //     // If there are no shifts at all, count the number of channels which are pairs of empty spike firing_times
        //     None => Ok(self
        //         .ref_spike_train
        //         .iter()
        //         .zip_eq(spike_train.iter())
        //         .filter(|(ref_spikes, spikes)| ref_spikes.is_empty() && spikes.is_empty())
        //         .count() as f64
        //         / self.num_channels() as f64),
        // }
    }

    /// Calculate the (unormalized) precision objective function of the shifted spike times with respect to the reference.
    fn precision_objective(&self, spike_train: &MultiChannelSpikeTrain, shift: f64) -> f64 {
        self.ref_spike_train.iter().zip_eq(spike_train.iter()).fold(
            0.0,
            |acc, (ref_spikes, spikes)| {
                acc + match spikes.is_empty() {
                    true => 1.0,
                    false => {
                        ref_spikes
                            .iter()
                            .cartesian_product(spikes.iter())
                            .map(|(ref_time, time)| self.dist(*ref_time, *time - shift))
                            .filter(|&dist| dist < REFRACTORY_PERIOD / 2.0)
                            .map(|dist| 1.0 - 2.0 * REFRACTORY_PERIOD * dist)
                            .sum::<f64>()
                            / spikes.len() as f64
                    }
                }
            },
        )
    }

    /// Calculate the (unormalized) recall objective function of the shifted spike times with respect to the reference.    
    fn recall_objective(&self, spike_train: &MultiChannelSpikeTrain, shift: f64) -> f64 {
        self.ref_spike_train.iter().zip_eq(spike_train.iter()).fold(
            0.0,
            |acc, (ref_spikes, spikes)| {
                acc + match ref_spikes.is_empty() {
                    true => 1.0,
                    false => {
                        ref_spikes
                            .iter()
                            .cartesian_product(spikes.iter())
                            .map(|(ref_time, time)| self.dist(*ref_time, *time - shift))
                            .filter(|&dist| dist < REFRACTORY_PERIOD / 2.0)
                            .map(|dist| 1.0 - 2.0 * REFRACTORY_PERIOD * dist)
                            .sum::<f64>()
                            / ref_spikes.len() as f64
                    }
                }
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_contains() {
        let interval = TimeInterval::Closed {
            start: 0.0,
            end: 10.0,
        };
        assert_eq!(interval.contains(-1.0), false);
        assert_eq!(interval.contains(0.0), true);
        assert_eq!(interval.contains(5.0), true);
        assert_eq!(interval.contains(10.0), true);
        assert_eq!(interval.contains(12.0), false);

        let intervals = TimeIntervalUnion::new_from(vec![
            TimeInterval::Closed {
                start: 0.0,
                end: 0.5,
            },
            TimeInterval::Closed {
                start: 3.0,
                end: 5.0,
            },
        ]);
        assert_eq!(intervals.contains(-1.0), false);
        assert_eq!(intervals.contains(0.0), true);
        assert_eq!(intervals.contains(2.0), false);
        assert_eq!(intervals.contains(5.0), true);
        assert_eq!(intervals.contains(10.0), false);
        assert_eq!(intervals.contains(12.0), false);
    }

    #[test]
    fn test_comparator_dist() {
        let ref_spike_train = MultiChannelCyclicSpikeTrain {
            spike_train: vec![vec![]],
            period: 5.0,
        };
        let comparator = Comparator::new(ref_spike_train);

        assert_eq!(comparator.dist(0.5, 4.5), 1.0);
        assert_eq!(comparator.dist(0.0, 5.0), 0.0);
        assert_eq!(comparator.dist(25.0, 5.0), 0.0);
        assert_eq!(comparator.dist(1.0, 4.0), 2.0);
        assert_eq!(comparator.dist(4.0, 1.0), 2.0);
    }

    #[test]
    fn test_objective_empty() {
        let ref_spike_train = MultiChannelCyclicSpikeTrain {
            spike_train: vec![vec![]],
            period: 5.0,
        };
        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![vec![]],
        };
        let comparator = Comparator::new(ref_spike_train);
        assert_eq!(comparator.precision_objective(&spike_train, 0.0), 1.0);
        assert_eq!(comparator.recall_objective(&spike_train, 0.0), 1.0);

        let ref_spike_train = MultiChannelCyclicSpikeTrain {
            spike_train: vec![vec![]],
            period: 5.0,
        };
        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![vec![1.0, 2.25, 3.5, 4.75]],
        };
        let comparator = Comparator::new(ref_spike_train);
        assert_eq!(comparator.precision_objective(&spike_train, 0.0), 0.0);
        // assert_eq!(comparator.recall_objective(&spike_train, 0.0), f64::NAN);

        let ref_spike_train = MultiChannelCyclicSpikeTrain {
            spike_train: vec![vec![1.0, 2.25, 3.5, 4.75]],
            period: 5.0,
        };
        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![vec![]],
        };
        let comparator = Comparator::new(ref_spike_train);
        // let firing_times_r = vec![vec![1.0, 2.25, 3.5, 4.75]];
        // let comparator = Comparator::new(firing_times_r, 5.0);
        // let firing_times = vec![vec![]];
        // assert_eq!(comparator.precision_objective(&spike_train, 0.0), 1.0);
        assert_eq!(comparator.recall_objective(&spike_train, 0.0), 0.0);
    }

    #[test]
    fn test_objective() {
        // Single channel
        let ref_spike_train = MultiChannelCyclicSpikeTrain {
            spike_train: vec![vec![0.75, 2.0, 3.25, 4.5]],
            period: 6.0,
        };
        let comparator = Comparator::new(ref_spike_train);

        // let spike_train = MultiChannelSpikeTrain::new_from(vec![vec![0.75, 2.0, 3.25, 4.5]]);

        // let firing_times_r = vec![vec![0.75, 2.0, 3.25, 4.5]];
        // let comparator = Comparator::new(firing_times_r, 6.0);

        // let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![vec![0.75, 2.0, 3.25, 4.5]],
        };
        assert_eq!(comparator.precision_objective(&spike_train, 0.0), 1.0);
        assert_eq!(comparator.recall_objective(&spike_train, 0.0), 1.0);

        // let firing_times = vec![vec![1.0, 2.25, 3.5, 4.75]];
        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![vec![1.0, 2.25, 3.5, 4.75]],
        };
        assert_eq!(comparator.precision_objective(&spike_train, 0.25), 1.0);
        assert_eq!(comparator.recall_objective(&spike_train, 0.25), 1.0);

        // let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![vec![0.75, 2.0, 3.25, 4.5]],
        };
        assert_eq!(comparator.precision_objective(&spike_train, 0.25), 0.5);
        assert_eq!(comparator.recall_objective(&spike_train, 0.25), 0.5);

        // let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5, 5.75]];
        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![vec![0.75, 2.0, 3.25, 4.5, 5.75]],
        };
        assert_eq!(comparator.precision_objective(&spike_train, 0.0), 0.8);
        assert_eq!(comparator.recall_objective(&spike_train, 0.0), 1.0);

        // let firing_times = vec![vec![0.75, 2.0, 4.5]];
        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![vec![0.75, 2.0, 4.5]],
        };
        assert_eq!(comparator.precision_objective(&spike_train, 0.0), 1.0);
        assert_eq!(comparator.recall_objective(&spike_train, 0.0), 0.75);

        // Multiple channels
        let ref_spike_train = MultiChannelCyclicSpikeTrain {
            spike_train: vec![vec![1.0, 3.25, 2.0], vec![1.5, 3.0], vec![3.5, 1.25]],
            period: 4.0,
        };
        let comparator = Comparator::new(ref_spike_train);
        // let firing_times_r = vec![vec![1.0, 3.25, 2.0], vec![1.5, 3.0], vec![3.5, 1.25]];
        // let comparator = Comparator::new(firing_times_r, 4.0);

        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]],
        };
        assert_eq!(comparator.precision_objective(&spike_train, 0.0), 1.5);
        assert_eq!(comparator.recall_objective(&spike_train, 0.0), 1.5);

        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]],
        };
        assert_eq!(comparator.precision_objective(&spike_train, 0.25), 3.0);
        assert_eq!(comparator.recall_objective(&spike_train, 0.0), 1.5);
    }

    // #[test]
    // fn test_recall_objective() {
    //     // single channel
    //     let firing_times_r = vec![vec![0.75, 2.0, 3.25, 4.5]];
    //     let comparator = Comparator::new(firing_times_r, 6.0);

    //     let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5]];
    //     assert_eq!(comparator.recall_objective(&firing_times, 0.0), 1.0);

    //     let firing_times = vec![vec![1.0, 2.25, 3.5, 4.75]];
    //     assert_eq!(comparator.recall_objective(&firing_times, 0.25), 1.0);

    //     let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5]];
    //     assert_eq!(comparator.recall_objective(&firing_times, 0.25), 0.5);

    //     let firing_times = vec![vec![0.75, 2.0, 3.25, 4.5, 5.75]];
    //     assert_eq!(comparator.recall_objective(&firing_times, 0.0), 1.0);

    //     let firing_times = vec![vec![0.75, 2.0, 4.5]];
    //     assert_eq!(comparator.recall_objective(&firing_times, 0.0), 0.75);

    //     // multiple channels
    //     let firing_times_r = vec![vec![1.0, 3.25, 2.0], vec![1.5, 3.0], vec![3.5, 1.25]];
    //     let comparator = Comparator::new(firing_times_r, 4.0);

    //     let firing_times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
    //     assert_eq!(comparator.recall_objective(&firing_times, 0.0), 1.5);

    //     let firing_times = vec![vec![1.25, 3.5, 2.25], vec![1.75, 3.25], vec![3.75, 1.5]];
    //     assert_eq!(comparator.recall_objective(&firing_times, 0.25), 3.0);
    // }

    #[test]
    fn test_precision_recall() {
        let ref_spike_train = MultiChannelCyclicSpikeTrain {
            spike_train: vec![vec![0.75, 2.0, 3.25, 4.5]],
            period: 6.0,
        };
        let comparator = Comparator::new(ref_spike_train);
        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![vec![5.75, 7.0, 8.25, 9.5]],
        };
        assert_eq!(comparator.precision(spike_train.clone(), &TimeInterval::new(5.0, 11.0), (4.0, 7.0)), Ok(1.0));
        assert_eq!(comparator.recall(spike_train.clone(), &TimeInterval::new(5.0, 11.0), (4.0, 7.0)), Ok(1.0));

        let ref_spike_train = MultiChannelCyclicSpikeTrain {
            spike_train: vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 10],
            period: 100.0,
        };
        let comparator = Comparator::new(ref_spike_train);

        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 5],
        };
        assert_eq!(
            comparator.precision(spike_train.clone(), &TimeInterval::new(0.0, 100.0), (-100.0, 100.0)),
            Err(SNNError::IncompatibleSpikeTrains)
        );
        assert_eq!(
            comparator.recall(spike_train.clone(), &TimeInterval::new(0.0, 100.0), (-100.0, 100.0)),
            Err(SNNError::IncompatibleSpikeTrains)
        );

        let spike_train = MultiChannelSpikeTrain {
            spike_train: vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 10],
        };
        assert_eq!(
            comparator.precision(spike_train.clone(), &TimeInterval::new(0.0, 100.0), (-100.0, 100.0)),
            Ok(1.0)
        );
        assert_eq!(
            comparator.recall(spike_train.clone(), &TimeInterval::new(0.0, 100.0), (-100.0, 100.0)),
            Ok(1.0)
        );

        // let firing_times_r: Vec<Vec<Spike>> =
        //     vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 10];
        // let comparator = Comparator::new(firing_times_r, 100.0);
        // let firing_times: Vec<Vec<Spike>> =
        //     vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        // assert_eq!(
        //     comparator.precision(&firing_times, 100.0),
        //     Err(SNNError::IncompatibleSpikeTrains)
        // );

        // let firing_times_r: Vec<Vec<Spike>> =
        //     vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
        // let comparator = Comparator::new(firing_times_r, 100.0);
        // let firing_times: Vec<Vec<Spike>> =
        //     vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
        // assert_eq!(comparator.precision(&firing_times, 100.0), Ok(1.0));
    }

    // #[test]
    // fn test_recall() {
    //     let firing_times_r: Vec<Vec<Spike>> =
    //         vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 10];
    //     let comparator = Comparator::new(firing_times_r, 100.0);
    //     let firing_times: Vec<Vec<Spike>> =
    //         vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
    //     assert_eq!(
    //         comparator.precision(&firing_times, 100.0),
    //         Err(SNNError::IncompatibleSpikeTrains)
    //     );

    //     let firing_times_r: Vec<Vec<Spike>> =
    //         vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 50];
    //     let comparator = Comparator::new(firing_times_r, 100.0);
    //     let firing_times: Vec<Vec<Spike>> =
    //         vec![(0..50).map(|i| 1.0_f64 + 2.0_f64 * i as f64).collect(); 50];
    //     assert_eq!(comparator.recall(&firing_times, 100.0), Ok(1.0));
    // }
}
