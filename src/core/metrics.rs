//! Memorization metrics module.
use core::f64;
use faer::Mat;
use itertools::{izip, Itertools};
use log;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::core::network::Network;
use crate::core::REFRACTORY_PERIOD;
use crate::error::SNNError;

use super::neuron::Neuron;
use super::utils::TimeValuePair;

/// Relative tolerance for two floating point numbers to be considered equal.
const REL_TOL: f64 = 1e-5;

/// Absolute tolerance for two time shifts to be considered equal.
const TOL: f64 = 1e-12;

#[derive(Debug, PartialEq)]
pub struct Similarity {
    /// The reference (periodic) spike train.
    rftimes: Vec<Vec<f64>>,
    /// The period of the reference spike train.
    period: f64,
}

struct Shift {
    value: f64,
    scale_precision: f64,
    scale_recall: f64,
}

impl PartialEq for Shift {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl PartialOrd for Shift {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Similarity {
    /// Create a new `Similarity` instance with the given reference spike times and period.
    pub fn new(rftimes: Vec<Vec<f64>>, period: f64) -> Result<Similarity, SNNError> {
        if period <= 0.0 {
            return Err(SNNError::InvalidParameter(
                "The period must be a positive number.".to_string(),
            ));
        }
        Ok(Similarity { rftimes, period })
    }

    /// Extract network spikes, keeping only those on the provided channels and within a single period after min_time, accounting for edge cases
    fn extract_ftimes<'a>(
        &self,
        network: &'a impl Network,
        channels: &[usize],
        min_time: f64,
    ) -> Vec<&'a [f64]> {
        channels
            .iter()
            .map(|id| {
                // Extract spikes within a single period after min_time:
                // 1. Find start index using binary search, including spikes up to REFRACTORY_PERIOD before min_time
                //    to avoid missing spikes that could be part of this period
                // 2. Find end index at min_time + period
                // 3. If the first and last spikes are within REFRACTORY_PERIOD when wrapped around the period,
                //    exclude the first spike to avoid counting essentially duplicate spikes across period boundaries
                let neuron = network.neuron_ref(*id).unwrap();
                let ftimes = neuron.ftimes_ref();
                let start = match ftimes.binary_search_by(|time| {
                    time.partial_cmp(&(min_time - REFRACTORY_PERIOD)).unwrap()
                }) {
                    Ok(pos) | Err(pos) => pos,
                };
                let end = match ftimes
                    .binary_search_by(|time| time.partial_cmp(&(min_time + self.period)).unwrap())
                {
                    Ok(pos) | Err(pos) => pos,
                };

                if start < end {
                    if (ftimes[start] + self.period - ftimes[end - 1]).abs() < REFRACTORY_PERIOD {
                        &ftimes[start + 1..end]
                    } else {
                        &ftimes[start..end]
                    }
                } else {
                    &[]
                }
            })
            .collect()
    }

    /// Extract reference spikes, keeping only those on the provided channels.
    fn extract_rftimes(&self, channels: &[usize]) -> Vec<&[f64]> {
        channels
            .iter()
            .map(|id| &self.rftimes[*id] as &[f64])
            .collect()
    }

    /// Compute all time shifts between the nominal and the actual spike times.
    fn compute_all_shifts(&self, rftimes: &Vec<&[f64]>, ftimes: &Vec<&[f64]>) -> Vec<Shift> {
        let mut shifts: Vec<Shift> = izip!(rftimes, ftimes)
            .flat_map(|(rcftimes, cftimes)| {
                rcftimes
                    .iter()
                    .cartesian_product(cftimes.iter())
                    .map(|(rft, ft)| {
                        let dt: f64 = ft - rft;
                        Shift {
                            value: dt
                                - self.period * ((dt + self.period / 2.0) / self.period).floor(),
                            scale_precision: cftimes.len() as f64,
                            scale_recall: rcftimes.len() as f64,
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
            })
            .collect();
        shifts.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        shifts
    }

    /// Compute the precision at the given time shift.
    fn precision(&self, all_shifts: &Vec<Shift>, shift: f64) -> f64 {
        let pos = match all_shifts.binary_search_by(|other| {
            other
                .value
                .partial_cmp(&(shift - REFRACTORY_PERIOD / 2.0))
                .unwrap()
        }) {
            Ok(pos) => all_shifts[..=pos]
                .iter()
                .rev()
                .enumerate()
                .take_while(|(_, other)| other.value > shift - REFRACTORY_PERIOD / 2.0)
                .last()
                .map(|(i, _)| pos - i)
                .unwrap_or_default(),
            Err(pos) => pos,
        };

        all_shifts[pos..]
            .iter()
            .take_while(|other| other.value < shift + REFRACTORY_PERIOD / 2.0)
            .map(|other| {
                (1.0 - 2.0 * (shift - other.value).abs() / REFRACTORY_PERIOD)
                    / other.scale_precision as f64
            })
            .chain(
                all_shifts
                    .iter()
                    .rev()
                    .take_while(|other| other.value > shift + self.period - REFRACTORY_PERIOD / 2.0)
                    .map(|other| {
                        (1.0 - 2.0 * (shift + self.period - other.value).abs() / REFRACTORY_PERIOD)
                            / other.scale_precision
                    }),
            )
            .sum()
    }

    /// Compute the recall at the given time shift.
    fn recall(&self, all_shifts: &Vec<Shift>, shift: f64) -> f64 {
        let pos = match all_shifts.binary_search_by(|other| {
            other
                .value
                .partial_cmp(&(shift - REFRACTORY_PERIOD / 2.0))
                .unwrap()
        }) {
            Ok(pos) => all_shifts[..=pos]
                .iter()
                .rev()
                .enumerate()
                .take_while(|(_, other)| other.value > shift - REFRACTORY_PERIOD / 2.0)
                .last()
                .map(|(i, _)| pos - i)
                .unwrap_or_default(),
            Err(pos) => pos,
        };

        all_shifts[pos..]
            .iter()
            .take_while(|other| other.value < shift + REFRACTORY_PERIOD / 2.0)
            .map(|other| {
                (1.0 - 2.0 * (shift - other.value).abs() / REFRACTORY_PERIOD)
                    / other.scale_recall as f64
            })
            .chain(
                all_shifts
                    .iter()
                    .rev()
                    .take_while(|other| other.value > shift + self.period - REFRACTORY_PERIOD / 2.0)
                    .map(|other| {
                        (1.0 - 2.0 * (shift + self.period - other.value).abs() / REFRACTORY_PERIOD)
                            / other.scale_recall
                    }),
            )
            .sum()
    }

    /// Calculate the precision and recall of a multi-channel spike train with respect to the reference.
    /// Returns an error if the number of channels in the spike trains to compare don't match.
    pub fn measure(
        &self,
        // times: &Vec<&[f64]>,
        network: &impl Network,
        channels: &[usize],
        min_time: f64,
    ) -> Result<(TimeValuePair<f64>, TimeValuePair<f64>), SNNError> {
        if channels
            .iter()
            .any(|id| network.neuron_ref(*id).is_none() || self.rftimes.get(*id).is_none())
        {
            return Err(SNNError::InvalidParameter(
                "The channel IDs must be valid indices of the spike trains.".to_string(),
            ));
        }

        let num_channels = channels.iter().count() as f64;

        let rftimes = self.extract_rftimes(channels);
        let ftimes = self.extract_ftimes(network, channels, min_time);
        let all_shifts = self.compute_all_shifts(&rftimes, &ftimes);

        if all_shifts.is_empty() {
            let value = izip!(rftimes, ftimes)
                .filter(|(rfctimes, fctimes)| rfctimes.is_empty() && fctimes.is_empty())
                .count() as f64
                / num_channels;
            Ok((
                TimeValuePair { time: 0.0, value },
                TimeValuePair { time: 0.0, value },
            ))
        } else {
            let precision_shift = all_shifts
                .iter()
                .dedup_by(|a, b| (a.value - b.value).abs() < TOL)
                .max_by(|a, b| {
                    self.precision(&all_shifts, a.value)
                        .partial_cmp(&self.precision(&all_shifts, b.value))
                        .unwrap()
                })
                .unwrap()
                .value;
            let precision_value = self.precision(&all_shifts, precision_shift) / num_channels;

            let recall_shift = all_shifts
                .iter()
                .dedup_by(|a, b| (a.value - b.value).abs() < TOL)
                .max_by(|a, b| {
                    self.recall(&all_shifts, a.value)
                        .partial_cmp(&self.recall(&all_shifts, b.value))
                        .unwrap()
                })
                .unwrap()
                .value;
            let recall_value = self.recall(&all_shifts, recall_shift) / num_channels;

            // let recall = match all_shifts
            //     .iter()
            //     .dedup_by(|a, b| (a.value - b.value).abs() < TOL)
            //     .max_by(|a, b| {
            //         self.recall(&all_shifts, a.value)
            //             .partial_cmp(&self.recall(&all_shifts, b.value))
            //             .unwrap()
            //     }) {
            //     Some(shift) => self.recall(&all_shifts, shift.value) / num_channels,
            //     // If there are no shifts at all, count the number of channels which are pairs of empty spike ftimes
            //     None => {
            //         izip!(rftimes, ftimes)
            //             .filter(|(rfctimes, fctimes)| rfctimes.is_empty() && fctimes.is_empty())
            //             .count() as f64
            //             / num_channels
            //     }
            // };

            Ok((
                TimeValuePair {
                    time: precision_shift,
                    value: precision_value,
                },
                TimeValuePair {
                    time: recall_shift,
                    value: recall_value,
                },
            ))
        }
    }
}

/// Returns a random vector on the unit sphere.
pub fn rand_unit<R: Rng>(dim: usize, rng: &mut R) -> Vec<f64> {
    let mut z = (0..dim).map(|_| rng.random::<f64>()).collect::<Vec<f64>>();
    let mut z_norm = l2_norm(&z);
    while z_norm < TOL {
        z = (0..dim).map(|_| rng.random::<f64>()).collect::<Vec<f64>>();
        z_norm = l2_norm(&z);
    }
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
/// In future version, this trait will be replaced by its [`faer`](https://faer-rs.github.io) equivalent.
pub trait RealLinearOperator {
    /// Returns the dimension of the linear operator.
    fn dim(&self) -> usize;

    /// Apply the linear operator to the vector z (in-place).
    fn apply_in(&self, z: &mut Vec<f64>);

    /// Returns the spectral radius, i.e., the largest eigenvalue magnitude, by computing a compression via Arnoldi iterations.
    fn spectral_radius(&self, seed: u64) -> Result<f64, SNNError> {
        // Init the random number generator from the provided seed
        let mut rng = StdRng::seed_from_u64(seed);

        // Init the previous spectral radius with NaN
        let mut spectral_radius = f64::NAN;
        let mut prev_spectral_radius = f64::NAN;

        // Init the vector z randomly on the unit sphere
        // Note: the algorithm might fail if the initial vector has zero components in the dominant eigenspace
        let mut z0: Vec<f64> = rand_unit(self.dim(), &mut rng);

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
            log::debug!("Iter {}: spectral radius is {}", k, spectral_radius);

            if (spectral_radius - prev_spectral_radius).abs()
                / spectral_radius.max(prev_spectral_radius)
                <= REL_TOL
            {
                log::trace!(
                    "Spectral radius determined! Value: {}. Number of Arnoldi iterations: {}",
                    spectral_radius,
                    k
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
