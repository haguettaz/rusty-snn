//! Memorization metrics module.
use itertools::Itertools;

use crate::core::REFRACTORY_PERIOD;
use crate::error::SNNError;

use core::f64;
use faer::Mat;
use log;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Relative tolerance for two floating point numbers to be considered equal.
const REL_TOL: f64 = 1e-5;

/// Absolute tolerance for two time shifts to be considered equal.
const TOL: f64 = 1e-12;

/// A similarity measure to evaluate the precision and recall of a spike train with respect to a reference.
///
/// # Examples
///
/// ```rust
/// use approx::assert_relative_eq;
/// use rusty_snn::core::metrics::Similarity;
///
/// // Create a similarity for a reference collection of spike times and a period of 10.0
/// let ref_times = vec![vec![0.75, 3.25, 4.5, 8.25], vec![2.0, 9.0], vec![1.0, 3.0, 5.75]];
/// let similarity = Similarity::new(ref_times, 10.0).unwrap();
///
/// // Compute the precision and recall of a collection of globally shifted spike times.
/// let times = vec![vec![0.5, 3.0, 4.25, 8.0], vec![1.75, 8.75], vec![0.75, 2.75, 5.5]];
/// let (precision, recall) = similarity.measure(&times, 0.0).unwrap();
///
/// assert_relative_eq!(precision, 1.0, epsilon = 1e-6);
/// assert_relative_eq!(recall, 1.0, epsilon = 1e-6);
/// ```
#[derive(Debug, PartialEq)]
pub struct Similarity {
    /// The reference (periodic) spike train.
    ref_times: Vec<Vec<f64>>,
    /// The period of the reference spike train.
    period: f64,
}

impl Similarity {
    /// Create a new `Similarity` instance with the given reference spike times and period.
    pub fn new(ref_times: Vec<Vec<f64>>, period: f64) -> Result<Similarity, SNNError> {
        if period <= 0.0 {
            return Err(SNNError::InvalidParameter(
                "The period must be a positive number.".to_string(),
            ));
        }
        Ok(Similarity { ref_times, period })
    }

    /// The number of channels in the reference spike train.
    pub fn num_channels(&self) -> usize {
        self.ref_times.len()
    }

    /// Process the spike train so that
    /// - only spikes after start are kept
    /// - the spikes cover only one period in total, with special care for border cases
    fn process_times<'a>(&self, times: &'a Vec<Vec<f64>>, min_time: f64) -> Vec<&'a [f64]> {
        times
            .iter()
            .map(|ctimes| {
                let start = match ctimes.binary_search_by(|time| {
                    time.partial_cmp(&(min_time - REFRACTORY_PERIOD)).unwrap()
                }) {
                    Ok(pos) | Err(pos) => pos,
                };
                let end = match ctimes
                    .binary_search_by(|time| time.partial_cmp(&(min_time + self.period)).unwrap())
                {
                    Ok(pos) | Err(pos) => pos,
                };

                if start < end {
                    if (ctimes[start] + self.period - ctimes[end - 1]).abs() < REFRACTORY_PERIOD {
                        &ctimes[start + 1..end]
                    } else {
                        &ctimes[start..end]
                    }
                } else {
                    &[]
                }
            })
            .collect()
    }

    /// Compute all time shifts between the reference and the spike train.
    fn extended_shifts(&self, times: &Vec<&[f64]>) -> Vec<(f64, usize, usize)> {
        let mut ext_shifts = self
            .ref_times
            .iter()
            .zip_eq(times.iter())
            .flat_map(|(ref_ctimes, ctimes)| {
                ref_ctimes
                    .iter()
                    .cartesian_product(ctimes.iter())
                    .map(|(ref_time, time)| {
                        let dt: f64 = time - ref_time;
                        (
                            dt - self.period * ((dt + self.period / 2.0) / self.period).floor(),
                            ctimes.len(),
                            ref_ctimes.len(),
                        )
                    })
            })
            .collect::<Vec<(f64, usize, usize)>>();
        ext_shifts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        ext_shifts
    }

    /// all_shifts is a triplet with
    /// 1. the shift
    /// 2. the number of spikes in the actual channel spike train (for precision)
    /// 3. the number of spikes in the nominal channel spike train (for recall)
    fn precision(&self, ext_shifts: &Vec<(f64, usize, usize)>, shift: f64) -> f64 {
        let pos = match ext_shifts.binary_search_by(|other| {
            other
                .0
                .partial_cmp(&(shift - REFRACTORY_PERIOD / 2.0))
                .unwrap()
        }) {
            Ok(pos) => ext_shifts[..=pos]
                .iter()
                .rev()
                .enumerate()
                .take_while(|(_, other)| other.0 > shift - REFRACTORY_PERIOD / 2.0)
                .last()
                .map(|(i, _)| pos - i)
                .unwrap_or_default(),
            Err(pos) => pos,
        };

        ext_shifts[pos..]
            .iter()
            .take_while(|other| other.0 < shift + REFRACTORY_PERIOD / 2.0)
            .map(|other| (1.0 - 2.0 * (shift - other.0).abs() / REFRACTORY_PERIOD) / other.1 as f64)
            .chain(
                ext_shifts
                    .iter()
                    .rev()
                    .take_while(|other| other.0 > shift + self.period - REFRACTORY_PERIOD / 2.0)
                    .map(|other| {
                        (1.0 - 2.0 * (shift + self.period - other.0).abs() / REFRACTORY_PERIOD)
                            / other.1 as f64
                    }),
            )
            .sum()
    }

    /// shifts are extended with
    /// 1. the number of spikes in the actual channel spike train (for precision)
    /// 2. the number of spikes in the nominal channel spike train (for recall)
    fn recall(&self, ext_shifts: &Vec<(f64, usize, usize)>, shift: f64) -> f64 {
        let pos = match ext_shifts.binary_search_by(|other| {
            other
                .0
                .partial_cmp(&(shift - REFRACTORY_PERIOD / 2.0))
                .unwrap()
        }) {
            Ok(pos) => ext_shifts[..=pos]
                .iter()
                .rev()
                .enumerate()
                .take_while(|(_, other)| other.0 > shift - REFRACTORY_PERIOD / 2.0)
                .last()
                .map(|(i, _)| pos - i)
                .unwrap_or_default(),
            Err(pos) => pos,
        };

        ext_shifts[pos..]
            .iter()
            .take_while(|other| other.0 < shift + REFRACTORY_PERIOD / 2.0)
            .map(|other| (1.0 - 2.0 * (shift - other.0).abs() / REFRACTORY_PERIOD) / other.2 as f64)
            .chain(
                ext_shifts
                    .iter()
                    .rev()
                    .take_while(|other| other.0 > shift + self.period - REFRACTORY_PERIOD / 2.0)
                    .map(|other| {
                        (1.0 - 2.0 * (shift + self.period - other.0).abs() / REFRACTORY_PERIOD)
                            / other.2 as f64
                    }),
            )
            .sum()
    }

    /// Calculate the precision and recall of a multi-channel spike train with respect to the reference.
    /// Returns an error if the number of channels in the spike trains to compare don't match.
    pub fn measure(&self, times: &Vec<Vec<f64>>, min_time: f64) -> Result<(f64, f64), SNNError> {
        if self.num_channels() != times.len() {
            return Err(SNNError::IncompatibleSpikeTrains(
                format!(
                    "The number of channels in the reference ({}) and the spike train ({}) don't match.",
                    self.num_channels(),
                    times.len()
                )
            ));
        }

        let times = self.process_times(times, min_time);
        let ext_shifts = self.extended_shifts(&times);

        let precision = match ext_shifts
            .iter()
            .dedup_by(|a, b| (a.0 - b.0).abs() < TOL)
            .max_by(|a, b| {
                self.precision(&ext_shifts, a.0)
                    .partial_cmp(&self.precision(&ext_shifts, b.0))
                    .unwrap()
            }) {
            Some(drift) => self.precision(&ext_shifts, drift.0) / self.num_channels() as f64,
            // If there are no shifts at all, count the number of channels which are pairs of empty spike ftimes
            None => {
                self.ref_times
                    .iter()
                    .zip_eq(times.iter())
                    .filter(|(ref_ctimes, ctimes)| ref_ctimes.is_empty() && ctimes.is_empty())
                    .count() as f64
                    / self.num_channels() as f64
            }
        };

        let recall = match ext_shifts
            .iter()
            .dedup_by(|a, b| (a.0 - b.0).abs() < TOL)
            .max_by(|a, b| {
                self.recall(&ext_shifts, a.0)
                    .partial_cmp(&self.recall(&ext_shifts, b.0))
                    .unwrap()
            }) {
            Some(drift) => self.recall(&ext_shifts, drift.0) / self.num_channels() as f64,
            // If there are no shifts at all, count the number of channels which are pairs of empty spike ftimes
            None => {
                self.ref_times
                    .iter()
                    .zip_eq(times.iter())
                    .filter(|(ref_ctimes, ctimes)| ref_ctimes.is_empty() && ctimes.is_empty())
                    .count() as f64
                    / self.num_channels() as f64
            }
        };

        Ok((precision, recall))
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
            log::trace!("Iter {}: spectral radius is {}", k, spectral_radius);

            if (spectral_radius - prev_spectral_radius).abs()
                / spectral_radius.max(prev_spectral_radius)
                <= REL_TOL
            {
                log::info!(
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_process_time() {
        let similarity = Similarity::new(vec![vec![]], 5.0).unwrap();

        assert_eq!(
            similarity.process_times(&vec![vec![]], 0.0),
            vec![&[] as &[f64]]
        );
        assert_eq!(
            similarity.process_times(&vec![vec![1.25, 3.5, 5.25]], 0.0),
            vec![&[1.25, 3.5]]
        );
        assert_eq!(
            similarity.process_times(&vec![vec![-0.25, 3.5, 4.75]], 0.0),
            vec![&[3.5, 4.75]]
        );
        assert_eq!(
            similarity.process_times(&vec![vec![-0.25, 3.5, 5.25]], 0.0),
            vec![&[-0.25, 3.5]]
        );
        assert_eq!(
            similarity.process_times(&vec![vec![0.25, 3.5, 4.75]], 0.0),
            vec![&[3.5, 4.75]]
        );
    }

    #[test]
    fn test_extended_shifts() {
        let similarity = Similarity::new(vec![vec![0.0, 2.5, 4.0]], 5.0).unwrap();

        let ctimes = vec![0.25, 2.0];
        let times = vec![&ctimes[..]];

        assert_eq!(similarity.extended_shifts(&times)[0], (-2.25, 2, 3));
        assert_eq!(similarity.extended_shifts(&times)[1], (-2.0, 2, 3));
        assert_eq!(similarity.extended_shifts(&times)[2], (-0.5, 2, 3));
        assert_eq!(similarity.extended_shifts(&times)[3], (0.25, 2, 3));
        assert_eq!(similarity.extended_shifts(&times)[4], (1.25, 2, 3));
        assert_eq!(similarity.extended_shifts(&times)[5], (2.0, 2, 3));
    }

    #[test]
    fn test_precision_recall() {
        let similarity = Similarity::new(vec![vec![0.75, 2.0, 3.25, 4.5]], 6.0).unwrap();

        let times = vec![vec![5.75, 7.0, 8.25, 9.5]];
        let (precision, recall) = similarity.measure(&times, 5.0).unwrap();
        assert_relative_eq!(precision, 1.0, epsilon = 1e-6);
        assert_relative_eq!(recall, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_precision_recall_empty() {
        let similarity = Similarity::new(vec![vec![0.75, 2.0, 3.25, 4.5]], 6.0).unwrap();
        let times = vec![vec![]];
        let (precision, recall) = similarity.measure(&times, 0.0).unwrap();
        assert_relative_eq!(precision, 0.0, epsilon = 1e-6);
        assert_relative_eq!(recall, 0.0, epsilon = 1e-6);

        let similarity = Similarity::new(vec![vec![]], 6.0).unwrap();
        let times = vec![vec![0.75, 2.0, 3.25, 4.5]];
        let (precision, recall) = similarity.measure(&times, 0.0).unwrap();
        assert_relative_eq!(precision, 0.0, epsilon = 1e-6);
        assert_relative_eq!(recall, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_precision_recall_many_spikes() {
        let similarity = Similarity::new(
            vec![(0..50).map(|i| 2.0_f64 * i as f64).collect(); 100],
            100.0,
        )
        .unwrap();

        let times = vec![(0..50).map(|i| 1.0 + 2.0_f64 * i as f64).collect(); 100];
        let (precision, recall) = similarity.measure(&times, 0.0).unwrap();

        assert_relative_eq!(precision, 1.0, epsilon = 1e-6);
        assert_relative_eq!(recall, 1.0, epsilon = 1e-6);
    }
}
