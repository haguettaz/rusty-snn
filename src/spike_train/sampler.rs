//! This module provides functionality for generating periodic spike trains with a given period and firing rate.
//!
//! The main struct in this module is `PeriodicSpikeTrainSampler`, which allows for the creation of spike trains
//! that follow a periodic pattern. The spike trains are generated based on a specified period and firing rate.
//!
//! # Structs
//!
//! - `PeriodicSpikeTrainSampler`: Represents a sampler for generating periodic spike trains.
//! - `PeriodicSpikeTrainSamplerError`: Enum representing possible errors that can occur when creating a `PeriodicSpikeTrainSampler`.
//!
//! # Methods
//!
//! - `PeriodicSpikeTrainSampler::new`: Creates a new `PeriodicSpikeTrainSampler` with the specified period and firing rate.
//! - `PeriodicSpikeTrainSampler::sample`: Generates spike trains for a given number of channels.
//! - `PeriodicSpikeTrainSampler::compute_num_spikes_probs`: Computes the probabilities for the number of spikes based on the given period and firing rate.
//!
//! # Errors
//!
//! The `PeriodicSpikeTrainSamplerError` enum defines the following errors:
//!
//! - `InvalidPeriod`: Returned when the specified period is not positive.
//! - `InvalidFiringRate`: Returned when the specified firing rate is negative.
//! - `InvalidNumSpikeWeights`: Returned when the computation of spike weights fails.
//!
//! # Examples
//!
//! ```
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//! use rsnn::spike_train::sampler::PeriodicSpikeTrainSampler;
//!
//! let mut rng = StdRng::seed_from_u64(42);
//! let sampler = PeriodicSpikeTrainSampler::new(100.0, 0.2).unwrap();
//! let firing_times = sampler.sample(2, &mut rng);
//! println!("{:?}", firing_times);
//! ```
//!
//! # Tests
//!
//! The module includes tests for the `PeriodicSpikeTrainSampler` struct and its methods. The tests cover the following scenarios:
//!
//! - `test_sampler_new`: Tests the `new` method to ensure it handles invalid inputs correctly.
//! - `test_compute_num_spikes_probs`: Tests the `compute_num_spikes_probs` method to verify the correctness of the computed probabilities.
//! - `test_sample_cyclically_sorted`: Tests the `sample` method to ensure that the firing times are cyclically sorted. (TODO)
//! - `test_sample_refractory_period`: Tests the `sample` method to ensure that the firing times respect the refractory period.
use itertools::enumerate;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand::Rng;

#[derive(Debug, PartialEq)]
pub enum PeriodicSpikeTrainSamplerError {
    InvalidPeriod(String),
    InvalidFiringRate(String),
    InvalidNumSpikeWeights(String),
}

#[derive(Debug, PartialEq)]
pub struct PeriodicSpikeTrainSampler {
    period: f64,
    firing_rate: f64,
    num_spikes_weights: WeightedIndex<f64>,
}

impl PeriodicSpikeTrainSampler {
    pub fn new(period: f64, firing_rate: f64) -> Result<Self, PeriodicSpikeTrainSamplerError> {
        if period <= 0.0 {
            return Err(PeriodicSpikeTrainSamplerError::InvalidPeriod(
                "The period must be positive.".to_string(),
            ));
        }
        
        if firing_rate < 0.0 {
            return Err(PeriodicSpikeTrainSamplerError::InvalidFiringRate(
                "The firing rate must be non-negative.".to_string(),
            ));
        }
        let num_spikes_weights =
        WeightedIndex::new(Self::compute_num_spikes_probs(period, firing_rate));
        
        match num_spikes_weights {
            Ok(num_spikes_weights) => Ok(PeriodicSpikeTrainSampler {
                period,
                firing_rate,
                num_spikes_weights,
            }),
            Err(e) => Err(PeriodicSpikeTrainSamplerError::InvalidNumSpikeWeights(
                e.to_string(),
            )),
        }
    }
    
    /// Samples spike trains with a given number of channels.
    ///
    /// # Parameters
    /// - `num_channels`: The number of channels to sample spike trains for.
    /// - `rng`: A mutable reference to a random number generator implementing the `Rng` trait.
    ///
    /// # Returns
    /// A vector of vectors, where each inner vector contains the firing times for a channel.
    ///
    /// # Example
    /// ```
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// use crate::spike_train::sampler::PeriodicSpikeTrainSampler;
    ///
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let sampler = PeriodicSpikeTrainSampler::new(100.0, 0.2).unwrap();
    /// let num_channels = 5;
    /// let spike_trains = sampler.sample(num_channels, &mut rng);
    ///
    /// for (i, train) in spike_trains.iter().enumerate() {
    ///     println!("Channel {}: {:?}", i, train);
    /// }
    /// ```
    pub fn sample<R: Rng>(&self, num_channels: usize, rng: &mut R) -> Vec<Vec<f64>> {
        let mut firing_times: Vec<Vec<f64>> = Vec::with_capacity(num_channels);

        let uniform_ref = Uniform::new(0.0, self.period);

        for _ in 0..num_channels {
            let num_spikes = self.num_spikes_weights.sample(rng);
            firing_times.push(Vec::with_capacity(num_spikes));

            if num_spikes == 0 {
                continue;
            }

            let ref_spike = uniform_ref.sample(rng);
            firing_times.last_mut().unwrap().push(ref_spike);

            if num_spikes == 1 {
                continue;
            }

            let uniform = Uniform::new(0.0, self.period - num_spikes as f64);

            let mut tmp_times: Vec<f64> =
                (0..num_spikes - 1).map(|_| uniform.sample(rng)).collect();

            // Sorting safely, treating NaN as equal
            tmp_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            firing_times.last_mut().unwrap().extend(
                enumerate(tmp_times).map(|(n, t)| (n as f64 + 1. + t + ref_spike) % self.period),
            );
        }

        firing_times
    }

    /// Computes the probabilities for the number of spikes based on the given period and firing rate.
    ///
    /// # Parameters
    /// - `period`: The period over which spikes are generated.
    /// - `firing_rate`: The rate at which spikes are fired.
    ///
    /// # Returns
    /// A vector of probabilities for the number of spikes.
    fn compute_num_spikes_probs(period: f64, firing_rate: f64) -> Vec<f64> {
        if firing_rate <= 0.0 {
            return vec![1.0];
        }

        let max_num_spikes = if period <= period.floor() {
            period as usize - 1
        } else {
            period as usize
        };

        let log_weights = (0..=max_num_spikes).scan(0.0, |state, n| {
            if n > 0 {
                *state += (n as f64).ln();
            }
            Some((n as f64 - 1.0) * (firing_rate * (period - n as f64)).ln() - *state)
        });

        // to avoid overflow when exponentiating, normalize the log probabilities by subtracting the maximum value
        let max = log_weights.clone().fold(f64::NEG_INFINITY, f64::max);
        let weights = log_weights.map(|log_p| (log_p - max).exp());

        let sum = weights.clone().sum::<f64>();
        weights.map(|w| w / sum).collect()
    }
}

#[cfg(test)]
/// This module contains tests for the `PeriodicSpikeTrainSampler` struct and its associated methods.
///
/// # Tests
///
/// - `test_sampler_new`: Tests the `new` method of `PeriodicSpikeTrainSampler` to ensure it handles invalid inputs correctly.
/// - `test_compute_num_spikes_probs`: Tests the `compute_num_spikes_probs` method to verify the correctness of the computed probabilities.
/// - `test_sample_cyclically_sorted`: Tests the `sample` method to ensure that the firing times are cyclically sorted.
/// - `test_sample_refractory_period`: Tests the `sample` method to ensure that the firing times respect the refractory period.
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_sampler_new() {
        assert_eq!(
            PeriodicSpikeTrainSampler::new(-10.0, 1.0),
            Err(PeriodicSpikeTrainSamplerError::InvalidPeriod(
                "The period must be positive.".into()
            ))
        );

        assert_eq!(
            PeriodicSpikeTrainSampler::new(0.0, 1.0),
            Err(PeriodicSpikeTrainSamplerError::InvalidPeriod(
                "The period must be positive.".into()
            ))
        );

        assert_eq!(
            PeriodicSpikeTrainSampler::new(10.0, -1.0),
            Err(PeriodicSpikeTrainSamplerError::InvalidFiringRate(
                "The firing rate must be non-negative.".into()
            ))
        );
    }

    #[test]
    fn test_compute_num_spikes_probs() {
        // test probabilities for a few cases
        assert!(
            PeriodicSpikeTrainSampler::compute_num_spikes_probs(2.0, 1.0)
                .into_iter()
                .zip([0.333333, 0.666667])
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
        assert!(
            PeriodicSpikeTrainSampler::compute_num_spikes_probs(10.0, 0.5)
                .into_iter()
                .zip([
                    0.029678, 0.148392, 0.296784, 0.302967, 0.166941, 0.048305, 0.006595, 0.000335,
                    0.000004, 0.000000
                ])
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
        assert!(
            PeriodicSpikeTrainSampler::compute_num_spikes_probs(100.0, 0.0)
                .into_iter()
                .zip([1.0])
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );

        // test length of output
        assert_eq!(
            PeriodicSpikeTrainSampler::compute_num_spikes_probs(1000.0001, 0.1).len(),
            1001
        );
        assert_eq!(
            PeriodicSpikeTrainSampler::compute_num_spikes_probs(1000.0, 0.1).len(),
            1000
        );

        // test probabilities sum to one
        assert!(
            (PeriodicSpikeTrainSampler::compute_num_spikes_probs(1000.0, 1.0)
                .into_iter()
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-6
        );

        // test expected value
        assert!(
            (enumerate(PeriodicSpikeTrainSampler::compute_num_spikes_probs(
                1.00001,
                100000000000.0
            ))
            .map(|(n, p)| (n as f64 * p))
            .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-6
        );
        assert!(
            (enumerate(PeriodicSpikeTrainSampler::compute_num_spikes_probs(
                50.0, 0.2
            ))
            .map(|(n, p)| (n as f64 * p))
            .sum::<f64>()
                - 7.225326)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn test_sample_cyclically_sorted() {
        let mut rng = StdRng::seed_from_u64(42);

        let sampler = PeriodicSpikeTrainSampler::new(100.0, 0.2).unwrap();
        let firing_times = sampler.sample(2, &mut rng);

        for times in firing_times {
            todo!();
        }
    }

    #[test]
    fn test_sample_refractory_period() {
        let mut rng = StdRng::seed_from_u64(42);

        let sampler = PeriodicSpikeTrainSampler::new(100.0, 10.0).unwrap();
        let firing_times = sampler.sample(2, &mut rng);

        for times in firing_times {
            let mut pairs = times
                .windows(2)
                .map(|w| (w[0].clone(), w[1].clone()))
                .collect::<Vec<_>>();
            if let (Some(first), Some(last)) = (times.first(), times.last()) {
                pairs.push((last.clone(), first.clone()));
            }
            for (t1, t2) in pairs {
                // assert!((t1 - t2).rem_euclid(100.0) >= 1.0); // no need since the firing times are (cyclically) sorted
                assert!((t2 - t1).rem_euclid(100.0) >= 1.0);
            }
        }
    }
}
