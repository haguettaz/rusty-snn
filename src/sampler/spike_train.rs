//! This module provides functionality for sampling random periodic spike trains.
//!
//! # Examples
//!
//! ```rust
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//! use rusty_snn::sampler::spike_train::PeriodicSpikeTrainSampler;
//!
//! let mut rng = StdRng::seed_from_u64(42);
//!
//! let period = 100.0;
//! let firing_rate = 0.2;
//! let num_channels = 10;
//!
//! let sampler = PeriodicSpikeTrainSampler::build(period, firing_rate).unwrap();
//! let spike_trains = sampler.sample(num_channels, &mut rng);
//!
//! for spike_train in spike_trains.iter() {
//!    println!("Channel {}: {:?}", spike_train.id(), spike_train.firing_times());
//! }
//! ```

use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand::Rng;
use std::error::Error;
use std::{fmt, vec};

use crate::core::spike_train::{self, SpikeTrain};

/// Error type for the `PeriodicSpikeTrainSampler` struct.
#[derive(Debug, PartialEq)]
pub enum PeriodicSpikeTrainSamplerError {
    /// Returned when the specified period is not positive.
    InvalidPeriod,
    /// Returned when the specified firing rate is negative.
    InvalidFiringRate,
    /// Returned when the computation of spike weights fails.
    InvalidNumSpikeWeights(String),
}

impl fmt::Display for PeriodicSpikeTrainSamplerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PeriodicSpikeTrainSamplerError::InvalidPeriod => {
                write!(f, "The spike train period must be positive.")
            }
            PeriodicSpikeTrainSamplerError::InvalidFiringRate => {
                write!(f, "The firing rate must be non-negative.")
            }
            PeriodicSpikeTrainSamplerError::InvalidNumSpikeWeights(e) => {
                write!(f, "Failed to compute the spike weights: {}", e)
            }
        }
    }
}

impl Error for PeriodicSpikeTrainSamplerError {}

/// Represents a sampler for generating periodic spike trains.
#[derive(Debug, PartialEq)]
pub struct PeriodicSpikeTrainSampler {
    /// The period over which spikes are generated.
    period: f64,
    /// The rate at which spikes are fired.
    firing_rate: f64,
    /// The probabilities for the number of spikes.
    num_spikes_probs: WeightedIndex<f64>,
}

impl PeriodicSpikeTrainSampler {
    /// Creates a new `PeriodicSpikeTrainSampler` instance with the specified period and firing rate.
    ///
    /// # Parameters
    /// - `period`: The period over which spikes are generated.
    /// - `firing_rate`: The rate at which spikes are fired.
    ///
    /// # Returns
    /// A new `PeriodicSpikeTrainSampler` instance.
    ///
    /// # Errors
    /// Returns an error if the period is not positive or the firing rate is negative.
    pub fn build(period: f64, firing_rate: f64) -> Result<Self, PeriodicSpikeTrainSamplerError> {
        if period <= 0.0 {
            return Err(PeriodicSpikeTrainSamplerError::InvalidPeriod);
        }

        if firing_rate < 0.0 {
            return Err(PeriodicSpikeTrainSamplerError::InvalidFiringRate);
        }
        let num_spikes_probs =
            WeightedIndex::new(Self::compute_num_spikes_probs(period, firing_rate)).map_err(|e| {
                PeriodicSpikeTrainSamplerError::InvalidNumSpikeWeights(e.to_string())
            })?;

        Ok(PeriodicSpikeTrainSampler {
            period,
            firing_rate,
            num_spikes_probs,
        })
    }


    /// Samples spike trains with a given number of channels.
    ///
    /// # Parameters
    /// - `num_channels`: The number of channels to sample spike trains for.
    /// - `rng`: A mutable reference to a random number generator implementing the `Rng` trait.
    ///
    /// # Returns
    /// A vector of vectors, where each inner vector contains the firing times for a channel.
    pub fn sample<R: Rng>(&self, num_channels: usize, rng: &mut R) -> Vec<SpikeTrain> {
        let mut spike_trains: Vec<SpikeTrain> = Vec::with_capacity(num_channels);

        let uniform_ref = Uniform::new(0.0, self.period);

        for id in 0..num_channels {
            let num_spikes = self.num_spikes_probs.sample(rng);
            let mut new_firing_times = Vec::with_capacity(num_spikes);

            if num_spikes == 0 {
                spike_trains.push(SpikeTrain::build(id, &new_firing_times).unwrap());
                continue;
            }

            let ref_spike = uniform_ref.sample(rng);
            new_firing_times.push(ref_spike);

            if num_spikes == 1 {
                spike_trains.push(SpikeTrain::build(id, &new_firing_times).unwrap());
                continue;
            }

            let uniform = Uniform::new(0.0, self.period - num_spikes as f64);

            let mut tmp_times: Vec<f64> =
                (0..num_spikes - 1).map(|_| uniform.sample(rng)).collect();

            // Sort the sampled intermediate times
            tmp_times.sort_by(|a, b| {
                a.partial_cmp(b)
                    .expect("Problem with sorting the intermediate times while sampling.")
            });

            // Add refractory periods to intermediate times and extend the new firing times
            new_firing_times.extend(
                tmp_times.iter().enumerate().map(|(n, t)| (n as f64 + 1. + t + ref_spike) % self.period),
            );

            // Sort the new firing times
            new_firing_times.sort_by(|a, b| {
                a.partial_cmp(b)
                    .expect("Problem with sorting the new firing times while sampling.")
            });

            spike_trains.push(SpikeTrain::build(id, &new_firing_times).unwrap());
        }

        spike_trains
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

        // Determine the maximum number of spikes based on the period
        let max_num_spikes = if period <= period.floor() {
            period as usize - 1
        } else {
            period as usize
        };

        // Compute the (unnormalized) log probabilities for the number of spikes (more numerically stable)
        let log_weights = (0..=max_num_spikes).scan(0.0, |state, n| {
            if n > 0 {
                *state += (n as f64).ln();
            }
            Some((n as f64 - 1.0) * (firing_rate * (period - n as f64)).ln() - *state)
        });

        // To avoid overflow when exponentiating, normalize the log probabilities by subtracting the maximum value
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
    fn test_sampler_build() {
        assert_eq!(
            PeriodicSpikeTrainSampler::build(-10.0, 1.0),
            Err(PeriodicSpikeTrainSamplerError::InvalidPeriod)
        );

        assert_eq!(
            PeriodicSpikeTrainSampler::build(0.0, 1.0),
            Err(PeriodicSpikeTrainSamplerError::InvalidPeriod)
        );

        assert_eq!(
            PeriodicSpikeTrainSampler::build(10.0, -1.0),
            Err(PeriodicSpikeTrainSamplerError::InvalidFiringRate)
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
            (PeriodicSpikeTrainSampler::compute_num_spikes_probs(1.00001, 100000000000.0)
                .iter()
                .enumerate()
                .map(|(n, p)| (n as f64 * p))
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-6
        );
        assert!(
            (PeriodicSpikeTrainSampler::compute_num_spikes_probs(50.0, 0.2)
                .iter()
                .enumerate()
                .map(|(n, p)| (n as f64 * p))
                .sum::<f64>()
                - 7.225326)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn test_sample_sorted() {
        let mut rng = StdRng::seed_from_u64(42);

        let sampler = PeriodicSpikeTrainSampler::build(100.0, 0.2).unwrap();
        let spike_trains = sampler.sample(10, &mut rng);

        for spike_train in spike_trains.iter() {
            assert!(spike_train.firing_times().windows(2).all(|w| w[0] <= w[1]));
        }
    }

    #[test]
    fn test_sample_refractory_period() {
        let mut rng = StdRng::seed_from_u64(42);

        let sampler = PeriodicSpikeTrainSampler::build(100.0, 10.0).unwrap();
        let spike_trains = sampler.sample(2, &mut rng);

        for spike_train in spike_trains {
            let mut pairs = spike_train.firing_times()
                .windows(2)
                .map(|w| (w[0].clone(), w[1].clone()))
                .collect::<Vec<_>>();
            if let (Some(first), Some(last)) = (spike_train.firing_times().first(), spike_train.firing_times().last()) {
                pairs.push((last.clone(), first.clone()));
            }
            for (t1, t2) in pairs {
                // assert!((t1 - t2).rem_euclid(100.0) >= 1.0); // no need since the firing times are (cyclically) sorted
                assert!((t2 - t1).rem_euclid(100.0) >= 1.0);
            }
        }
    }
}
