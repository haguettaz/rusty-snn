//! Spike train related structures.
use log;
use rand::RngCore;
use rand_distr::{weighted::WeightedIndex, Distribution, Uniform};
use serde::{Deserialize, Serialize};

use crate::core::REFRACTORY_PERIOD;
use crate::error::SNNError;

/// An output spike, i.e., a spike emitted by a neuron.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Spike {
    /// The ID of the neuron producing the spike.
    pub neuron_id: usize,
    /// The time at which the spike is produced.
    pub time: f64,
}

impl Spike {
    pub fn new(neuron_id: usize, time: f64) -> Self {
        Spike { neuron_id, time }
    }
}

impl PartialOrd for Spike {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.time.partial_cmp(&other.time)
    }
}

/// A multi-channel (output) spike train, i.e., a sequence of spikes emitted by multiple neurons.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct MultiChannelSpikeTrain {
    pub spike_train: Vec<Vec<f64>>,
}

impl MultiChannelSpikeTrain {
    /// Creates a new empty multi-channel spike train.
    pub fn new_empty() -> Self {
        MultiChannelSpikeTrain {
            spike_train: vec![],
        }
    }

    /// Creates a new multi-channel spike train from a collection of spikes.
    pub fn new_from(mut spikes: Vec<Spike>) -> Self {
        spikes.sort_by(|spike_1, spike_2| spike_1.partial_cmp(&spike_2).unwrap());
        let mut spike_train =
            vec![vec![]; spikes.iter().map(|spike| spike.neuron_id).max().unwrap() + 1];
        spikes.iter().for_each(|spike| {
            spike_train[spike.neuron_id].push(spike.time);
        });
        Self { spike_train }
    }

    pub fn get(&self, source_id: usize) -> Option<&Vec<f64>> {
        self.spike_train.get(source_id)
    }

    pub fn num_channels(&self) -> usize {
        self.spike_train.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Vec<f64>> {
        self.spike_train.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Vec<f64>> {
        self.spike_train.iter_mut()
    }

    /// Returns a new perturbed spike train where every spikes have been randomly jittered.
    /// The new perturbed spike train still satisfies the refractory period.
    #[allow(unused_variables)]
    pub fn perturb<R: RngCore>(&self, jitter_std: f64, rng: &mut R) -> Self {
        todo!();
    }
}

/// A multi-channel cyclic (output) spike train, i.e., a periodic sequence of spikes emitted by multiple neurons.
/// The spikes are sorted by 1) source ID and 2) time.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct MultiChannelCyclicSpikeTrain {
    pub spike_train: Vec<Vec<f64>>,
    pub period: f64,
}

impl MultiChannelCyclicSpikeTrain {
    pub fn new_from(mut spike_train: Vec<Vec<f64>>, period: f64) -> Self {
        spike_train.iter_mut().for_each(|train| {
            train.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        });
        MultiChannelCyclicSpikeTrain {
            spike_train,
            period,
        }
    }

    pub fn new_from_flatten(mut spikes: Vec<Spike>, period: f64) -> Self {
        spikes.sort_by(|spike_1, spike_2| spike_1.partial_cmp(&spike_2).unwrap());
        let mut spike_train =
            vec![vec![]; spikes.iter().map(|spike| spike.neuron_id).max().unwrap() + 1];
        spikes.iter().for_each(|spike| {
            spike_train[spike.neuron_id].push(spike.time);
        });

        Self {
            spike_train,
            period,
        }
    }

    pub fn num_channels(&self) -> usize {
        self.spike_train.len()
    }

    pub fn num_spikes(&self) -> usize {
        self.spike_train.iter().map(|spikes| spikes.len()).sum()
    }

    pub fn get(&self, source_id: usize) -> Option<&Vec<f64>> {
        self.spike_train.get(source_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Vec<f64>> {
        self.spike_train.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Vec<f64>> {
        self.spike_train.iter_mut()
    }

    pub fn flatten(&self) -> Vec<Spike> {
        self.spike_train
            .iter()
            .enumerate()
            .flat_map(|(source_id, spikes)| {
                spikes.iter().map(move |time| Spike {
                    neuron_id: source_id,
                    time: *time,
                })
            })
            .collect()
    }

    pub fn rand<R: RngCore>(
        num_neurons: usize,
        period: f64,
        firing_rate: f64,
        rng: &mut R,
    ) -> Result<Self, SNNError> {
        if period <= 0.0 {
            return Err(SNNError::InvalidParameters(
                "Invalid period value: must be positive".to_string(),
            ));
        }

        if firing_rate < 0.0 {
            return Err(SNNError::InvalidParameters(
                "Invalid firing rate value: must be non-negative".to_string(),
            ));
        }

        let mut spike_train: Vec<Vec<f64>> = vec![vec![]; num_neurons];

        let weights = Self::compute_num_spikes_probs(period, firing_rate);
        let num_spikes_mean = weights
            .iter()
            .enumerate()
            .fold(0.0, |acc, (n, p)| acc + n as f64 * p);
        let num_spikes_probs = WeightedIndex::new(weights)
            .map_err(|e| SNNError::InvalidNumSpikeWeights(e.to_string()))?;

        let uniform_ref =
            Uniform::new(0.0, period).map_err(|e| SNNError::InvalidParameters(e.to_string()))?;

        for id in 0..num_neurons {
            let num_spikes = num_spikes_probs.sample(rng);

            if num_spikes == 0 {
                continue;
            }

            // let mut new_firing_times = Vec::with_capacity(num_spikes);
            let ref_time = uniform_ref.sample(rng);

            // if num_spikes == 1 {
            //     spikes.push(Spike {
            //         source_id: id,
            //         time: ref_time,
            //     });
            //     continue;
            // }

            spike_train[id].push(ref_time);
            let uniform = Uniform::new(0.0, period - num_spikes as f64)
                .map_err(|e| SNNError::InvalidParameters(e.to_string()))?;

            let mut tmp_times: Vec<f64> =
                (0..num_spikes - 1).map(|_| uniform.sample(rng)).collect();

            // Sort the sampled intermediate times
            tmp_times.sort_by(|a, b| {
                a.partial_cmp(b)
                    .expect("Problem with sorting the intermediate times while sampling.")
            });

            // Add refractory periods to intermediate times and extend the new firing times
            spike_train[id].extend(
                tmp_times
                    .iter()
                    .enumerate()
                    .map(|(n, t)| (((n + 1) as f64) * REFRACTORY_PERIOD + t + ref_time) % period),
            );

            // Sort the new firing times
            spike_train[id].sort_by(|a, b| {
                a.partial_cmp(b)
                    .expect("Problem with sorting the new firing times while sampling.")
            });

            // spikes[id].extend(new_firing_times.into_iter());

            // spikes.extend(new_firing_times.into_iter().map(|time| Spike {
            //     source_id: id,
            //     time,
            // }));
        }

        let spike_train = MultiChannelCyclicSpikeTrain {
            period,
            spike_train,
        };
        log::info!(
            "{} spikes sampled over {} channels (expected number of spikes per channel is {})",
            spike_train.num_spikes(),
            spike_train.num_channels(),
            num_spikes_mean
        );

        Ok(spike_train)
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
mod tests {
    use super::*;
    use itertools::Itertools;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::core::REFRACTORY_PERIOD;

    const SEED: u64 = 42;

    #[test]
    fn test_rand_spike_train() {
        let mut rng = StdRng::seed_from_u64(SEED);

        // Test invalid parameters
        assert_eq!(
            MultiChannelCyclicSpikeTrain::rand(10, -10.0, 1.0, &mut rng),
            Err(SNNError::InvalidParameters(
                "Invalid period value: must be positive".to_string()
            ))
        );

        assert_eq!(
            MultiChannelCyclicSpikeTrain::rand(10, 0.0, 1.0, &mut rng),
            Err(SNNError::InvalidParameters(
                "Invalid period value: must be positive".to_string()
            ))
        );

        assert_eq!(
            MultiChannelCyclicSpikeTrain::rand(10, 10.0, -1.0, &mut rng),
            Err(SNNError::InvalidParameters(
                "Invalid firing rate value: must be non-negative".to_string()
            ))
        );

        let multi_cyclic_spike_train =
            MultiChannelCyclicSpikeTrain::rand(50, 100.0, 1.0, &mut rng).unwrap();

        // Test for sorted firing times and refractory period
        for id in 0..50 {
            let spikes = multi_cyclic_spike_train.get(id).unwrap();
            assert!(spikes
                .iter()
                .tuple_windows()
                .all(|(spike_1, spike_2)| spike_1 <= spike_2));
            assert!(spikes
                .iter()
                .circular_tuple_windows()
                .all(|(ft0, ft1)| (*ft1 - *ft0).rem_euclid(100.0) >= REFRACTORY_PERIOD));
        }
    }

    #[test]
    fn test_compute_num_spikes_probs() {
        // test probabilities for a few cases
        assert!(
            MultiChannelCyclicSpikeTrain::compute_num_spikes_probs(2.0, 1.0)
                .into_iter()
                .zip([0.333333, 0.666667])
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
        assert!(
            MultiChannelCyclicSpikeTrain::compute_num_spikes_probs(10.0, 0.5)
                .into_iter()
                .zip([
                    0.029678, 0.148392, 0.296784, 0.302967, 0.166941, 0.048305, 0.006595, 0.000335,
                    0.000004, 0.000000
                ])
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
        assert!(
            MultiChannelCyclicSpikeTrain::compute_num_spikes_probs(100.0, 0.0)
                .into_iter()
                .zip([1.0])
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );

        // test length of output
        assert_eq!(
            MultiChannelCyclicSpikeTrain::compute_num_spikes_probs(1000.0001, 0.1).len(),
            1001
        );
        assert_eq!(
            MultiChannelCyclicSpikeTrain::compute_num_spikes_probs(1000.0, 0.1).len(),
            1000
        );

        // test probabilities sum to one
        assert!(
            (MultiChannelCyclicSpikeTrain::compute_num_spikes_probs(1000.0, 1.0)
                .into_iter()
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-6
        );

        // test expected value
        assert!(
            (MultiChannelCyclicSpikeTrain::compute_num_spikes_probs(1.00001, 100000000000.0)
                .iter()
                .enumerate()
                .map(|(n, p)| (n as f64 * p))
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-6
        );
        assert!(
            (MultiChannelCyclicSpikeTrain::compute_num_spikes_probs(50.0, 0.2)
                .iter()
                .enumerate()
                .map(|(n, p)| (n as f64 * p))
                .sum::<f64>()
                - 7.225326)
                .abs()
                < 1e-6
        );
    }
}
