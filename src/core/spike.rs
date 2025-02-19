//! Spike train related module.
use itertools::Itertools;
use log;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{weighted::WeightedIndex, Distribution, Uniform};
use rtnorm_rust::rtnorm;
use serde::{Deserialize, Serialize};

use crate::core::REFRACTORY_PERIOD;
use crate::error::SNNError;

/// The number of iterations for Gibbs sampling.
pub const NUM_ITER_GIBBS_SAMPLING: usize = 100;

/// A spike produced by a neuron at a given time.
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

/// Implement the `PartialOrd` trait for `Spike` to allow sorting by time.
impl PartialOrd for Spike {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.time.partial_cmp(&other.time)
    }
}

/// Returns a new perturbed spike train where every spikes have been randomly jittered.
/// The new perturbed spike train still satisfies the refractory period.
pub fn rand_jitter(
    rtimes: &Vec<Vec<f64>>,
    start: f64,
    end: f64,
    sigma: f64,
    seed: u64,
) -> Result<Vec<Vec<f64>>, SNNError> {
    let mut rng = rgsl::Rng::new(rgsl::RngType::default()).unwrap();
    rng.set(seed as usize);

    if rtimes.iter().any(|rctimes| rctimes.iter().any(|ft| (*ft < start) | (*ft > end))) {
        return Err(SNNError::InvalidParameter(
            "Invalid time range: all nominal firing times should be in [start, end]".to_string(),
        ));
    }

    let mut times = rtimes.clone();
    times
        .iter_mut()
        .zip_eq(rtimes.iter())
        .for_each(|(ctimes, rctimes)| {
            let num_spikes = ctimes.len();
            for _ in 0..NUM_ITER_GIBBS_SAMPLING {
                std::iter::once(None)
                    .chain((0..num_spikes).map(|i| Some(i)))
                    .chain(std::iter::once(None))
                    .tuple_windows::<(_, _, _)>()
                    .map(|(prev_i, i, next_i)| (prev_i, i.unwrap(), next_i))
                    .step_by(2)
                    .for_each(|(prev_i, i, next_i)| {
                        (ctimes[i], _) = match (prev_i, next_i) {
                            (None, None) => rtnorm(&mut rng, start, end, rctimes[i], sigma),
                            (None, Some(next_i)) => rtnorm(
                                &mut rng,
                                start,
                                (ctimes[next_i] - REFRACTORY_PERIOD).clamp(start, end),
                                rctimes[i],
                                sigma,
                            ),
                            (Some(prev_i), None) => rtnorm(
                                &mut rng,
                                (ctimes[prev_i] + REFRACTORY_PERIOD).clamp(start, end),
                                end,
                                rctimes[i],
                                sigma,
                            ),
                            (Some(prev_i), Some(next_i)) => rtnorm(
                                &mut rng,
                                (ctimes[prev_i] + REFRACTORY_PERIOD).clamp(start, end),
                                (ctimes[next_i] - REFRACTORY_PERIOD).clamp(start, end),
                                rctimes[i],
                                sigma,
                            ),
                        }
                    });

                (0..num_spikes)
                    .map(|i| Some(i))
                    .chain(std::iter::once(None))
                    .tuple_windows::<(_, _, _)>()
                    .map(|(prev_i, i, next_i)| (prev_i.unwrap(), i.unwrap(), next_i))
                    .step_by(2)
                    .for_each(|(prev_i, i, next_i)| {
                        (ctimes[i], _) = match next_i {
                            None => rtnorm(
                                &mut rng,
                                (ctimes[prev_i] + REFRACTORY_PERIOD).clamp(start, end),
                                end,
                                rctimes[i],
                                sigma,
                            ),
                            Some(next_i) => rtnorm(
                                &mut rng,
                                (ctimes[prev_i] + REFRACTORY_PERIOD).clamp(start, end),
                                (ctimes[next_i] - REFRACTORY_PERIOD).clamp(start, end),
                                rctimes[i],
                                sigma,
                            ),
                        }
                    });
            }
        });

    Ok(times)
}

/// Returns a random cyclic spike train with a given period and firing rate.
pub fn rand_cyclic(
    num_neurons: usize,
    period: f64,
    firing_rate: f64,
    seed: u64,
) -> Result<Vec<Vec<f64>>, SNNError> {
    if period <= 0.0 {
        return Err(SNNError::InvalidParameter(
            "Invalid period value: must be positive".to_string(),
        ));
    }

    if firing_rate < 0.0 {
        return Err(SNNError::InvalidParameter(
            "Invalid firing rate value: must be non-negative".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut times: Vec<Vec<f64>> = vec![vec![]; num_neurons];

    let weights = p_num_spikes(period, firing_rate);
    let mean_num_spikes = weights
        .iter()
        .enumerate()
        .fold(0.0, |acc, (n, p)| acc + n as f64 * p);
    let p_num_spikes =
        WeightedIndex::new(weights).map_err(|e| SNNError::InvalidOperation(e.to_string()))?;

    let uniform_ref =
        Uniform::new(0.0, period).map_err(|e| SNNError::InvalidParameter(e.to_string()))?;

    for id in 0..num_neurons {
        let num_spikes = p_num_spikes.sample(&mut rng);

        if num_spikes == 0 {
            continue;
        }

        let t0 = uniform_ref.sample(&mut rng);

        times[id].push(t0);
        let uniform = Uniform::new(0.0, period - num_spikes as f64)
            .map_err(|e| SNNError::InvalidParameter(e.to_string()))?;

        let mut tmp_times: Vec<f64> = (0..num_spikes - 1)
            .map(|_| uniform.sample(&mut rng))
            .collect();

        // Sort the sampled intermediate times
        tmp_times.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("Problem with sorting the intermediate times while sampling.")
        });

        // Add refractory periods to intermediate times and extend the new firing times
        times[id].extend(
            tmp_times
                .iter()
                .enumerate()
                .map(|(n, t)| (((n + 1) as f64) * REFRACTORY_PERIOD + t + t0) % period),
        );

        // Sort the new firing times
        times[id].sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("Problem with sorting the new firing times while sampling.")
        });
    }

    let num_spikes = times.iter().map(|ctimes| ctimes.len()).sum::<usize>();
    log::trace!(
        "{} spikes sampled over {} channels (expected number of spikes per channel is {})",
        num_spikes,
        num_neurons,
        mean_num_spikes
    );

    Ok(times)
}

/// Computes the probabilities for the number of spikes based on the given period and firing rate.
fn p_num_spikes(period: f64, firing_rate: f64) -> Vec<f64> {
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
// }

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;
    use itertools::Itertools;

    use crate::core::REFRACTORY_PERIOD;

    const SEED: u64 = 42;

    #[test]
    fn test_rand_cyclic() {
        // Test invalid parameters
        assert_eq!(
            rand_cyclic(10, -10.0, 1.0, SEED),
            Err(SNNError::InvalidParameter(
                "Invalid period value: must be positive".to_string()
            ))
        );

        assert_eq!(
            rand_cyclic(10, 0.0, 1.0, SEED),
            Err(SNNError::InvalidParameter(
                "Invalid period value: must be positive".to_string()
            ))
        );

        assert_eq!(
            rand_cyclic(10, 10.0, -1.0, SEED),
            Err(SNNError::InvalidParameter(
                "Invalid firing rate value: must be non-negative".to_string()
            ))
        );

        let ftimes = rand_cyclic(1000, 100.0, 1.0, SEED).unwrap();
        assert!(ftimes
            .iter()
            .all(|ctimes| ctimes.iter().tuple_windows().all(|(ft0, ft1)| ft0 <= ft1)));

        assert!(ftimes.iter().all(|ctimes| ctimes
            .iter()
            .circular_tuple_windows()
            .all(|(ft0, ft1)| (*ft1 - *ft0).rem_euclid(100.0) >= REFRACTORY_PERIOD)));
    }

    #[test]
    fn test_rand_cyclic_corner_cases() {
        // Test with period close to refractory period
        let small_period = REFRACTORY_PERIOD + 1e-6;
        let ftimes = rand_cyclic(10, small_period, 1.0, SEED).unwrap();
        assert!(ftimes.iter().all(|ctimes| ctimes.len() <= 1)); // At most one spike per neuron
        assert!(ftimes
            .iter()
            .all(|ctimes| ctimes.iter().all(|&t| t >= 0.0 && t < small_period)));

        // Test with very high firing rate
        let high_rate = 1e9;
        let ftimes = rand_cyclic(10, 10.0, high_rate, SEED).unwrap();
        assert!(ftimes.iter().all(|ctimes| !ctimes.is_empty())); // Each neuron should spike
        assert!(ftimes.iter().all(|ctimes| ctimes
            .iter()
            .tuple_windows()
            .all(|(t1, t2)| t2 - t1 >= REFRACTORY_PERIOD)));

        // Test with large number of neurons
        let large_neurons = 10_000;
        let ftimes = rand_cyclic(large_neurons, 100.0, 1.0, SEED).unwrap();
        assert_eq!(ftimes.len(), large_neurons);
        assert!(ftimes
            .iter()
            .all(|ctimes| ctimes.iter().all(|&t| t >= 0.0 && t < 100.0)));

        // Test with very large period
        let large_period = 1e6;
        let ftimes = rand_cyclic(10, large_period, 0.1, SEED).unwrap();
        assert!(ftimes
            .iter()
            .all(|ctimes| ctimes.iter().all(|&t| t >= 0.0 && t < large_period)));
        assert!(ftimes.iter().all(|ctimes| ctimes
            .iter()
            .tuple_windows()
            .all(|(t1, t2)| t2 - t1 >= REFRACTORY_PERIOD)));
    }

    #[test]
    fn test_rand_jitter() {
        let rtimes =
            rand_cyclic(1, 100.0, 1.0, SEED).expect("Failed to generate periodic spike train");
        assert!(rtimes.iter().all(|ctimes| ctimes
            .iter()
            .tuple_windows()
            .all(|(ft0, ft1)| *ft0 <= *ft1 + REFRACTORY_PERIOD)));

        let times = rand_jitter(&rtimes, 0.0, 100.0, 1.0, SEED).unwrap();
        assert!(times.iter().all(|ctimes| ctimes
            .iter()
            .tuple_windows()
            .all(|(ft0, ft1)| *ft0 <= *ft1 + REFRACTORY_PERIOD)));
        assert!(times
            .iter()
            .all(|ctimes| ctimes.iter().all(|ft| (*ft >= 0.0) & (*ft <= 100.0))));

        let times = rand_jitter(&rtimes, 0.0, 100.0, 1000.0, SEED).unwrap();
        assert!(times.iter().all(|ctimes| ctimes
            .iter()
            .tuple_windows()
            .all(|(ft0, ft1)| *ft0 <= *ft1 + REFRACTORY_PERIOD)));
        assert!(times
            .iter()
            .all(|ctimes| ctimes.iter().all(|ft| (*ft >= 0.0) & (*ft <= 100.0))));

        let times = rand_jitter(&rtimes, f64::NEG_INFINITY, f64::INFINITY, 1.0, SEED).unwrap();
        assert!(times.iter().all(|ctimes| ctimes
            .iter()
            .tuple_windows()
            .all(|(ft0, ft1)| *ft0 <= *ft1 + REFRACTORY_PERIOD)));
    }

    #[test]
    fn test_p_num_spikes() {
        // test probabilities for a few cases
        assert!(p_num_spikes(2.0, 1.0)
            .into_iter()
            .zip([0.333333, 0.666667])
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(p_num_spikes(10.0, 0.5)
            .into_iter()
            .zip([
                0.029678, 0.148392, 0.296784, 0.302967, 0.166941, 0.048305, 0.006595, 0.000335,
                0.000004, 0.000000
            ])
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(p_num_spikes(100.0, 0.0)
            .into_iter()
            .zip([1.0])
            .all(|(a, b)| (a - b).abs() < 1e-6));

        // test length of output
        assert_eq!(p_num_spikes(1000.0001, 0.1).len(), 1001);
        assert_eq!(p_num_spikes(1000.0, 0.1).len(), 1000);

        // test probabilities sum to one
        assert!((p_num_spikes(1000.0, 1.0).into_iter().sum::<f64>() - 1.0).abs() < 1e-6);

        // test expected value
        assert!(
            (p_num_spikes(1.00001, 100000000000.0)
                .iter()
                .enumerate()
                .map(|(n, p)| (n as f64 * p))
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-6
        );
        assert!(
            (p_num_spikes(50.0, 0.2)
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
