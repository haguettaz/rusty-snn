//! Module implementing the concept of a spike train.
//! What if we redefine a spike train as a (sorted) Vec<Spike>?

use itertools::Itertools;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::error::SNNError;

/// Represents a spike train associated with a specific neuron.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Spike {
    /// The ID of the neuron producing the spike.
    source_id: usize,
    /// The time at which the spike is produced.
    time: f64,
}

impl Spike {
    /// Create a new output spike with the specified parameters.
    pub fn new(source_id: usize, time: f64) -> Self {
        Spike { source_id, time }
    }

    /// Returns the ID of the neuron producing the spike.
    pub fn source_id(&self) -> usize {
        self.source_id
    }

    /// Returns the time at which the spike is produced.
    pub fn time(&self) -> f64 {
        self.time
    }
}

/// Represents a spike train associated with a specific neuron.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct InSpike {
    /// The weight of the synapse along which the spike is received.
    weight: f64,
    /// The time at which the spike is received.
    time: f64,
}

impl InSpike {
    /// Create a new input spike with the specified parameters.
    pub fn new(weight: f64, time: f64) -> Self {
        InSpike { weight, time }
    }

    /// Returns the weight of the synapse along which the spike is received.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Returns the time at which the spike is received.
    pub fn time(&self) -> f64 {
        self.time
    }
}

/// Represents a spike train associated with a specific neuron.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct InputSpike {
    /// The ID of the input along which the spike is received.
    input_id: usize,
    /// The time at which the spike is received.
    time: f64,
}

impl InputSpike {
    /// Create a new input spike with the specified parameters.
    pub fn new(input_id: usize, time: f64) -> Self {
        InputSpike { input_id, time }
    }

    /// Returns the ID of the input along which the spike is received.
    pub fn input_id(&self) -> usize {
        self.input_id
    }

    /// Returns the time at which the spike is received.
    pub fn time(&self) -> f64 {
        self.time
    }
}

// /// Represents a spike train associated with a specific neuron.
// #[derive(Debug, PartialEq, Clone)]
// pub struct SpikeTrain {
//     // id: usize,
//     // firing_times: Vec<f64>,
//     spikes: Vec<Spike>,
// }

// impl SpikeTrain {
// /// Create a spike train with the specified parameters.
// /// If necessary, the firing times are sorted.
// /// The function returns an error for invalid firing times.
// pub fn build(id: usize, firing_times: &[f64]) -> Result<Self, SNNError> {
//     for t in firing_times {
//         if !t.is_finite() {
//             return Err(SNNError::InvalidFiringTimes);
//         }
//     }

//     let mut firing_times = firing_times.to_vec();
//     firing_times.sort_by(|t1, t2| {
//         t1.partial_cmp(t2).unwrap_or_else(|| {
//             panic!("Comparison failed: NaN values should have been caught earlier")
//         })
//     });

//     for ts in firing_times.windows(2) {
//         if ts[1] - ts[0] <= REFRACTORY_PERIOD {
//             return Err(SNNError::RefractoryPeriodViolation {
//                 t1: ts[0],
//                 t2: ts[1],
//             });
//         }
//     }

//     Ok(SpikeTrain {
//         id,
//         firing_times: firing_times,
//     })
// }

/// Samples spike trains with a given number of channels.
///
/// # Parameters
/// - `num_neurons`: The number of channels to sample spike trains for.
/// - `rng`: A mutable reference to a random number generator implementing the `Rng` trait.
///
/// # Returns
/// A vector of vectors, where each inner vector contains the firing times for a channel.
pub fn rand<R: Rng>(
    num_neurons: usize,
    period: f64,
    firing_rate: f64,
    rng: &mut R,
) -> Result<Vec<Spike>, SNNError> {
    let mut spike_train: Vec<Spike> = vec![];

    if period <= 0.0 {
        return Err(SNNError::InvalidPeriod);
    }

    if firing_rate < 0.0 {
        return Err(SNNError::InvalidFiringRate);
    }
    let num_spikes_probs = WeightedIndex::new(compute_num_spikes_probs(period, firing_rate))
        .map_err(|e| SNNError::InvalidNumSpikeWeights(e.to_string()))?;

    let uniform_ref = Uniform::new(0.0, period);

    for id in 0..num_neurons {
        let num_spikes = num_spikes_probs.sample(rng);

        if num_spikes == 0 {
            // spike_trains.push(Spike {
            //     neuron_id: id,
            //     time: new_firing_times,
            // });
            continue;
        }

        let mut new_firing_times = Vec::with_capacity(num_spikes);

        let ref_time = uniform_ref.sample(rng);

        if num_spikes == 1 {
            spike_train.push(Spike {
                source_id: id,
                time: ref_time,
            });
            continue;
        }

        new_firing_times.push(ref_time);
        let uniform = Uniform::new(0.0, period - num_spikes as f64);

        let mut tmp_times: Vec<f64> = (0..num_spikes - 1).map(|_| uniform.sample(rng)).collect();

        // Sort the sampled intermediate times
        tmp_times.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("Problem with sorting the intermediate times while sampling.")
        });

        // Add refractory periods to intermediate times and extend the new firing times
        new_firing_times.extend(
            tmp_times
                .iter()
                .enumerate()
                .map(|(n, t)| (n as f64 + 1. + t + ref_time) % period),
        );

        // // Sort the new firing times
        // new_firing_times.sort_by(|a, b| {
        //     a.partial_cmp(b)
        //         .expect("Problem with sorting the new firing times while sampling.")
        // });

        let new_spikes: Vec<Spike> = new_firing_times
            .iter()
            .map(|t| Spike::new(id, *t))
            .collect();

        spike_train.extend(new_spikes);
    }

    spike_train.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .expect("Problem with sorting the new firing times.")
    });

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

//     /// Returns the ID of the neuron associated with the spike train.
//     pub fn id(&self) -> usize {
//         self.id
//     }

//     /// Returns the firing times of the spike train.
//     pub fn firing_times(&self) -> &[f64] {
//         &self.firing_times[..]
//     }
// }

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;
    use crate::REFRACTORY_PERIOD;

    const SEED: u64 = 42;

    // #[test]
    // fn test_spike_train_build() {
    //     // Test valid spike trains with unsorted firing times
    //     let spike_train = SpikeTrain::build(0, &[0.0, 2.0, 5.0]).unwrap();
    //     assert_eq!(spike_train.firing_times(), &[0.0, 2.0, 5.0]);

    //     // Test valid spike trains with unsorted firing times
    //     let spike_train = SpikeTrain::build(0, &[0.0, 5.0, 2.0]).unwrap();
    //     assert_eq!(spike_train.firing_times(), &[0.0, 2.0, 5.0]);

    //     // Test empty spike train
    //     let spike_train = SpikeTrain::build(0, &[]).unwrap();
    //     assert_eq!(spike_train.firing_times(), &[] as &[f64]);

    //     // Test invalid spike train (NaN values)
    //     let spike_train = SpikeTrain::build(0, &[0.0, 5.0, f64::NAN]);
    //     assert_eq!(spike_train, Err(SNNError::InvalidFiringTimes));

    //     // Test invalid spike train (refractory period violation)
    //     let spike_train = SpikeTrain::build(0, &[0.0, 1.0]);
    //     assert_eq!(
    //         spike_train,
    //         Err(SNNError::RefractoryPeriodViolation { t1: 0.0, t2: 1.0 })
    //     );

    //     // Test invalid spike train (strict refractory period violation)
    //     let spike_train = SpikeTrain::build(0, &[0.0, 5.0, 3.0, 4.5]);
    //     assert_eq!(
    //         spike_train,
    //         Err(SNNError::RefractoryPeriodViolation { t1: 4.5, t2: 5.0 })
    //     );
    // }

    #[test]
    fn test_spike_train_rand() {
        let mut rng = StdRng::seed_from_u64(SEED);

        // Test invalid parameters
        assert_eq!(rand(10, -10.0, 1.0, &mut rng), Err(SNNError::InvalidPeriod));

        assert_eq!(rand(10, 0.0, 1.0, &mut rng), Err(SNNError::InvalidPeriod));

        assert_eq!(
            rand(10, 10.0, -1.0, &mut rng),
            Err(SNNError::InvalidFiringRate)
        );

        let spike_trains = rand(50, 100.0, 1.0, &mut rng).unwrap();

        // Test for sorted firing times and refractory period
        for id in 0..50 {
            let spike_train = spike_trains
                .iter()
                .filter(|spike| spike.source_id == id)
                .collect::<Vec<_>>();

            assert!(spike_train
                .windows(2)
                .all(|spikes| spikes[0].time <= spikes[1].time));

            assert!(spike_train
                .iter()
                .circular_tuple_windows()
                .all(
                    |(spike_1, spike_2)| (spike_2.time() - spike_1.time()).rem_euclid(100.0)
                        >= REFRACTORY_PERIOD
                ));
        }
    }

    #[test]
    fn test_compute_num_spikes_probs() {
        // test probabilities for a few cases
        assert!(compute_num_spikes_probs(2.0, 1.0)
            .into_iter()
            .zip([0.333333, 0.666667])
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(compute_num_spikes_probs(10.0, 0.5)
            .into_iter()
            .zip([
                0.029678, 0.148392, 0.296784, 0.302967, 0.166941, 0.048305, 0.006595, 0.000335,
                0.000004, 0.000000
            ])
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(compute_num_spikes_probs(100.0, 0.0)
            .into_iter()
            .zip([1.0])
            .all(|(a, b)| (a - b).abs() < 1e-6));

        // test length of output
        assert_eq!(compute_num_spikes_probs(1000.0001, 0.1).len(), 1001);
        assert_eq!(compute_num_spikes_probs(1000.0, 0.1).len(), 1000);

        // test probabilities sum to one
        assert!(
            (compute_num_spikes_probs(1000.0, 1.0)
                .into_iter()
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-6
        );

        // test expected value
        assert!(
            (compute_num_spikes_probs(1.00001, 100000000000.0)
                .iter()
                .enumerate()
                .map(|(n, p)| (n as f64 * p))
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-6
        );
        assert!(
            (compute_num_spikes_probs(50.0, 0.2)
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
