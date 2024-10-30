use super::spike_train::{PeriodicSpikeTrain, SpikeTrain};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

fn compute_num_spikes_weights(period: f64, firing_rate: f64) -> Vec<f64> {
    let max_num_spikes = if period == period.floor() {
        period as usize - 1
    } else {
        period as usize
    };

    let log_weights: Vec<f64> = (0..=max_num_spikes)
        .scan(0.0, |state, n| {
            if n > 0 {
                *state += (n as f64).ln();
            }
            Some((n as f64 - 1.0) * (firing_rate * (period - n as f64)).ln() - *state)
        })
        .collect();

    // to avoid overflow when exponentiating, normalize the log probabilities by subtracting the maximum value
    let max = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    log_weights
        .iter()
        .map(|log_p| (log_p - max).exp())
        .collect()
}

impl PeriodicSpikeTrain {
    pub fn new_random<R: Rng>(
        period: f64,
        firing_rate: f64,
        num_channels: usize,
        rng: &mut R,
    ) -> Result<Self, &'static str> {
        let mut firing_times = Vec::new();

        let num_spikes_dist =
            WeightedIndex::new(&compute_num_spikes_weights(period, firing_rate)).unwrap();

        for _ in 0..num_channels {
            let num_spikes = rng.sample(&num_spikes_dist);
            let start = rng.gen_range(0.0..period);

            let mut times = (0..num_spikes - 1)
                .map(|_| rng.gen_range(0.0..period - num_spikes as f64))
                .collect::<Vec<_>>();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); // Sorting safely, treating NaN as equal

            firing_times.push(
                (0..num_spikes)
                    .map(|n| {
                        if n > 0 {
                            (n as f64 + times[n - 1] + start) % period
                        } else {
                            start
                        }
                    })
                    .collect(),
            );
        }

        PeriodicSpikeTrain::build(firing_times, period)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_compute_num_spikes_weights() {
        // test weights for a few cases
        assert!(compute_num_spikes_weights(1.0, 1000.0)
            .into_iter()
            .zip([1.0])
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(compute_num_spikes_weights(1.001, 1000.0)
            .into_iter()
            .zip([0.001, 1.0])
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(compute_num_spikes_weights(5.0, 0.5)
            .into_iter()
            .zip([0.4, 1.0, 0.75, 0.166667, 0.005208])
            .all(|(a, b)| (a - b).abs() < 1e-6));

        // test all weights are positive
        assert!(
            (compute_num_spikes_weights(100.0, 0.1)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                >= 0.0)
        );
        assert!(
            (compute_num_spikes_weights(100.0, 1.0)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                >= 0.0)
        );
        assert!(
            (compute_num_spikes_weights(10000.0, 1.0)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                >= 0.0)
        );

        // test length of output
        assert_eq!(compute_num_spikes_weights(100.1, 0.1).len(), 101);
        assert_eq!(compute_num_spikes_weights(100.0, 0.1).len(), 100);
        assert_eq!(compute_num_spikes_weights(99.9, 0.1).len(), 100);
    }

    #[test]
    fn test_new_random_periodic_spike_train() {
        let mut rng = StdRng::seed_from_u64(42);

        let spike_train = PeriodicSpikeTrain::new_random(10.0, 1000.0, 100, &mut rng).unwrap();
        assert!(spike_train
            .firing_times()
            .iter()
            .all(|times| times.len() < 10));

        let min_iter = spike_train.firing_times().iter().map(|times| {
            times
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        });
        let max_iter = spike_train.firing_times().iter().map(|times| {
            times
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        });
        assert!(min_iter
            .zip(max_iter)
            .all(|(min_t, max_t)| max_t - min_t < 10.0));
    }
}
