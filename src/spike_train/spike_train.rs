use rand::distributions::WeightedIndex;
use rand::Rng;

// enum SpikeTrain {
//     FIRSpikeTrain,
//     PeriodicSpikeTrain,
// }

#[derive(Debug, PartialEq)]
pub struct SpikeTrain {
    firing_times: Vec<Vec<f64>>,
    start: f64,
    duration: f64,
}

impl SpikeTrain {
    pub fn build(
        firing_times: Vec<Vec<f64>>,
        start: f64,
        duration: f64,
    ) -> Result<Self, &'static str> {
        // sort the firing times in each channel
        let mut firing_times = firing_times;
        for times in firing_times.iter_mut() {
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if times.iter().any(|&t| t < start || t >= start + duration) {
                return Err("Firing times must be within the interval [start, start + duration).");
            }
        }

        Ok(SpikeTrain {
            firing_times,
            start,
            duration,
        })
    }
}

#[derive(Debug, PartialEq)]
pub struct PeriodicSpikeTrain {
    firing_times: Vec<Vec<f64>>,
    period: f64,
}

impl PeriodicSpikeTrain {
    fn build(firing_times: Vec<Vec<f64>>, period: f64) -> Result<Self, &'static str> {
        if period <= 0.0 {
            return Err("Period must be positive.");
        }

        Ok(PeriodicSpikeTrain {
            firing_times,
            period,
        })
    }

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

    pub fn random<R: Rng>(
        period: f64,
        firing_rate: f64,
        num_channels: usize,
        rng: &mut R,
    ) -> Result<Self, &'static str> {
        let mut firing_times = Vec::new();

        let num_spikes_dist =
            WeightedIndex::new(&Self::compute_num_spikes_weights(period, firing_rate)).unwrap();

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

    pub fn firing_times(&self, c: usize) -> &[f64] {
        &self.firing_times[c][..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_compute_num_spikes_weights() {
        println!(
            "{:?}",
            PeriodicSpikeTrain::compute_num_spikes_weights(5.0, 0.5)
        );

        // test weights for a few cases
        assert!(PeriodicSpikeTrain::compute_num_spikes_weights(1.0, 1000.0)
            .into_iter()
            .zip([1.0])
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(
            PeriodicSpikeTrain::compute_num_spikes_weights(1.001, 1000.0)
                .into_iter()
                .zip([0.001, 1.0])
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
        assert!(PeriodicSpikeTrain::compute_num_spikes_weights(5.0, 0.5)
            .into_iter()
            .zip([0.4, 1.0, 0.75, 0.166667, 0.005208])
            .all(|(a, b)| (a - b).abs() < 1e-6));

        // test all weights are positive
        assert!(
            (PeriodicSpikeTrain::compute_num_spikes_weights(100.0, 0.1)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                >= 0.0)
        );
        assert!(
            (PeriodicSpikeTrain::compute_num_spikes_weights(100.0, 1.0)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                >= 0.0)
        );
        assert!(
            (PeriodicSpikeTrain::compute_num_spikes_weights(10000.0, 1.0)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                >= 0.0)
        );

        // test length of output
        assert_eq!(
            PeriodicSpikeTrain::compute_num_spikes_weights(100.1, 0.1).len(),
            101
        );
        assert_eq!(
            PeriodicSpikeTrain::compute_num_spikes_weights(100.0, 0.1).len(),
            100
        );
        assert_eq!(
            PeriodicSpikeTrain::compute_num_spikes_weights(99.9, 0.1).len(),
            100
        );
    }

    #[test]
    fn test_spike_train_build() {
        // sort the firing times in each channel
        assert_eq!(
            SpikeTrain::build(vec![vec![1.0, 3.0, 2.0], vec![2.0, 3.1], vec![3.9, 0.1]], 0.0, 4.0)
                .unwrap()
                .firing_times,
                vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.1], vec![0.1, 3.9]]
        );

        // reject firing times outside the interval
        assert_eq!(
            SpikeTrain::build(vec![vec![0.0, 1.0, 2.0]], 0.0, 2.0),
            Err("Firing times must be within the interval [start, start + duration).")
        );
    }

    #[test]
    fn test_periodic_spike_train_build() {
        assert_eq!(
            PeriodicSpikeTrain::build(vec![vec![0.0, 1.0, 2.0], vec![10.5, 2.5, 1.5]], 4.0)
                .unwrap()
                .firing_times,
                vec![vec![0.0, 1.0, 2.0], vec![10.5, 2.5, 1.5]]
        );
        assert_eq!(
            PeriodicSpikeTrain::build(vec![vec![0.0, 1.0, 2.0], vec![10.5, 2.5, 1.5]], -4.0),
            Err("Period must be positive.")
        );
    }

    #[test]
    fn test_periodic_spike_train_random() {
        let mut rng = StdRng::seed_from_u64(42);

        let spike_train = PeriodicSpikeTrain::random(10.0, 1000.0, 1, &mut rng).unwrap();
        assert!(spike_train.firing_times(0).len() < 10);

        let min_t = spike_train
            .firing_times(0)
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_t = spike_train
            .firing_times(0)
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(max_t - min_t < 10.0);
    }
}
