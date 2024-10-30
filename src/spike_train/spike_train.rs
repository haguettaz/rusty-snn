//! A spike train is a multi-channel sequence of firing times separated by a minimum refractory period.
//! 
//! This module provides the following spike train types:
//! - `SpikeTrain`: A generic spike train with arbitrary firing times
//! - `PeriodicSpikeTrain`: A periodic spike train
//! 

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
    end: f64,
}

impl SpikeTrain {
    pub fn build(
        firing_times: Vec<Vec<f64>>,
        start: f64,
        end: f64,
    ) -> Result<Self, &'static str> {
        // sort the firing times in each channel
        let mut firing_times = firing_times;
        for times in firing_times.iter_mut() {
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if times.iter().any(|&t| t < start || t >= end) {
                return Err("Firing times must be within the interval [start, end).");
            }
        }

        Ok(SpikeTrain {
            firing_times,
            start,
            end,
        })
    }

    pub fn firing_times(&self) -> &[Vec<f64>] {
        &self.firing_times[..]
    }

    pub fn is_empty(&self) -> bool {
        self.firing_times.iter().all(|times| times.is_empty())
    }

    pub fn num_channels(&self) -> usize {
        self.firing_times.len()
    }

    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

#[derive(Debug, PartialEq)]
pub struct PeriodicSpikeTrain {
    firing_times: Vec<Vec<f64>>,
    period: f64,
}

impl PeriodicSpikeTrain {
    pub fn build(firing_times: Vec<Vec<f64>>, period: f64) -> Result<Self, &'static str> {
        if period <= 0.0 {
            return Err("Period must be positive.");
        }

        Ok(PeriodicSpikeTrain {
            firing_times,
            period,
        })
    }

    pub fn firing_times(&self) -> &[Vec<f64>] {
        &self.firing_times[..]
    }

    pub fn num_channels(&self) -> usize {
        self.firing_times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.firing_times.iter().all(|times| times.is_empty())
    }

    pub fn period(&self) -> f64 {
        self.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_spike_train_build() {
        // sort the firing times in each channel
        assert_eq!(
            SpikeTrain::build(
                vec![vec![1.0, 3.0, 2.0], vec![2.0, 3.1], vec![3.9, 0.1]],
                0.0,
                4.0
            )
            .unwrap()
            .firing_times,
            vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.1], vec![0.1, 3.9]]
        );

        // reject firing times outside the interval
        assert_eq!(
            SpikeTrain::build(vec![vec![0.0, 1.0, 2.0]], 0.0, 2.0),
            Err("Firing times must be within the interval [start, end).")
        );
    }

    #[test]
    fn test_spike_train_num_channels() {
        assert_eq!(
            SpikeTrain::build(
                vec![vec![1.0, 3.0, 2.0], vec![2.0, 3.1], vec![3.9, 0.1]],
                0.0,
                10.0
            )
            .unwrap()
            .num_channels(),
            3
        );

        assert_eq!(
            SpikeTrain::build(vec![], 0.0, 1.0).unwrap().num_channels(),
            0
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
    fn test_periodic_spike_train_num_channels() {
        assert_eq!(
            PeriodicSpikeTrain::build(
                vec![vec![1.0, 3.0, 2.0], vec![2.0, 3.1], vec![3.9, 0.1]],
                10.0
            )
            .unwrap()
            .num_channels(),
            3
        );

        assert_eq!(
            PeriodicSpikeTrain::build(vec![], 10.0)
                .unwrap()
                .num_channels(),
            0
        );
    }
}
