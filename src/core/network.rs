//! Network-related module.
use core::f64;
use itertools::{izip, Itertools};
use lazy_static::lazy_static;
use log;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::neuron::{Input, InputSpikeTrain, Neuron};
use crate::core::optim::{Objective, TimeTemplate};
use crate::core::utils::TimeInterval;
use crate::error::SNNError;

lazy_static! {
    /// Minimum number of neurons to parallelize the computation.
    pub static ref MIN_NEURONS_PAR: usize = {
        std::env::var("RUSTY_SNN_MIN_NEURONS_PAR")
            .unwrap_or("50".to_string())
            .parse::<usize>()
            .unwrap_or_else(|_| {
                log::warn!("Invalid value for RUSTY_SNN_MIN_NEURONS_PAR. Using default value 50.");
                50
            })
    };
}

/// A connection between two neurons.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// The ID of the neuron producing spikes.
    pub source_id: usize,
    /// The ID of the neuron receiving spikes.
    pub target_id: usize,
    /// The weight of the connection. Positive for excitatory, negative for inhibitory.
    pub weight: f64,
    /// The delay of the connection.
    pub delay: f64,
}

impl Connection {
    /// Create a new connection between two neurons.
    pub fn new(source_id: usize, target_id: usize, weight: f64, delay: f64) -> Self {
        Connection {
            source_id,
            target_id,
            weight,
            delay,
        }
    }

    /// Returns an iterator over randomly generated real delays.
    fn rand_delays(
        lim_delays: (f64, f64),
        seed: u64,
    ) -> Result<rand_distr::Iter<Uniform<f64>, StdRng, f64>, SNNError> {
        if lim_delays.0 < 0.0 {
            return Err(SNNError::InvalidParameter(
                "Connection delay must be non-negative".to_string(),
            ));
        }
        let delay_rng = StdRng::seed_from_u64(seed);
        let delay_dist = Uniform::new(lim_delays.0, lim_delays.1).map_err(|e| {
            SNNError::InvalidParameter(format!("Invalid delay distribution: {}", e))
        })?;
        Ok(delay_dist.sample_iter(delay_rng))
    }

    /// Returns an iterator over randomly generated integer IDs.
    fn rand_ids(
        lim_ids: (usize, usize),
        seed: u64,
    ) -> Result<rand_distr::Iter<Uniform<usize>, StdRng, usize>, SNNError> {
        let id_rng = StdRng::seed_from_u64(seed);
        let id_dist = Uniform::new(lim_ids.0, lim_ids.1)
            .map_err(|e| SNNError::InvalidParameter(format!("Invalid id distribution: {}", e)))?;
        Ok(id_dist.sample_iter(id_rng))
    }

    /// Generate a collection of connections between neurons.
    fn generate_connections(
        source_ids: impl Iterator<Item = usize>,
        target_ids: impl Iterator<Item = usize>,
        weights: impl Iterator<Item = f64>,
        delays: impl Iterator<Item = f64>,
    ) -> Vec<Connection> {
        izip!(source_ids, target_ids, weights, delays)
            .map(|(source_id, target_id, weight, delay)| {
                Connection::new(source_id, target_id, weight, delay)
            })
            .collect()
    }

    /// A random collection of connections between neurons, where every neurons has the same number of inputs.
    pub fn rand_fin(
        num_inputs: usize,
        lim_source_ids: (usize, usize),
        lim_target_ids: (usize, usize),
        lim_delays: (f64, f64),
        seed: u64,
    ) -> Result<Vec<Connection>, SNNError> {
        let source_ids = Self::rand_ids(lim_source_ids, seed)?;
        let target_ids = (lim_target_ids.0..lim_target_ids.1)
            .flat_map(|target_id| std::iter::repeat(target_id).take(num_inputs));
        let weights = std::iter::repeat(0.0);
        let delays = Self::rand_delays(lim_delays, seed)?;

        let connections: Vec<Connection> =
            Self::generate_connections(source_ids, target_ids, weights, delays);

        Ok(connections)
    }

    /// A random collection of connections between neurons, where every neurons has the same number of outputs.
    pub fn rand_fout(
        num_outputs: usize,
        lim_source_ids: (usize, usize),
        lim_target_ids: (usize, usize),
        lim_delays: (f64, f64),
        seed: u64,
    ) -> Result<Vec<Connection>, SNNError> {
        let source_ids = (lim_source_ids.0..lim_source_ids.1)
            .flat_map(|source_id| std::iter::repeat(source_id).take(num_outputs));
        let target_ids = Self::rand_ids(lim_target_ids, seed)?;
        let weights = std::iter::repeat(0.0);
        let delays = Self::rand_delays(lim_delays, seed)?;

        let connections: Vec<Connection> =
            Self::generate_connections(source_ids, target_ids, weights, delays);

        Ok(connections)
    }

    /// A random collection of connections between neurons, where every neurons has the same number of inputs and outputs.
    #[allow(unused_variables)]
    pub fn rand_fin_fout(
        num_neurons: usize,
        num_inputs_outputs: usize,
        lim_delays: (f64, f64),
        seed: u64,
    ) -> Result<Vec<Connection>, SNNError> {
        // FIXME: Implement random connection generation with fixed in/out degree
        // Should follow similar pattern to rand_fin and rand_fout but ensure
        // each neuron has exactly num_inputs_outputs inputs and outputs
        todo!()
    }

    /// A random collection of connections between neurons, where every neuron is connected to every other neuron (including itself).
    #[allow(unused_variables)]
    pub fn rand_fc(
        lim_neurons_ids: (usize, usize),
        lim_delays: (f64, f64),
        seed: u64,
    ) -> Result<Vec<Connection>, SNNError> {
        if lim_delays.0 < 0.0 {
            return Err(SNNError::InvalidParameter(
                "Connection delay must be non-negative".to_string(),
            ));
        }
        let mut delay_rng = StdRng::seed_from_u64(seed);
        let delay_dist = Uniform::new_inclusive(lim_delays.0, lim_delays.1).map_err(|e| {
            SNNError::InvalidParameter(format!("Invalid delay distribution: {}", e))
        })?;

        let connections: Vec<Connection> = izip!(
            (lim_neurons_ids.0..lim_neurons_ids.1)
                .cartesian_product(lim_neurons_ids.0..lim_neurons_ids.1),
            delay_dist.sample_iter(&mut delay_rng)
        )
        .map(|((source_id, target_id), delay)| {
            Connection::new(source_id, target_id, f64::NAN, delay)
        })
        .collect();

        Ok(connections)
    }
}

/// A network of spiking neurons.
pub trait Network {
    type InputSpikeTrain: InputSpikeTrain + std::fmt::Debug;
    type Neuron: Neuron<InputSpikeTrain = Self::InputSpikeTrain>;

    /// A reference to a specific neuron in the network.
    /// Returns `None` if the neuron is not found.
    fn neuron_ref(&self, neuron_id: usize) -> Option<&Self::Neuron>;

    /// A mutable reference to a specific neuron in the network.
    /// Returns `None` if the neuron is not found.
    fn neuron_mut(&mut self, neuron_id: usize) -> Option<&mut Self::Neuron>;

    /// An iterator over the neurons in the network.
    fn neurons_iter(&self) -> impl Iterator<Item = &Self::Neuron> + '_;

    /// A mutable iterator over the neurons in the network.
    fn neurons_iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Neuron> + '_;

    /// A parallel iterator over the neurons in the network.
    fn neurons_par_iter(&self) -> impl ParallelIterator<Item = &Self::Neuron> + '_;

    /// A mutable parallel iterator over the neurons in the network.
    fn neurons_par_iter_mut(&mut self) -> impl ParallelIterator<Item = &mut Self::Neuron> + '_;

    /// The number of neurons in the network.
    fn num_neurons(&self) -> usize {
        self.neurons_iter().count()
    }

    /// Clear the firing times of the neurons in the network.
    fn clear_ftimes(&mut self) {
        self.neurons_iter_mut().for_each(|neuron| {
            neuron.ftimes_mut().clear();
        });
    }

    /// Set the firing times of the neurons in the network from a multi-channel spike train.
    fn init_ftimes(&mut self, times: &Vec<Vec<f64>>) {
        self.clear_ftimes();
        self.extend_ftimes(times);
    }

    /// Extend the firing times of the neurons in the network from a multi-channel spike train.
    fn extend_ftimes(&mut self, times: &Vec<Vec<f64>>) {
        self.neurons_iter_mut().for_each(|neuron| {
            let id = neuron.id();
            neuron.ftimes_mut().extend(times[id].clone());
        });
    }

    /// The multi-channel spike train of the network.
    fn spike_train_ref(&self) -> Vec<&[f64]> {
        self.neurons_iter()
            .map(|neuron| neuron.ftimes_ref().as_slice())
            .collect()
    }

    /// The multi-channel spike train of the network.
    fn spike_train_clone(&self) -> Vec<Vec<f64>> {
        self.neurons_iter()
            .map(|neuron| neuron.ftimes_ref().clone())
            .collect()
    }

    fn connections_ref(&self) -> Vec<&[Input]> {
        self.neurons_iter().map(|neuron| neuron.inputs()).collect()
    }

    /// The number of connections in the network.
    fn num_connections(&self) -> usize {
        self.neurons_iter().map(|neuron| neuron.num_inputs()).sum()
    }

    /// Add a connection to the network.
    fn push_connection(&mut self, connection: &Connection) {
        if let Some(neuron) = self.neuron_mut(connection.target_id) {
            neuron.push_input(connection.source_id, connection.weight, connection.delay)
        }
    }

    /// Returns the minimum delay in spike propagation between each pair of neurons in the network.
    /// The delay from a neuron to itself is zero (due to its refractory period).
    fn min_delay_from_to(&mut self, source_id: usize, target_id: usize) -> f64;

    /// Optimize the synaptic weights of the network to memorize the given spike trains.
    fn memorize_cyclic(
        &mut self,
        spike_trains: &Vec<&Vec<Vec<f64>>>,
        periods: &Vec<f64>,
        lim_weights: (f64, f64),
        max_level: f64,
        min_slope: f64,
        half_width: f64,
        objective: Objective,
    ) -> Result<(), SNNError> {
        if self.num_neurons() >= *MIN_NEURONS_PAR {
            self.neurons_par_iter_mut().try_for_each(|neuron| {
                if neuron.num_inputs() > 0 {
                    let mut time_templates: Vec<TimeTemplate> =
                        Vec::with_capacity(spike_trains.len());
                    let mut input_spike_trains: Vec<Self::InputSpikeTrain> =
                        Vec::with_capacity(spike_trains.len());

                    for (ftimes, period) in spike_trains.iter().zip(periods.iter()) {
                        let time_template = TimeTemplate::new_cyclic_from(
                            &ftimes[neuron.id()],
                            half_width,
                            *period,
                        );
                        let input_spike_train = Self::InputSpikeTrain::new_cyclic_from(
                            neuron.inputs(),
                            *ftimes,
                            *period,
                            &time_template.interval,
                        );
                        time_templates.push(time_template);
                        input_spike_trains.push(input_spike_train);
                    }

                    neuron.memorize(
                        time_templates,
                        input_spike_trains,
                        lim_weights,
                        max_level,
                        min_slope,
                        objective,
                    )
                } else {
                    log::trace!("Neuron {} has no synaptic weights to optimize", neuron.id());
                    Ok(())
                }
            })
        } else {
            self.neurons_iter_mut().try_for_each(|neuron| {
                if neuron.num_inputs() > 0 {
                    let mut time_templates: Vec<TimeTemplate> =
                        Vec::with_capacity(spike_trains.len());
                    let mut input_spike_trains: Vec<Self::InputSpikeTrain> =
                        Vec::with_capacity(spike_trains.len());

                    for (ftimes, period) in spike_trains.iter().zip(periods.iter()) {
                        let time_template = TimeTemplate::new_cyclic_from(
                            &ftimes[neuron.id()],
                            half_width,
                            *period,
                        );
                        let input_spike_train = Self::InputSpikeTrain::new_cyclic_from(
                            neuron.inputs(),
                            *ftimes,
                            *period,
                            &time_template.interval,
                        );
                        time_templates.push(time_template);
                        input_spike_trains.push(input_spike_train);
                    }

                    neuron.memorize(
                        time_templates,
                        input_spike_trains,
                        lim_weights,
                        max_level,
                        min_slope,
                        objective,
                    )
                } else {
                    log::trace!("Neuron {} has no synaptic weights to optimize", neuron.id());
                    Ok(())
                }
            })
        }
    }

    /// Collect a list of potential spikes from all neurons after the specified time.
    fn collect_spikes(&self, min_time: f64, max_time: f64) -> Vec<(usize, Option<f64>)> {
        self.neurons_iter()
            .map(|neuron| match neuron.next_spike(min_time) {
                Some(time) => {
                    if time <= max_time {
                        (neuron.id(), Some(time))
                    } else {
                        (neuron.id(), None)
                    }
                }
                None => (neuron.id(), None),
            })
            .collect::<Vec<(usize, Option<f64>)>>()
    }

    /// Collect a list of potential spikes from all neurons after the specified time.
    /// Computations are done in parallel.
    fn collect_spikes_par(&self, min_time: f64, max_time: f64) -> Vec<(usize, Option<f64>)> {
        self.neurons_par_iter()
            .map(|neuron| match neuron.next_spike(min_time) {
                Some(time) => {
                    if time <= max_time {
                        (neuron.id(), Some(time))
                    } else {
                        (neuron.id(), None)
                    }
                }
                None => (neuron.id(), None),
            })
            .collect::<Vec<(usize, Option<f64>)>>()
        // spikes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    fn accept_spikes(
        &mut self,
        spikes: &[(usize, Option<f64>)],
        accepted_spikes: &mut Vec<Option<f64>>,
    ) {
        spikes.iter().for_each(|(id, time)| match time {
            None => {
                accepted_spikes[*id] = None;
            }
            Some(time) => {
                if spikes
                    .iter()
                    .filter_map(|(other_id, other_time)| match other_time {
                        Some(other_time) => Some((other_id, other_time)),
                        None => None,
                    })
                    .all(|(other_id, other_time)| {
                        *time <= other_time + self.min_delay_from_to(*other_id, *id)
                    })
                {
                    accepted_spikes[*id] = Some(*time)
                } else {
                    accepted_spikes[*id] = None
                }
            }
        });
    }

    /// Simulate the network activity on the specified time interval.
    fn run(
        &mut self,
        time_interval: &TimeInterval,
        threshold_noise: f64,
        seed: u64,
    ) -> Result<(), SNNError> {
        if let TimeInterval::Closed { start, end } = time_interval {
            // Initialize the neurons with
            // 1. The random number generators and the threshold noise
            // 2. The input spike trains
            let network_spike_train = self.spike_train_clone();
            if self.num_neurons() > *MIN_NEURONS_PAR {
                self.neurons_par_iter_mut().for_each(|neuron| {
                    neuron.init_threshold_sampler(threshold_noise, seed);
                    neuron.sample_threshold();
                    neuron.init_input_spike_train(&network_spike_train);
                })
            } else {
                self.neurons_iter_mut().for_each(|neuron| {
                    neuron.init_threshold_sampler(threshold_noise, seed);
                    neuron.sample_threshold();
                    neuron.init_input_spike_train(&network_spike_train);
                })
            };

            let mut time = *start;
            let mut new_spikes: Vec<Option<f64>> = vec![None; self.num_neurons()];

            while time < *end {
                // Collect the candidate next spikes from all neurons, using parallel computation if the number of neurons is large
                // The candidate spikes are sorted by neuron ID
                let candidate_new_spikes = if self.num_neurons() > *MIN_NEURONS_PAR {
                    self.collect_spikes_par(time, *end)
                } else {
                    self.collect_spikes(time, *end)
                };

                // Accept as many spikes as possible at the current time
                self.accept_spikes(&candidate_new_spikes, &mut new_spikes);

                // Get the lowest among all accepted spikes if any, or end the simulation
                time = new_spikes
                    .iter()
                    .filter_map(|x| *x)
                    .min_by(|a, b| a.partial_cmp(&b).unwrap())
                    .unwrap_or(*end);

                // Update the neuron internal state: 1) fire the accepted spikes and 2) update the input spike trains
                // let new_spike_train = MultiChannelSpikeTrain::new_from(new_times);
                if self.num_neurons() > *MIN_NEURONS_PAR {
                    self.neurons_par_iter_mut().for_each(|neuron| {
                        if let Some(ft) = new_spikes[neuron.id()] {
                            neuron.fire(ft);
                        }
                        neuron.drain_input_spike_train(time);
                        neuron.receive_spikes(&new_spikes);
                    })
                } else {
                    self.neurons_iter_mut().for_each(|neuron| {
                        if let Some(ft) = new_spikes[neuron.id()] {
                            neuron.fire(ft);
                        }
                        neuron.drain_input_spike_train(time);
                        neuron.receive_spikes(&new_spikes);
                    })
                };
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection() {
        let connection = Connection::new(0, 1, 1.0, 1.0);
        assert_eq!(connection.source_id, 0);
        assert_eq!(connection.target_id, 1);
        assert_eq!(connection.weight, 1.0);
        assert_eq!(connection.delay, 1.0);
    }

    #[test]
    fn test_connection_rand_fin() {
        let connections = Connection::rand_fin(2, (0, 2), (0, 2), (0.0, 1.0), 0).unwrap();
        assert_eq!(connections.len(), 4);
        assert_eq!(connections.iter().filter(|c| c.target_id == 0).count(), 2);
        assert_eq!(connections.iter().filter(|c| c.target_id == 1).count(), 2);
        assert_eq!(connections.iter().filter(|c| c.target_id == 2).count(), 0);
    }

    #[test]
    fn test_connection_rand_fout() {
        let connections = Connection::rand_fout(2, (0, 2), (0, 2), (0.0, 1.0), 0).unwrap();
        assert_eq!(connections.len(), 4);
        assert_eq!(connections.iter().filter(|c| c.source_id == 0).count(), 2);
        assert_eq!(connections.iter().filter(|c| c.source_id == 1).count(), 2);
        assert_eq!(connections.iter().filter(|c| c.source_id == 2).count(), 0);
    }

    #[test]
    fn test_connection_rand_fc() {
        let connections = Connection::rand_fc((0, 2), (0.0, 1.0), 0).unwrap();
        assert_eq!(connections.len(), 4);
        assert_eq!(
            connections
                .iter()
                .filter(|c| (c.source_id == 0) & (c.target_id == 0))
                .count(),
            1
        );
        assert_eq!(
            connections
                .iter()
                .filter(|c| (c.source_id == 0) & (c.target_id == 1))
                .count(),
            1
        );
        assert_eq!(
            connections
                .iter()
                .filter(|c| (c.source_id == 1) & (c.target_id == 0))
                .count(),
            1
        );
        assert_eq!(
            connections
                .iter()
                .filter(|c| (c.source_id == 1) & (c.target_id == 1))
                .count(),
            1
        );
    }
}
