//! Network-related module.
use core::f64;
use itertools::izip;
use log;
// use rand::distributions::{Distribution, Uniform}
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::neuron::{Input, InputSpikeTrain, Neuron};
use crate::core::optim::{Objective, TimeTemplate};
// use crate::core::spikes::MultiChannelCyclicSpikeTrain;
use crate::core::utils::TimeInterval;
use crate::error::SNNError;

/// Minimum number of neurons to parallelize the computation.
pub const MIN_NEURONS_PAR: usize = 10000;

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

    /// A random collection of connections between neurons, where every neurons has the same number of inputs.
    pub fn rand_fin(
        num_inputs: usize,
        lim_source_ids: (usize, usize),
        lim_target_ids: (usize, usize),
        lim_delays: (f64, f64),
        seed: u64,
    ) -> Result<Vec<Connection>, SNNError> {
        if lim_delays.0 < 0.0 {
            return Err(SNNError::InvalidParameters(
                "Connection delay must be non-negative".to_string(),
            ));
        }
        let mut delay_rng = StdRng::seed_from_u64(seed);
        let delay_dist = Uniform::new_inclusive(lim_delays.0, lim_delays.1).map_err(|e| {
            SNNError::InvalidParameters(format!("Invalid delay distribution: {}", e))
        })?;

        let mut source_rng = StdRng::seed_from_u64(seed);
        let source_dist = Uniform::new(lim_source_ids.0, lim_source_ids.1).map_err(|e| {
            SNNError::InvalidParameters(format!("Invalid source distribution: {}", e))
        })?;

        let connections: Vec<Connection> = izip!(
            source_dist.sample_iter(&mut source_rng),
            (lim_target_ids.0..lim_target_ids.1)
                .flat_map(|target_id| std::iter::repeat(target_id).take(num_inputs)),
            delay_dist.sample_iter(&mut delay_rng)
        )
        .map(|(source_id, target_id, delay)| Connection::new(source_id, target_id, f64::NAN, delay))
        .collect();

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
        if lim_delays.0 < 0.0 {
            return Err(SNNError::InvalidParameters(
                "Connection delay must be non-negative".to_string(),
            ));
        }
        let mut delay_rng = StdRng::seed_from_u64(seed);
        let delay_dist = Uniform::new_inclusive(lim_delays.0, lim_delays.1).map_err(|e| {
            SNNError::InvalidParameters(format!("Invalid delay distribution: {}", e))
        })?;

        let mut target_rng = StdRng::seed_from_u64(seed);
        let target_dist = Uniform::new(lim_target_ids.0, lim_target_ids.1).map_err(|e| {
            SNNError::InvalidParameters(format!("Invalid target distribution: {}", e))
        })?;

        let connections: Vec<Connection> = izip!(
            (lim_source_ids.0..lim_source_ids.1)
                .flat_map(|source_id| std::iter::repeat(source_id).take(num_outputs)),
            target_dist.sample_iter(&mut target_rng),
            delay_dist.sample_iter(&mut delay_rng)
        )
        .map(|(source_id, target_id, delay)| Connection::new(source_id, target_id, f64::NAN, delay))
        .collect();

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
        todo!();
    }

    /// A random collection of connections between neurons, where every neuron is connected to every other neuron (including itself).
    #[allow(unused_variables)]
    pub fn rand_fc(
        lim_neurons_ids: (usize, usize),
        lim_delays: (f64, f64),
        seed: u64,
    ) -> Result<Vec<Connection>, SNNError> {
        if lim_delays.0 < 0.0 {
            return Err(SNNError::InvalidParameters(
                "Connection delay must be non-negative".to_string(),
            ));
        }
        let mut delay_rng = StdRng::seed_from_u64(seed);
        let delay_dist = Uniform::new_inclusive(lim_delays.0, lim_delays.1).map_err(|e| {
            SNNError::InvalidParameters(format!("Invalid delay distribution: {}", e))
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

    fn connections_ref(&self) -> Vec<&Vec<Input>> {
        self.neurons_iter().map(|neuron| neuron.inputs()).collect()
    }

    /// The number of connections in the network.
    fn num_connections(&self) -> usize {
        self.neurons_iter().map(|neuron| neuron.num_inputs()).sum()
    }

    /// Add a connection to the network.
    fn add_connection(&mut self, connection: &Connection) {
        if let Some(neuron) = self.neuron_mut(connection.target_id) {
            neuron.add_input(connection.source_id, connection.weight, connection.delay)
        }
    }

    /// Returns the minimum delay in spike propagation between each pair of neurons in the network.
    /// The delay from a neuron to itself is zero (due to its refractory period).
    /// Otherwise, the delay is the minimum delay between all possible paths connecting the two neurons.
    /// They are computed using the [Floyd-Warshall algorithm](https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm).
    fn min_delays(&self) -> Vec<Vec<f64>> {
        let mut min_delays = self
            .neurons_iter()
            .map(|target_neuron| {
                self.neurons_iter()
                    .map(|source_neuron| {
                        if target_neuron.id() == source_neuron.id() {
                            0.0
                        } else {
                            target_neuron
                                .inputs_iter()
                                .filter(|input| input.source_id == source_neuron.id())
                                .map(|input| input.delay)
                                .min_by(|a, b| a.partial_cmp(b).expect("Invalid delay"))
                                .unwrap_or(f64::INFINITY)
                        }
                    })
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();

        for inter_neuron in self.neurons_iter() {
            for target_neuron in self.neurons_iter() {
                for source_neuron in self.neurons_iter() {
                    let source_target_delay = min_delays[target_neuron.id()][source_neuron.id()];
                    let source_inter_target_delay = min_delays[inter_neuron.id()]
                        [source_neuron.id()]
                        + min_delays[target_neuron.id()][inter_neuron.id()];
                    if source_target_delay > source_inter_target_delay {
                        min_delays[target_neuron.id()][source_neuron.id()] =
                            source_inter_target_delay;
                    }
                }
            }
        }

        min_delays
    }

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
        if self.num_neurons() >= MIN_NEURONS_PAR {
            self.neurons_par_iter_mut().try_for_each(|neuron| {
                if neuron.num_inputs() > 0 {
                    log::info!("Optimizing neuron {}", neuron.id());
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
                    log::info!("Neuron {} has no synaptic weights to optimize", neuron.id());
                    Ok(())
                }
            })
        } else {
            self.neurons_iter_mut().try_for_each(|neuron| {
                if neuron.num_inputs() > 0 {
                    log::info!("Optimizing neuron {}", neuron.id());
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
                    log::info!("Neuron {} has no synaptic weights to optimize", neuron.id());
                    Ok(())
                }
            })
        }
    }

    /// Simulate the network activity on the specified time interval.
    fn run(&mut self, time_interval: &TimeInterval, threshold_noise: f64) -> Result<(), SNNError> {
        if let TimeInterval::Closed { start, end } = time_interval {
            log::info!("Starting simulation...");

            // For logging purposes
            let total_duration = end - start;
            let mut last_log_time = *start;
            let log_interval = total_duration / 100.0;

            // Compute the minimum delays between each pair of neurons using Floyd-Warshall algorithm (O(L^3))
            // This is done only once for the whole simulation
            // It is used to accept the largest number of spikes at each time step
            let min_delays = self.min_delays();
            log::info!("Minimum propagation delays computed successfully!");

            // Initialize the neurons with
            // 1. The random number generators and the threshold noise
            // 2. The input spike trains
            let network_spike_train = self.spike_train_clone();
            if self.num_neurons() > MIN_NEURONS_PAR {
                self.neurons_par_iter_mut().for_each(|neuron| {
                    neuron.init_threshold_sampler(threshold_noise);
                    neuron.sample_threshold();
                    neuron.init_input_spike_train(&network_spike_train);
                })
            } else {
                self.neurons_iter_mut().for_each(|neuron| {
                    neuron.init_threshold_sampler(threshold_noise);
                    neuron.sample_threshold();
                    neuron.init_input_spike_train(&network_spike_train);
                })
            };

            let mut time = *start;
            while time < *end {
                // Collect the candidate next spikes from all neurons, using parallel computation if the number of neurons is large
                let mut candidate_new_times = if self.num_neurons() > MIN_NEURONS_PAR {
                    self.neurons_par_iter()
                        .map(|neuron| match neuron.next_spike(time) {
                            Some(time) => (neuron.id(), Some(time)),
                            None => (neuron.id(), None),
                        })
                        .collect::<Vec<(usize, Option<f64>)>>()
                } else {
                    self.neurons_iter()
                        .map(|neuron| match neuron.next_spike(time) {
                            Some(time) => (neuron.id(), Some(time)),
                            None => (neuron.id(), None),
                        })
                        .collect::<Vec<(usize, Option<f64>)>>()
                };
                candidate_new_times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                // Accept as many spikes as possible at the current time and sort the spikes by neuron ID
                let new_times = candidate_new_times
                    .iter()
                    .map(|(id, time)| match time {
                        Some(time) => {
                            if candidate_new_times
                                .iter()
                                .filter_map(|(other_id, other_time)| match other_time {
                                    Some(other_time) => Some((other_id, other_time)),
                                    None => None,
                                })
                                .all(|(other_id, other_time)| {
                                    *time <= other_time + min_delays[*other_id][*id]
                                })
                            {
                                Some(*time)
                            } else {
                                None
                            }
                        }
                        None => None,
                    })
                    .collect::<Vec<Option<f64>>>();

                // Get the lowest among all accepted spikes if any or end the simulation
                match new_times
                    .iter()
                    .filter_map(|x| *x)
                    .min_by(|a, b| a.partial_cmp(&b).unwrap())
                {
                    Some(min_time) => time = min_time,
                    None => {
                        log::info!("Network activity has ceased...");
                        return Ok(());
                    }
                }

                // Update the neuron internal state: 1) fire the accepted spikes and 2) update the input spike trains
                // let new_spike_train = MultiChannelSpikeTrain::new_from(new_times);
                if self.num_neurons() > MIN_NEURONS_PAR {
                    self.neurons_par_iter_mut().for_each(|neuron| {
                        if let Some(ft) = new_times[neuron.id()] {
                            neuron.fire(ft);
                        }
                        neuron.drain_input_spike_train(time);
                        neuron.receive_spikes(&new_times);
                    })
                } else {
                    self.neurons_iter_mut().for_each(|neuron| {
                        if let Some(ft) = new_times[neuron.id()] {
                            neuron.fire(ft);
                        }
                        neuron.drain_input_spike_train(time);
                        neuron.receive_spikes(&new_times);
                    })
                };

                // Check if it's time to log progress
                if time - last_log_time >= log_interval {
                    let progress = ((time - start) / total_duration) * 100.0;
                    log::debug!(
                        "Simulation progress: {:.2}% (Time: {:.2}/{:.2})",
                        progress,
                        time,
                        end
                    );
                    last_log_time = time;
                }
            }
        }

        log::info!("Simulation completed successfully!");
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
