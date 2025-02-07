//! Network-related structures and traits.
use core::f64;
use log;
// use rand::distributions::{Distribution, Uniform};
use rand::RngCore;
use rand_distr::{Distribution,Uniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::neuron::{Input, InputSpikeTrain, Neuron};
use crate::core::optim::{Objective, TimeTemplate};
use crate::core::spikes::{MultiChannelCyclicSpikeTrain, MultiChannelSpikeTrain, Spike};
use crate::core::utils::TimeInterval;
use crate::error::SNNError;

/// Minimum number of neurons to parallelize the computation.
pub const MIN_NEURONS_PAR: usize = 10;

/// A connection between two neurons.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// The ID of the neuron producing spikes.
    pub source_id: usize,
    /// The ID of the neuron receiving spikes.
    pub target_id: usize,
    /// The weight of the synapse along which the spikes are transmitted.
    pub weight: f64,
    /// The delay of the synapse along which the spikes are transmitted.
    pub delay: f64,
}

impl Connection {
    pub fn new(source_id: usize, target_id: usize, weight: f64, delay: f64) -> Self {
        Connection {
            source_id,
            target_id,
            weight,
            delay,
        }
    }

    pub fn rand_fin<R: RngCore>(
        num_neurons: usize,
        num_inputs: usize,
        lim_delays: (f64, f64),
        rng: &mut R,
    ) -> Result<Vec<Connection>, SNNError> {
        let (min_delay, max_delay) = lim_delays;
        if min_delay < 0.0 {
            return Err(SNNError::InvalidParameters(
                "Connection delay must be non-negative".to_string(),
            ));
        }
        let delay_dist = Uniform::new_inclusive(min_delay, max_delay).map_err(|e| {
            SNNError::InvalidParameters(format!("Invalid delay distribution: {}", e))
        })?;
        let source_dist = Uniform::new(0, num_neurons).map_err(|e| {
            SNNError::InvalidParameters(format!("Invalid source distribution: {}", e))
        })?;

        let connections: Vec<Connection> = (0..num_neurons * num_inputs)
            .map(|i| {
                Connection::new(
                    source_dist.sample(rng),
                    i % num_neurons,
                    f64::NAN,
                    delay_dist.sample(rng),
                )
            })
            .collect();

        Ok(connections)
    }

    pub fn rand_fout<R: RngCore>(
        num_neurons: usize,
        num_outputs: usize,
        lim_delays: (f64, f64),
        rng: &mut R,
    ) -> Result<Vec<Connection>, SNNError> {
        let (min_delay, max_delay) = lim_delays;
        if min_delay < 0.0 {
            return Err(SNNError::InvalidParameters(
                "Connection delay must be non-negative".to_string(),
            ));
        }
        let delay_dist = Uniform::new_inclusive(min_delay, max_delay).map_err(|e| {
            SNNError::InvalidParameters(format!("Invalid delay distribution: {}", e))
        })?;
        let target_dist = Uniform::new(0, num_neurons).map_err(|e| {
            SNNError::InvalidParameters(format!("Invalid target distribution: {}", e))
        })?;

        let connections: Vec<Connection> = (0..num_neurons * num_outputs)
            .map(|i| {
                Connection::new(
                    i % num_neurons,
                    target_dist.sample(rng),
                    f64::NAN,
                    delay_dist.sample(rng),
                )
            })
            .collect();

        Ok(connections)
    }

    #[allow(unused_variables)]
    pub fn rand_fin_fout<R: RngCore>(
        num_neurons: usize,
        num_inputs_outputs: usize,
        lim_delays: (f64, f64),
        rng: &mut R,
    ) -> Result<Vec<Connection>, SNNError> {
        todo!();
    }
}

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

    /// Set the firing times of the neurons in the network from a spike train.
    fn init_firing_times(&mut self, spike_train: &MultiChannelSpikeTrain) {
        self.neurons_iter_mut().for_each(|neuron| {
            if let Some(firing_times) = spike_train.get(neuron.id()) {
                neuron.firing_times_mut().clear();
                neuron.firing_times_mut().extend(firing_times);
            }
        });
    }

    /// The multi-channel spike train of the network.
    fn spike_train(&self) -> MultiChannelSpikeTrain {
        let spike_train: Vec<Vec<f64>> = self
            .neurons_iter()
            .map(|neuron| neuron.firing_times_ref().clone())
            .collect();
        MultiChannelSpikeTrain { spike_train }
    }

    fn connections(&self) -> Vec<&Vec<Input>> {
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

    fn memorize_cyclic(
        &mut self,
        spike_trains: &Vec<&MultiChannelCyclicSpikeTrain>,
        lim_weights: (f64, f64),
        max_level: f64,
        min_slope: f64,
        half_width: f64,
        objective: Objective,
    ) -> Result<(), SNNError> {
        if self.num_neurons() >= MIN_NEURONS_PAR {
            self.neurons_par_iter_mut().try_for_each(|neuron| {
                log::info!("Optimizing neuron {}", neuron.id());
                let mut time_templates: Vec<TimeTemplate> = Vec::with_capacity(spike_trains.len());
                let mut input_spike_trains: Vec<Self::InputSpikeTrain> = Vec::with_capacity(spike_trains.len());

                for spike_train in spike_trains.iter() {
                    let firing_times = spike_train.get(neuron.id()).cloned().unwrap_or_default();
                    let time_template = TimeTemplate::new_cyclic_from(
                        &firing_times,
                        half_width,
                        spike_train.period,
                    );
                    let input_spike_train = Self::InputSpikeTrain::new_cyclic_from(
                        neuron.inputs(),
                        spike_train,
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
                )?;
                log::info!("Neuron {} successfully optimized", neuron.id());
                Ok(())
            })
        } else {
            self.neurons_iter_mut().try_for_each(|neuron| {
                log::info!("Optimizing neuron {}", neuron.id());
                let mut time_templates: Vec<TimeTemplate> = Vec::new();
                let mut input_spike_trains: Vec<Self::InputSpikeTrain> = Vec::new();

                for spike_train in spike_trains.iter() {
                    let firing_times = spike_train.get(neuron.id()).cloned().unwrap_or(Vec::new());
                    let time_template = TimeTemplate::new_cyclic_from(
                        &firing_times,
                        half_width,
                        spike_train.period,
                    );
                    let input_spike_train = Self::InputSpikeTrain::new_cyclic_from(
                        neuron.inputs(),
                        spike_train,
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
                )?;
                log::info!("Neuron {} successfully optimized", neuron.id());
                Ok(())
            })
        }
    }

    // fn memorize(
    //     &mut self,
    //     spike_trains: &Vec<MultiOutputSpikeTrain>,
    //     lim_weights: (f64, f64),
    //     max_level: f64,
    //     min_slope: f64,
    //     half_width: f64,
    //     objective: Objective,
    // ) -> Result<(), SNNError> {
    //     todo!()
    // }

    /// Run the simulation of the network for the specified time interval.
    fn run(
        &mut self,
        time_interval: &TimeInterval,
        threshold_noise: f64,
    ) -> Result<(), SNNError> {
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
            let spike_train = self.spike_train();
            if self.num_neurons() > MIN_NEURONS_PAR {
                self.neurons_par_iter_mut().for_each(|neuron| {
                    // let new_input_spike_train =
                    //     Self::InputSpikeTrain::new_from(neuron.inputs(), &new_spike_train);
                    // neuron.clear_input_spike_train();
                    // neuron.extend_input_spike_train(new_input_spike_train);
                    neuron.init_threshold_sampler(threshold_noise);
                    neuron.sample_threshold();
                    neuron.init_input_spike_train(&spike_train);
                })
            } else {
                self.neurons_iter_mut().for_each(|neuron| {
                    // let new_input_spike_train =
                    //     Self::InputSpikeTrain::new_from(neuron.inputs(), &new_spike_train);
                    // neuron.clear_input_spike_train();
                    // neuron.extend_input_spike_train(new_input_spike_train);
                    neuron.init_threshold_sampler(threshold_noise);
                    neuron.sample_threshold();
                    neuron.init_input_spike_train(&spike_train);
                })
            };

            // // Init the normal distribution for threshold noise
            // let normal = Normal::new(0.0, threshold_noise).unwrap();

            let mut time = *start;
            while time < *end {
                // Collect the candidate next spikes from all neurons, using parallel computation if the number of neurons is large
                let candidate_new_spikes = if self.num_neurons() > MIN_NEURONS_PAR {
                    self.neurons_par_iter()
                        .filter_map(|neuron| neuron.next_spike(time))
                        .collect::<Vec<Spike>>()
                } else {
                    self.neurons_iter()
                        .filter_map(|neuron| neuron.next_spike(time))
                        .collect::<Vec<Spike>>()
                };

                // Accept as many spikes as possible at the current time
                let new_spikes = candidate_new_spikes
                    .iter()
                    .filter(|spike| {
                        candidate_new_spikes.iter().all(|other_spike| {
                            spike.time
                                <= other_spike.time
                                    + min_delays[other_spike.neuron_id][spike.neuron_id]
                        })
                    })
                    .cloned()
                    .collect::<Vec<Spike>>();

                // Get the lowest among all accepted spikes if any or end the simulation
                match new_spikes.iter().min_by(|a, b| a.partial_cmp(&b).unwrap()) {
                    Some(spike) => time = spike.time,
                    None => {
                        log::info!("Network activity has ceased...");
                        return Ok(());
                    }
                }

                // Update the neuron internal state: 1) fire the accepted spikes and 2) update the input spike trains
                let new_spike_train = MultiChannelSpikeTrain::new_from(new_spikes);
                if self.num_neurons() > MIN_NEURONS_PAR {
                    self.neurons_par_iter_mut().try_for_each(|neuron| {
                        new_spike_train.get(neuron.id()).map(|spikes| {
                            spikes.iter().for_each(|spike| neuron.fire(*spike))
                        });
                        neuron.update_input_spikes(time, &new_spike_train)
                    })
                } else {
                    self.neurons_iter_mut().try_for_each(|neuron| {
                        new_spike_train.get(neuron.id()).map(|spikes| {
                            spikes.iter().for_each(|spike| neuron.fire(*spike))
                        });
                        neuron.update_input_spikes(time, &new_spike_train)
                    })
                }?;


                // // Fire the accepted spikes
                // new_spikes.iter().for_each(|spike| {
                //     self.neuron_mut(spike.neuron_id)
                //         .unwrap()
                //         .fire(spike.time, normal.sample(rng))
                // });

                // Update the input spike trains for each neuron
                // let new_spike_train = MultiChannelSpikeTrain::new_from(new_spikes);

                // for neuron in self.neurons_iter() {
                //     let new_input_spike_train =
                //         InputSpikeTrain::new_from(neuron.inputs_ref(), &new_spike_train);
                //     input_spike_trains[neuron.id()]
                //         .drain_before(time - neuron.input_kernel_support_length(MIN_CONTRIBUTION));
                //     input_spike_trains[neuron.id()].merge_with(new_input_spike_train);

                // Check if it's time to log progress
                if time - last_log_time >= log_interval {
                    let progress = ((time - start) / total_duration) * 100.0;
                    log::debug!(
                        "Simulation progress: {:.2}% (Time: {:.2}/{:.2})",
                        progress, time, end
                    );
                    last_log_time = time;
                }
            }
        }

        log::info!("Simulation completed successfully!");
        Ok(())
    }
}
