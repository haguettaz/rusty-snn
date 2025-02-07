//! Alpha neuron related implementations.
use core::f64;
use itertools::Itertools;
use lambert_w::{lambert_w0, lambert_wm1};
use log;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::E;

use crate::core::neuron::MIN_INPUT_VALUE;
use crate::core::neuron::{Input, InputSpike, InputSpikeTrain, Neuron};
use crate::core::spikes::{MultiChannelCyclicSpikeTrain, MultiChannelSpikeTrain};
use crate::core::utils::{TimeInterval, TimeIntervalUnion, TimeValuePair};
use crate::core::FIRING_THRESHOLD;
use crate::error::SNNError;

const RIGHT_LIMIT: f64 = 1e-12;

/// A spiking neuron with alpha-shaped synaptic kernels.
#[derive(Debug, Clone, Serialize)]
pub struct AlphaNeuron {
    // The neuron ID.
    id: usize,
    // The neuron inputs.
    inputs: Vec<Input>,
    // The neuron firing threshold.
    threshold: f64,
    /// The neuron firing times.
    firing_times: Vec<f64>,
    /// The input spike train.
    input_spike_train: AlphaInputSpikeTrain,
    /// The threshold noise sampler.
    #[serde(skip)]
    threshold_sampler: Normal<f64>,
    /// The random number generator.
    #[serde(skip)]
    rng: ChaCha8Rng,
}

impl<'de> Deserialize<'de> for AlphaNeuron {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct AlphaNeuronData {
            id: usize,
            inputs: Vec<Input>,
            threshold: f64,
            firing_times: Vec<f64>,
            input_spike_train: AlphaInputSpikeTrain,
        }

        let data = AlphaNeuronData::deserialize(deserializer)?;
        Ok(AlphaNeuron {
            id: data.id,
            inputs: data.inputs,
            threshold: data.threshold,
            firing_times: data.firing_times,
            input_spike_train: data.input_spike_train,
            threshold_sampler: Normal::new(FIRING_THRESHOLD, 0.0).unwrap(),
            rng: ChaCha8Rng::seed_from_u64(0),
        })
    }
}

impl AlphaNeuron {
    pub fn new_empty(id: usize, seed: u64) -> Self {
        AlphaNeuron {
            id,
            inputs: vec![],
            threshold: FIRING_THRESHOLD,
            firing_times: vec![],
            input_spike_train: AlphaInputSpikeTrain::new_empty(),
            threshold_sampler: Normal::new(FIRING_THRESHOLD, 0.0).unwrap(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn new_from(id: usize, inputs: Vec<Input>, threshold: f64, seed: u64) -> Self {
        AlphaNeuron {
            id,
            inputs,
            threshold,
            firing_times: vec![],
            input_spike_train: AlphaInputSpikeTrain::new_empty(),
            threshold_sampler: Normal::new(FIRING_THRESHOLD, 0.0).unwrap(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl Neuron for AlphaNeuron {
    type InputSpike = AlphaInputSpike;
    type InputSpikeTrain = AlphaInputSpikeTrain;

    fn id(&self) -> usize {
        self.id
    }

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn init_threshold_sampler(&mut self, sigma: f64) {
        self.threshold_sampler = Normal::new(FIRING_THRESHOLD, sigma).unwrap();
    }

    fn sample_threshold(&mut self) {
        self.threshold = self.threshold_sampler.sample(&mut self.rng);
    }

    // fn set_threshold(&mut self, threshold: f64) {
    //     self.threshold = threshold;
    // }

    fn firing_times_ref(&self) -> &Vec<f64> {
        self.firing_times.as_ref()
    }

    fn firing_times_mut(&mut self) -> &mut Vec<f64> {
        self.firing_times.as_mut()
    }

    /// A reference to the vector of inputs of the neuron
    fn inputs(&self) -> &Vec<Input> {
        &self.inputs
    }

    /// A mutable reference to the vector of inputs of the neuron
    fn inputs_mut(&mut self) -> &mut Vec<Input> {
        &mut self.inputs
    }

    /// Get a reference to the vector of inputs of the neuron
    fn inputs_iter(&self) -> impl Iterator<Item = &Input> + '_ {
        self.inputs.iter()
    }

    /// Get a reference to the inputs of the neuron.
    fn inputs_iter_mut(&mut self) -> impl Iterator<Item = &mut Input> + '_ {
        self.inputs.iter_mut()
    }

    /// A reference to the ith input.
    fn input_ref(&self, i: usize) -> Option<&Input> {
        self.inputs.get(i)
    }

    /// A mutable reference to a specific input.
    fn input_mut(&mut self, i: usize) -> Option<&mut Input> {
        self.inputs.get_mut(i)
    }

    /// Get a reference to the inputs of the neuron.
    fn input_spike_train(&self) -> &Self::InputSpikeTrain {
        &self.input_spike_train
    }

    /// Get a reference to the inputs of the neuron.
    fn input_spike_train_mut(&mut self) -> &mut Self::InputSpikeTrain {
        &mut self.input_spike_train
    }

    // fn clear_input_spike_train(&mut self) {
    //     self.input_spike_train = AlphaInputSpikeTrain::new_empty();
    // }

    // fn extend_input_spike_train(&mut self, new_input_spike_train: Self::InputSpikeTrain) {
    //     self.input_spike_train.merge(new_input_spike_train);
    // }

    /// Initialize the input spikes of the neuron from the provided spike train.
    fn init_input_spike_train(&mut self, spike_train: &MultiChannelSpikeTrain) {
        self.input_spike_train = Self::InputSpikeTrain::new_from(self.inputs(), spike_train);
    }

    /// Update the input spikes of the neuron from the provided spike train.
    /// The input spikes are updated by removing all input spikes which are irrelevant from the provided time.
    /// Raise an error if the provided time is smaller than any of the input spikes, in which case, merge is meaningless.
    fn update_input_spikes(
        &mut self,
        time: f64,
        spike_train: &MultiChannelSpikeTrain,
    ) -> Result<(), SNNError> {
        if spike_train
            .iter()
            .flat_map(|times| times.iter())
            .any(|&t| t < time)
        {
            return Err(SNNError::InvalidParameters(
                "Time must be smaller than any provided spikes.".to_string(),
            ));
        }
        let pos = self.input_spike_train.find_before(time).unwrap_or(0);
        self.input_spike_train.input_spikes.drain(..pos);
        let new_input_spike_train = Self::InputSpikeTrain::new_from(self.inputs(), spike_train);
        self.input_spike_train.merge(new_input_spike_train);
        Ok(())
    }
}

/// An input spike through an alpha-shaped synapse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaInputSpike {
    /// The ID of the input along which the spike is received.
    pub input_id: usize,
    /// The time at which the spike is received.
    pub time: f64,
    /// The weight of the synapse along which the spike is received.
    pub weight: f64,
    /// The coefficient sum_{j <= i} w_j * exp(1-(s_i - s_j))
    pub a: f64,
    /// The coefficient sum_{j <= i} w_j * s_j * exp(1-(s_i - s_j))
    pub b: f64,
}

impl PartialEq for AlphaInputSpike {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.weight == other.weight && self.input_id == other.input_id
    }
}

impl PartialOrd for AlphaInputSpike {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.time.partial_cmp(&other.time)
    }
}

impl InputSpike for AlphaInputSpike {
    fn new(input_id: usize, time: f64, weight: f64) -> Self {
        AlphaInputSpike {
            input_id,
            time,
            weight,
            a: weight * E,
            b: weight * time * E,
        }
    }

    fn time(&self) -> f64 {
        self.time
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn input_id(&self) -> usize {
        self.input_id
    }

    fn kernel(&self, dt: f64) -> f64 {
        if dt > 0.0 {
            dt * (1.0 - dt).exp()
        } else {
            0.0
        }
    }

    fn kernel_deriv(&self, dt: f64) -> f64 {
        if dt > 0.0 {
            (1.0 - dt) * (1.0 - dt).exp()
        } else {
            0.0
        }
    }
}

/// An input spike train through alpha-shaped synapses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaInputSpikeTrain {
    /// The (sorted) collection of input spikes.
    pub input_spikes: Vec<AlphaInputSpike>,
}

impl AlphaInputSpikeTrain {
    /// Returns a new input spike train.
    pub fn new(mut input_spikes: Vec<AlphaInputSpike>) -> Self {
        if !input_spikes.is_empty() {
            input_spikes.sort_by(|input_spike_1, input_spike_2| {
                input_spike_1.partial_cmp(&input_spike_2).unwrap()
            });

            input_spikes[0].a = input_spikes[0].weight * E;
            input_spikes[0].b = input_spikes[0].weight * input_spikes[0].time * E;

            for i in 1..input_spikes.len() {
                input_spikes[i].a = input_spikes[i - 1].a
                    * (input_spikes[i - 1].time - input_spikes[i].time).exp()
                    + input_spikes[i].weight * E;
                input_spikes[i].b = input_spikes[i - 1].b
                    * (input_spikes[i - 1].time - input_spikes[i].time).exp()
                    + input_spikes[i].weight * input_spikes[i].time * E;
            }
        }
        AlphaInputSpikeTrain { input_spikes }
    }

    /// Find the position of the last input_spike (strictly) before time, if any.
    /// Otherwise, return None.
    pub fn find_before(&self, time: f64) -> Option<usize> {
        match self
            .input_spikes
            .binary_search_by(|input_spike| input_spike.time.partial_cmp(&time).unwrap())
        {
            Ok(pos) => {
                let pos = self.input_spikes[..=pos]
                    .iter()
                    .rev()
                    .enumerate()
                    .take_while_inclusive(|(_, input_spike)| input_spike.time >= time)
                    .last()
                    .map(|(i, _)| pos - i)
                    .unwrap();
                if self.input_spikes[pos].time < time {
                    Some(pos)
                } else {
                    None
                }
            }
            Err(pos) => {
                if pos > 0 {
                    Some(pos - 1)
                } else {
                    None
                }
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.input_spikes.is_empty()
    }

    // fn is_sorted(&self) -> bool {
    //     self.input_spikes
    //         .iter()
    //         .tuple_windows()
    //         .all(|(input_spike_1, input_spike_2)| input_spike_1 <= input_spike_2)
    // }
}

impl InputSpikeTrain for AlphaInputSpikeTrain {
    /// An input spike through an alpha-shaped synapse.
    type InputSpike = AlphaInputSpike;

    /// Returns a new empty input spike train.
    fn new_empty() -> Self {
        AlphaInputSpikeTrain {
            input_spikes: vec![],
        }
    }

    /// Returns a new input spike train from the provided inputs and spike train.
    fn new_from(inputs: &Vec<Input>, spike_train: &MultiChannelSpikeTrain) -> Self {
        let mut input_spikes: Vec<Self::InputSpike> = inputs
            .iter()
            .enumerate()
            .filter_map(|(input_id, input)| {
                spike_train
                    .get(input.source_id)
                    .map(|times| (input_id, input, times))
            })
            .flat_map(|(input_id, input, times)| {
                times.iter().map(move |time| {
                    AlphaInputSpike::new(input_id, *time + input.delay, input.weight)
                })
            })
            .collect();
        // {
        //     Ok(output_spikes_iter) => output_spikes_iter.map(|time| AlphaInputSpike::new(input.input_id, time + input.delay, input.weight)),
        //     Err(_) => iter::empty(),
        // }

        // spike_train
        //     .spikes
        //     .iter()
        //     .flat_map(|spike| {
        //         inputs
        //             .iter()
        //             .filter(|input| input.source_id == spike.source_id)
        //             .map(|input| {
        //                 AlphaInputSpike::new(input.input_id, spike.time + input.delay, input.weight)
        //             })
        //     })
        //     .collect();

        input_spikes.sort_by(|input_spike_1, input_spike_2| {
            input_spike_1.partial_cmp(&input_spike_2).unwrap()
        });

        for i in 1..input_spikes.len() {
            input_spikes[i].a = input_spikes[i - 1].a
                * (input_spikes[i - 1].time - input_spikes[i].time).exp()
                + input_spikes[i].weight * E;
            input_spikes[i].b = input_spikes[i - 1].b
                * (input_spikes[i - 1].time - input_spikes[i].time).exp()
                + input_spikes[i].weight * input_spikes[i].time * E;
        }

        AlphaInputSpikeTrain { input_spikes }
    }

    /// Returns a new input spike train from the provided inputs and cyclic spike train.
    fn new_cyclic_from(
        inputs: &Vec<Input>,
        spike_train: &MultiChannelCyclicSpikeTrain,
        interval: &TimeInterval,
    ) -> Self {
        match interval {
            TimeInterval::Empty => AlphaInputSpikeTrain {
                input_spikes: vec![],
            },
            TimeInterval::Closed { start, end } => {
                let min_time = *start + lambert_wm1(-MIN_INPUT_VALUE / E);
                let mut input_spikes: Vec<Self::InputSpike> = inputs
                    .iter()
                    .enumerate()
                    .filter_map(|(input_id, input)| {
                        spike_train
                            .get(input.source_id)
                            .map(|times| (input_id, input, times))
                    })
                    .flat_map(|(input_id, input, times)| {
                        times
                            .iter()
                            .map(|time| {
                                let time = time + input.delay;
                                time - ((time - end) / spike_train.period).ceil()
                                    * spike_train.period
                            })
                            .flat_map(move |time| {
                                (0..).map_while(move |i| {
                                    let input_time = time - i as f64 * spike_train.period;
                                    if input_time >= min_time {
                                        Some(AlphaInputSpike::new(
                                            input_id,
                                            input_time,
                                            input.weight,
                                        ))
                                    } else {
                                        None
                                    }
                                })
                            })
                    })
                    .collect();

                input_spikes.sort_by(|input_spike_1, input_spike_2| {
                    input_spike_1.partial_cmp(&input_spike_2).unwrap()
                });

                for i in 1..input_spikes.len() {
                    input_spikes[i].a = input_spikes[i - 1].a
                        * (input_spikes[i - 1].time - input_spikes[i].time).exp()
                        + input_spikes[i].weight * E;
                    input_spikes[i].b = input_spikes[i - 1].b
                        * (input_spikes[i - 1].time - input_spikes[i].time).exp()
                        + input_spikes[i].weight * input_spikes[i].time * E;
                }

                AlphaInputSpikeTrain { input_spikes }
            }
        }
    }

    /// Update the input spike train from the provided inputs.
    /// Sort inputs by source_id?
    fn update_from(&mut self, inputs: &Vec<Input>) -> Result<(), SNNError> {
        // for input_spike in self.input_spikes.iter_mut() {
        //     if let Some(input) = inputs
        //         .iter()
        //         .find(|(input_id, input)| *input_id == input_spike.input_id)
        //     {
        //         input_spike.weight = input.weight;
        //         input_spike.a = input_spike.weight * E;
        //         input_spike.b = input_spike.weight * input_spike.time * E;
        //     }
        // }
        self.input_spikes.iter_mut().try_for_each(|input_spike| {
            let input = inputs
                .get(input_spike.input_id)
                .ok_or(SNNError::OutOfBounds(
                    "Input spike train and inputs are not aligned".to_string(),
                ))?;
            input_spike.weight = input.weight;
            input_spike.a = input_spike.weight * E;
            input_spike.b = input_spike.weight * input_spike.time * E;
            Ok(())
        })?;

        for i in 1..self.input_spikes.len() {
            self.input_spikes[i].a = self.input_spikes[i - 1].a
                * (self.input_spikes[i - 1].time - self.input_spikes[i].time).exp()
                + self.input_spikes[i].weight * E;
            self.input_spikes[i].b = self.input_spikes[i - 1].b
                * (self.input_spikes[i - 1].time - self.input_spikes[i].time).exp()
                + self.input_spikes[i].weight * self.input_spikes[i].time * E;
        }

        Ok(())
    }

    /// Insert a new input spike while preserving the order, and return the insertion index.
    /// If there are multiple input spikes at the same time, the new input spike is inserted after them.
    fn insert(&mut self, new_input_spike: Self::InputSpike) -> usize {
        let pos = match self.input_spikes.binary_search_by(|input_spike| {
            input_spike.time.partial_cmp(&new_input_spike.time).unwrap()
        }) {
            Ok(pos) => self.input_spikes[pos..]
                .iter()
                .enumerate()
                .take_while(|(_, spike)| spike.time == new_input_spike.time)
                .map(|(i, _)| pos + i + 1)
                .last()
                .unwrap(),
            Err(pos) => pos,
        };

        self.input_spikes.insert(pos, new_input_spike);
        pos
    }

    /// Merge input spikes to the existing ones.
    fn merge(&mut self, input_spike_train: Self) {
        if !input_spike_train.is_empty() {
            // Insert the new input spikes and collect the insertion indices.
            let mut first_insert: usize = input_spike_train
                .input_spikes
                .into_iter()
                .map(|input_spike| self.insert(input_spike))
                .min()
                .unwrap();

            // Initialize the coefficients a and b of the first new input_spike, if it is the new first input spike of the train.
            if first_insert == 0 {
                self.input_spikes[0].a = self.input_spikes[0].weight * E;
                self.input_spikes[0].b =
                    self.input_spikes[0].weight * self.input_spikes[0].time * E;
                first_insert += 1;
            }

            // Update the coefficients a and b of the input spikes.
            // Note that the input spikes before the first new input spike do not need to be updated.
            for id in first_insert..self.input_spikes.len() {
                self.input_spikes[id].a = self.input_spikes[id - 1].a
                    * (self.input_spikes[id - 1].time - self.input_spikes[id].time).exp()
                    + self.input_spikes[id].weight * E;
                self.input_spikes[id].b = self.input_spikes[id - 1].b
                    * (self.input_spikes[id - 1].time - self.input_spikes[id].time).exp()
                    + self.input_spikes[id].weight * self.input_spikes[id].time * E;
            }
        }
    }

    /// An iterator over the input spikes.
    fn input_spikes_iter(&self) -> impl Iterator<Item = &Self::InputSpike> + '_ {
        self.input_spikes.iter()
    }

    /// A mutable iterator over the input spikes.
    fn input_spikes_iter_mut(&mut self) -> impl Iterator<Item = &mut Self::InputSpike> + '_ {
        self.input_spikes.iter_mut()
    }

    /// Returns the k greatest maximizers of the potential in the windows.
    fn max_potential_in_windows(
        &self,
        windows: &TimeIntervalUnion,
        k: usize,
    ) -> Vec<TimeValuePair<f64>> {
        let mut pairs: Vec<TimeValuePair<f64>> = vec![];

        // Compute the minimum potential derivative in the continuous parts, i.e., in between input spikes (exclusive)
        self.input_spikes
            .iter()
            .map(Some)
            .chain(std::iter::once(None))
            .tuple_windows()
            .for_each(|(input_spike, next_input_spike)| {
                let input_spike = input_spike.unwrap();

                // On the continuous part, the maximum potential is maximized if its second derivative is negative, i.e., if a > 0.
                if input_spike.a > 0.0 {
                    let time = 1.0 + input_spike.b / input_spike.a;
                    let value =
                        (time * input_spike.a - input_spike.b) * (input_spike.time - time).exp();
                    match next_input_spike {
                        Some(next_input_spike) => {
                            if time > input_spike.time
                                && time < next_input_spike.time
                                && windows.contains(time)
                            {
                                pairs.push(TimeValuePair { time, value });
                            }
                        }
                        None => {
                            if time > input_spike.time && windows.contains(time) {
                                pairs.push(TimeValuePair { time, value });
                            }
                        }
                    }
                }
            });

        // Compute the minimum potential derivative in the discontinuous parts, i.e., at the input spikes
        self.input_spikes.iter().for_each(|input_spike| {
            if windows.contains(input_spike.time) {
                let time = input_spike.time;
                let value = input_spike.time * input_spike.a - input_spike.b;
                pairs.push(TimeValuePair { time, value });
            }
        });

        // self.input_spikes
        //     .iter()
        //     .map(Some)
        //     .chain(std::iter::once(None))
        //     .tuple_windows()
        //     .for_each(|(input_spike, next_input_spike)| {
        //         let input_spike = input_spike.unwrap();
        //         if windows.contains(input_spike.time) {
        //             let time = input_spike.time;
        //             let value = input_spike.time * input_spike.a - input_spike.b;
        //             pairs.push(TimeValuePair { time, value });
        //         }

        //         // On the continuous part, the potential is maximized if the second derivative is negative.
        //         if input_spike.a > 0.0 {
        //             let time = 1.0 + input_spike.b / input_spike.a;
        //             let value =
        //                 (time * input_spike.a - input_spike.b) * (input_spike.time - time).exp();
        //             match next_input_spike {
        //                 Some(next_input_spike) => {
        //                     if time >= input_spike.time
        //                         && time <= next_input_spike.time
        //                         && windows.contains(time)
        //                     {
        //                         pairs.push(TimeValuePair { time, value });
        //                     }
        //                 }
        //                 None => {
        //                     if time >= input_spike.time && windows.contains(time) {
        //                         pairs.push(TimeValuePair { time, value });
        //                     }
        //                 }
        //             }
        //         }
        //     });

        //     if (input_spike.time - 1.0) * input_spike.a >= input_spike.b {
        //         let time = input_spike.time;
        //         let value = input_spike.time * input_spike.a - input_spike.b;
        //         if windows.contains(time) {
        //             Some(TimeValuePair { time, value })
        //         } else {
        //             None
        //         }
        //     } else {
        //         if input_spike.a < 0.0 {
        //             None
        //         } else if input_spike.a == 0.0 && input_spike.b == 0.0 {
        //             let time = input_spike.time;
        //             let value = 0.0;
        //             if time >= input_spike.time && windows.contains(time) {
        //                 Some(TimeValuePair { time, value })
        //             } else {
        //                 None
        //             }
        //         } else {
        //             let time = 1.0 + input_spike.b / input_spike.a;
        //             let value = input_spike.a * (input_spike.time - time).exp();
        //             match next_input_spike {
        //                 Some(next_input_spike) => {
        //                     if time >= input_spike.time
        //                         && time <= next_input_spike.time
        //                         && windows.contains(time)
        //                     {
        //                         Some(TimeValuePair { time, value })
        //                     } else {
        //                         None
        //                     }
        //                 }
        //                 None => {
        //                     if time >= input_spike.time && windows.contains(time) {
        //                         Some(TimeValuePair { time, value })
        //                     } else {
        //                         None
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // })

        // Add the windows's borders to the list of potential maximizers.
        for window in windows.iter() {
            if let TimeInterval::Closed { start, end } = window {
                {
                    pairs.push(TimeValuePair {
                        time: *start,
                        value: self.potential(*start),
                    });
                    pairs.push(TimeValuePair {
                        time: *end,
                        value: self.potential(*end),
                    });
                }
            }
        }

        // Sort the pairs by value in descending order, make sure the pairs are unique, and take the k greatest maximizers.
        pairs.sort_by(|pair_1, pair_2| pair_2.partial_cmp(&pair_1).unwrap());
        pairs.dedup();
        pairs.into_iter().take(k).collect()
    }

    /// Returns the k lowest minimizers of the potential derivative in the windows.
    fn min_potential_deriv_in_windows(
        &self,
        windows: &TimeIntervalUnion,
        k: usize,
    ) -> Vec<TimeValuePair<f64>> {
        let mut pairs: Vec<TimeValuePair<f64>> = vec![];

        // Compute the minimum potential derivative in the continuous parts, i.e., in between input spikes (exclusive)
        self.input_spikes
            .iter()
            .map(Some)
            .chain(std::iter::once(None))
            .tuple_windows()
            .for_each(|(input_spike, next_input_spike)| {
                let input_spike = input_spike.unwrap();

                // On the continuous part, the potential derivative is minimized if its third derivative is positive, i.e., if a > 0.
                if input_spike.a > 0.0 {
                    let time = 2.0 + input_spike.b / input_spike.a;
                    let value = -input_spike.a * (input_spike.time - time).exp();
                    match next_input_spike {
                        Some(next_input_spike) => {
                            if time > input_spike.time
                                && time < next_input_spike.time
                                && windows.contains(time)
                            {
                                pairs.push(TimeValuePair { time, value });
                            }
                        }
                        None => {
                            if time > input_spike.time && windows.contains(time) {
                                pairs.push(TimeValuePair { time, value });
                            }
                        }
                    }
                }
            });

        // Compute the minimum potential derivative in the discontinuous parts, i.e., at the input spikes
        self.input_spikes.iter().for_each(|input_spike| {
            if windows.contains(input_spike.time) {
                let time = input_spike.time;
                let value = -((input_spike.time - 1.0) * input_spike.a + input_spike.weight * E
                    - input_spike.b);
                pairs.push(TimeValuePair { time, value });
            }
        });

        // Compute the minimum potential derivative in the right-limit discontinuity parts, i.e., to the right of the input spikes
        self.input_spikes
            .iter()
            .map(Some)
            .chain(std::iter::once(None))
            .tuple_windows()
            .for_each(|(input_spike, next_input_spike)| {
                let input_spike = input_spike.unwrap();
                let time = input_spike.time + RIGHT_LIMIT;
                match next_input_spike {
                    Some(next_input_spike) => {
                        if time < next_input_spike.time && windows.contains(time) {
                            let value = -((time - 1.0) * input_spike.a - input_spike.b)
                                * (input_spike.time - time).exp();
                            pairs.push(TimeValuePair { time, value });
                        }
                    }
                    None => {
                        if windows.contains(time) {
                            let value = -((time - 1.0) * input_spike.a - input_spike.b)
                                * (input_spike.time - time).exp();
                            pairs.push(TimeValuePair { time, value });
                        }
                    }
                }
            });

        // Add the windows's borders to the list of potential maximizers.
        for window in windows.iter() {
            if let TimeInterval::Closed { start, end } = window {
                {
                    pairs.push(TimeValuePair {
                        time: *start,
                        value: self.potential_deriv(*start),
                    });
                    pairs.push(TimeValuePair {
                        time: *end,
                        value: self.potential_deriv(*end),
                    });
                }
            }
        }

        // Sort the pairs by value in descending order, make sure the pairs are unique and take the k greatest maximizers.
        pairs.sort_by(|pair_1, pair_2| pair_1.partial_cmp(&pair_2).unwrap());
        pairs.dedup();
        pairs.into_iter().take(k).collect()
    }

    /// Returns the next potential threshold crossing time after the provided time.
    fn next_potential_threshold_crossing(&self, start: f64, threshold: f64) -> Option<f64> {
        if self.potential(start) >= threshold {
            log::debug!(
                "The potential at time {} is already above the threshold.",
                start
            );
            Some(start)
        } else {
            // self.input_spikes
            let pos = self.find_before(start).unwrap_or(0);
            self.input_spikes[pos..]
                .iter()
                .map(Some)
                .chain(std::iter::once(None))
                .tuple_windows()
                .find_map(|(input_spike, next_input_spike)| match input_spike {
                    Some(input_spike) => {
                        let pred_time = if input_spike.a == 0.0 {
                            input_spike.time - (-threshold / input_spike.b).ln()
                        } else {
                            input_spike.b / input_spike.a
                                - lambert_w0(
                                    -threshold
                                        * (input_spike.b / input_spike.a - input_spike.time).exp()
                                        / input_spike.a,
                                )
                        };
                        match next_input_spike {
                            Some(next_input_spike) => {
                                if pred_time >= start
                                    && pred_time > input_spike.time
                                    && pred_time <= next_input_spike.time
                                {
                                    Some(pred_time)
                                } else {
                                    None
                                }
                            }
                            None => {
                                if pred_time >= start && pred_time > input_spike.time {
                                    Some(pred_time)
                                } else {
                                    None
                                }
                            }
                        }
                    }
                    None => None,
                })
        }
    }

    /// Compute the potential at the provided time.
    fn potential(&self, time: f64) -> f64 {
        match self.find_before(time) {
            Some(pos) => {
                (time * self.input_spikes[pos].a - self.input_spikes[pos].b)
                    * (self.input_spikes[pos].time - time).exp()
            }
            None => 0.0,
        }
    }

    /// Compute the potential derivative at the provided time.
    fn potential_deriv(&self, time: f64) -> f64 {
        match self.find_before(time) {
            Some(pos) => {
                ((1.0 - time) * self.input_spikes[pos].a + self.input_spikes[pos].b)
                    * (self.input_spikes[pos].time - time).exp()
            }
            None => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use core::{f64, panic};

    use super::*;
    use crate::core::{
        optim::{Objective, TimeTemplate},
        spikes::{MultiChannelCyclicSpikeTrain, Spike},
    };

    #[test]
    fn test_input_spike_train_new_from() {
        todo!();
    }

    #[test]
    fn test_input_spike_train_new_cyclic_from() {
        let inputs = vec![
            Input::new(0, 0.5, 1.0),
            Input::new(1, 1.0, 3.0),
            Input::new(2, 1.0, 0.5),
            Input::new(1, -0.5, 2.0),
        ];

        let spike_train = MultiChannelCyclicSpikeTrain {
            spike_train: vec![vec![0.0, 22.0], vec![10.0, 47.0], vec![34.5, 98.5]],
            period: 100.0,
        };
        let interval = TimeInterval::new(3.0, 103.0);

        let input_spike_train =
            AlphaInputSpikeTrain::new_cyclic_from(&inputs, &spike_train, &interval);

        assert_relative_eq!(input_spike_train.input_spikes[0].time, -1.0);
        assert_relative_eq!(input_spike_train.input_spikes[1].time, 1.0);
        assert_relative_eq!(input_spike_train.input_spikes[2].time, 12.0);
        assert_relative_eq!(input_spike_train.input_spikes[3].time, 13.0);
        assert_relative_eq!(input_spike_train.input_spikes[4].time, 23.0);
        assert_relative_eq!(input_spike_train.input_spikes[5].time, 35.0);
        assert_relative_eq!(input_spike_train.input_spikes[6].time, 49.0);
        assert_relative_eq!(input_spike_train.input_spikes[7].time, 50.0);
        assert_relative_eq!(input_spike_train.input_spikes[8].time, 99.0);
        assert_relative_eq!(input_spike_train.input_spikes[9].time, 101.0);
    }

    #[test]
    fn test_input_spike_train_find_before() {
        let input_spike_train = AlphaInputSpikeTrain::new_empty();
        assert_eq!(input_spike_train.find_before(0.0), None);

        let input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(0, 0.0, 1.0),
            AlphaInputSpike::new(0, 0.0, 1.0),
            AlphaInputSpike::new(0, 0.5, 1.0),
            AlphaInputSpike::new(0, 1.0, 1.0),
            AlphaInputSpike::new(0, 1.0, 1.0),
            AlphaInputSpike::new(0, 1.0, 1.0),
            AlphaInputSpike::new(0, 2.0, 1.0),
        ]);
        assert_eq!(input_spike_train.find_before(-1.0), None);
        assert_eq!(input_spike_train.find_before(0.0), None);
        assert_eq!(input_spike_train.find_before(0.5), Some(1));
        assert_eq!(input_spike_train.find_before(1.0), Some(2));
        assert_eq!(input_spike_train.find_before(1.5), Some(5));
        assert_eq!(input_spike_train.find_before(2.0), Some(5));
        assert_eq!(input_spike_train.find_before(3.0), Some(6));
    }

    #[test]
    fn test_input_spike_train_merge() {
        let mut input_spike_train = AlphaInputSpikeTrain::new_empty();

        let new_input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(0, 1.0, 0.5),
            AlphaInputSpike::new(0, 2.0, 0.5),
        ]);
        input_spike_train.merge(new_input_spike_train);
        assert_relative_eq!(input_spike_train.input_spikes[0].a, 1.3591409142295225);
        assert_relative_eq!(input_spike_train.input_spikes[0].b, 1.3591409142295225);
        assert_relative_eq!(input_spike_train.input_spikes[1].a, 1.8591409142295225);
        assert_relative_eq!(input_spike_train.input_spikes[1].b, 3.218281828459045);

        let new_input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(0, 1.5, -0.25),
            AlphaInputSpike::new(0, 0.0, 0.75),
        ]);
        input_spike_train.merge(new_input_spike_train);
        assert_relative_eq!(input_spike_train.input_spikes[0].a, 2.038711371344284);
        assert_relative_eq!(input_spike_train.input_spikes[0].b, 0.0);
        assert_relative_eq!(input_spike_train.input_spikes[1].a, 2.1091409142295223);
        assert_relative_eq!(input_spike_train.input_spikes[1].b, 1.3591409142295225);
        assert_relative_eq!(input_spike_train.input_spikes[2].a, 0.5996881730197777);
        assert_relative_eq!(input_spike_train.input_spikes[2].b, -0.19499505032207798);
        assert_relative_eq!(input_spike_train.input_spikes[3].a, 1.7228701774330721);
        assert_relative_eq!(input_spike_train.input_spikes[3].b, 2.600011351946497);

        let new_input_spike_train =
            AlphaInputSpikeTrain::new(vec![AlphaInputSpike::new(0, 1.75, -0.5)]);
        input_spike_train.merge(new_input_spike_train);
        assert_relative_eq!(input_spike_train.input_spikes[0].a, 2.038711371344284);
        assert_relative_eq!(input_spike_train.input_spikes[0].b, 0.0);
        assert_relative_eq!(input_spike_train.input_spikes[1].a, 2.1091409142295223);
        assert_relative_eq!(input_spike_train.input_spikes[1].b, 1.3591409142295225);
        assert_relative_eq!(input_spike_train.input_spikes[2].a, 0.5996881730197777);
        assert_relative_eq!(input_spike_train.input_spikes[2].b, -0.19499505032207798);
        assert_relative_eq!(input_spike_train.input_spikes[3].a, -0.8921032954830594);
        assert_relative_eq!(input_spike_train.input_spikes[3].b, -2.5303588977875466);
        assert_relative_eq!(input_spike_train.input_spikes[4].a, 0.664370169126735);
        assert_relative_eq!(input_spike_train.input_spikes[4].b, 0.7476363374104069);
    }

    #[test]
    fn test_input_spike_train_potential() {
        let input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(0, 1.0, 0.5),
            AlphaInputSpike::new(0, 2.0, 0.5),
        ]);

        assert_relative_eq!(
            input_spike_train
                .input_spikes
                .iter()
                .fold(0.0, |acc, input_spike| acc
                    + input_spike.weight
                        * input_spike.kernel(1.0 - input_spike.time)),
            input_spike_train.potential(1.0)
        );
        assert_relative_eq!(
            input_spike_train
                .input_spikes
                .iter()
                .fold(0.0, |acc, input_spike| acc
                    + input_spike.weight
                        * input_spike.kernel(2.0 - input_spike.time)),
            input_spike_train.potential(2.0)
        );
        assert_relative_eq!(
            input_spike_train
                .input_spikes
                .iter()
                .fold(0.0, |acc, input_spike| acc
                    + input_spike.weight
                        * input_spike.kernel(4.0 - input_spike.time)),
            input_spike_train.potential(4.0)
        );
        assert_relative_eq!(
            input_spike_train
                .input_spikes
                .iter()
                .fold(0.0, |acc, input_spike| acc
                    + input_spike.weight
                        * input_spike.kernel(8.0 - input_spike.time)),
            input_spike_train.potential(8.0)
        );
    }

    #[test]
    fn test_input_spike_train_potential_deriv() {
        let input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(0, 1.0, 0.5),
            AlphaInputSpike::new(0, 2.0, 0.5),
        ]);

        assert_relative_eq!(
            input_spike_train
                .input_spikes
                .iter()
                .fold(0.0, |acc, input_spike| acc
                    + input_spike.weight
                        * input_spike.kernel_deriv(1.0 - input_spike.time)),
            input_spike_train.potential_deriv(1.0)
        );
        assert_relative_eq!(
            input_spike_train
                .input_spikes
                .iter()
                .fold(0.0, |acc, input_spike| acc
                    + input_spike.weight
                        * input_spike.kernel_deriv(2.0 - input_spike.time)),
            input_spike_train.potential_deriv(2.0)
        );
        assert_relative_eq!(
            input_spike_train
                .input_spikes
                .iter()
                .fold(0.0, |acc, input_spike| acc
                    + input_spike.weight
                        * input_spike.kernel_deriv(4.0 - input_spike.time)),
            input_spike_train.potential_deriv(4.0)
        );
        assert_relative_eq!(
            input_spike_train
                .input_spikes
                .iter()
                .fold(0.0, |acc, input_spike| acc
                    + input_spike.weight
                        * input_spike.kernel_deriv(8.0 - input_spike.time)),
            input_spike_train.potential_deriv(8.0)
        );
    }

    #[test]
    fn test_input_spike_train_max_potential() {
        let input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(0, -0.5, -0.25),
            AlphaInputSpike::new(0, 1.0, 0.5),
            AlphaInputSpike::new(0, 3.0, 0.5),
            AlphaInputSpike::new(0, 0.0, 0.75),
            AlphaInputSpike::new(0, 1.5, -0.125),
            AlphaInputSpike::new(0, 2.5, -0.5),
        ]);

        let windows = TimeIntervalUnion::new_from(vec![TimeInterval::Closed {
            start: 0.0,
            end: 10.0,
        }]);
        let pairs = input_spike_train.max_potential_in_windows(&windows, 5);
        assert_relative_eq!(pairs[0].time, 1.5);
        assert_relative_eq!(pairs[0].value, 0.9105875892660236);
        assert_relative_eq!(pairs[1].time, 2.5);
        assert_relative_eq!(pairs[1].value, 0.6467655826353216);
        assert_relative_eq!(pairs[2].time, 1.0);
        assert_relative_eq!(pairs[2].value, 0.5225510026077624);
        assert_relative_eq!(pairs[3].time, 3.896869317893622);
        assert_relative_eq!(pairs[3].value, 0.2952320551017868);
        assert_relative_eq!(pairs[4].time, 3.0);
        assert_relative_eq!(pairs[4].value, 0.07465463828675833);

        let windows = TimeIntervalUnion::new_from(vec![
            TimeInterval::Closed {
                start: 0.0,
                end: 0.5,
            },
            TimeInterval::Closed {
                start: 3.0,
                end: 5.0,
            },
        ]);
        let pairs = input_spike_train.max_potential_in_windows(&windows, 1);
        assert_relative_eq!(pairs[0].time, 0.5);
        assert_relative_eq!(pairs[0].value, 0.3682704765125481);
        // assert_relative_eq!(pairs[1].time, 3.896869317893622);
        // assert_relative_eq!(pairs[1].value, 0.2952320551017868);
        // assert_relative_eq!(pairs[2].time, 5.0);
        // assert_relative_eq!(pairs[2].value, 0.20603746641634715);
    }

    #[test]
    fn test_input_spike_train_min_potential_deriv() {
        let input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(0, -0.5, -0.25),
            AlphaInputSpike::new(0, 1.0, 0.5),
            AlphaInputSpike::new(0, 3.0, 0.5),
            AlphaInputSpike::new(0, 0.0, 0.75),
            AlphaInputSpike::new(0, 1.5, -0.125),
            AlphaInputSpike::new(0, 2.5, -0.5),
        ]);

        let windows = TimeIntervalUnion::new_from(vec![TimeInterval::Closed {
            start: 0.0,
            end: 10.0,
        }]);
        let pairs = input_spike_train.min_potential_deriv_in_windows(&windows, 5);
        assert_relative_eq!(pairs[0].time, 2.500000000001);
        assert_relative_eq!(pairs[0].value, -1.6941273677036164);
        assert_relative_eq!(pairs[1].time, 3.0);
        assert_relative_eq!(pairs[1].value, -0.7099116727436954);
        assert_relative_eq!(pairs[2].time, 2.4255648561820418);
        assert_relative_eq!(pairs[2].value, -0.33587200892375646);
        assert_relative_eq!(pairs[3].time, 2.5);
        assert_relative_eq!(pairs[3].value, -0.3349864534768354);
        assert_relative_eq!(pairs[4].time, 0.0);
        assert_relative_eq!(pairs[4].value, -0.20609015883751605);

        let windows = TimeIntervalUnion::new_from(vec![
            TimeInterval::Closed {
                start: 0.0,
                end: 0.5,
            },
            TimeInterval::Closed {
                start: 3.0,
                end: 5.0,
            },
        ]);
        let pairs = input_spike_train.min_potential_deriv_in_windows(&windows, 1);
        assert_relative_eq!(pairs[0].time, 3.0);
        assert_relative_eq!(pairs[0].value, -0.7099116727436958);
        // assert_relative_eq!(pairs[1].time, 0.0);
        // assert_relative_eq!(pairs[1].value, -0.20609015883751605);
        // assert_relative_eq!(pairs[2].time, 4.896869317893621);
        // assert_relative_eq!(pairs[2].value, -0.10860980344674183);
    }

    #[test]
    fn test_input_spike_train_next_potential_threshold_crossing() {
        // let input_spike_train = AlphaInputSpikeTrain::new(vec![
        //     AlphaInputSpike::new(0, -0.5, -0.25),
        //     AlphaInputSpike::new(0, 1.0, 0.5),
        //     AlphaInputSpike::new(0, 3.0, 0.5),
        //     AlphaInputSpike::new(0, 0.0, 0.75),
        //     AlphaInputSpike::new(0, 1.5, -0.125),
        //     AlphaInputSpike::new(0, 2.5, -0.5),
        // ]);

        let input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(0, -0.5, -0.25),
            AlphaInputSpike::new(0, 1.0, 0.5),
            AlphaInputSpike::new(0, 3.0, 0.5),
            AlphaInputSpike::new(0, 0.0, 0.75),
            AlphaInputSpike::new(0, 1.5, -0.125),
            AlphaInputSpike::new(0, 2.5, -0.5),
        ]);

        assert_relative_eq!(
            input_spike_train
                .next_potential_threshold_crossing(0.0, 0.5)
                .unwrap(),
            0.8357256680627244
        );

        assert_relative_eq!(
            input_spike_train
                .next_potential_threshold_crossing(0.0, 0.75)
                .unwrap(),
            1.2019214521888513
        );

        assert_eq!(
            input_spike_train.next_potential_threshold_crossing(0.0, 1.0),
            None
        );
    }

    #[test]
    fn test_neuron_potential() {
        // let neuron = AlphaNeuron::new_from(0);
        todo!();
    }

    #[test]
    fn test_neuron_potential_deriv() {
        // let neuron = AlphaNeuron::new_from(0);
        todo!();
    }

    //     // assert_eq!(
    //     //     neuron.saddle_potential_in_windows(&input_spike_train, (8.0, 1.0)),
    //     // );
    //     // assert_eq!(
    //     //     neuron.max_potential_in_window(&input_spike_train, (3.0, 2.0)),
    //     //     None
    //     // );

    //     // Without any input spike
    //     let neuron = AlphaNeuron::new_empty(0);
    //     let spike_train = LoopSpikeTrain::new_empty(10.0);
    //     let input_spike_train = InLoopSpikeTrain::new_from(&neuron.inputs, &spike_train);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (0.0, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 0.0);
    //     assert_relative_eq!(zmax, 0.0);

    //     // With a single input spike
    //     let inputs = vec![Input {
    //         source_id: 0,
    //         weight: 1.0,
    //         delay: 1.0,
    //     }];
    //     let neuron = AlphaNeuron::new_with_inputs(42, inputs);
    //     let spikes = vec![Spike {
    //         source_id: 0,
    //         time: 0.0,
    //     }];
    //     let spike_train = LoopSpikeTrain::build(spikes, 10.0).unwrap();
    //     let input_spike_train = InLoopSpikeTrain::new_from(&neuron.inputs, &spike_train);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (0.0, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 2.0);
    //     assert_relative_eq!(zmax, 1.0);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (2.5, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 2.5);
    //     assert_relative_eq!(zmax, 0.9097959895689501);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (70.0, 80.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 72.0);
    //     assert_relative_eq!(zmax, 1.0);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (1.25, 2.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 2.0);
    //     assert_relative_eq!(zmax, 1.0);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (2.0, 3.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 2.0);
    //     assert_relative_eq!(zmax, 1.0);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (1.0, 3.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 2.0);
    //     assert_relative_eq!(zmax, 1.0);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (0.0, 1.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 0.0);
    //     assert_relative_eq!(zmax, 0.003019163651122607);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (8.0, 11.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 8.0);
    //     assert_relative_eq!(zmax, 0.01735126523666451);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (8.0, 12.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 12.0);
    //     assert_relative_eq!(zmax, 1.0);

    //     // With multiple input spikes
    //     let inputs = vec![
    //         Input {
    //             source_id: 0,
    //             weight: 1.0,
    //             delay: 1.0,
    //         },
    //         Input {
    //             source_id: 1,
    //             weight: 1.0,
    //             delay: 2.5,
    //         },
    //         Input {
    //             source_id: 2,
    //             weight: 1.0,
    //             delay: 4.0,
    //         },
    //         Input {
    //             source_id: 3,
    //             weight: -1.0,
    //             delay: 3.5,
    //         },
    //     ];
    //     let neuron = AlphaNeuron::new_with_inputs(42, inputs);
    //     let spikes = vec![
    //         Spike {
    //             source_id: 0,
    //             time: 0.0,
    //         },
    //         Spike {
    //             source_id: 1,
    //             time: 0.0,
    //         },
    //         Spike {
    //             source_id: 2,
    //             time: 0.0,
    //         },
    //         Spike {
    //             source_id: 3,
    //             time: 0.0,
    //         },
    //     ];
    //     let spike_train = LoopSpikeTrain::build(spikes, 10.0).unwrap();
    //     let input_spike_train = InLoopSpikeTrain::new_from(&neuron.inputs, &spike_train);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (0.0, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 3.225873747625197);
    //     assert_relative_eq!(zmax, 1.6089873115477644);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (2.0, 4.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 3.225873747625197);
    //     assert_relative_eq!(zmax, 1.6089873115477644);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (3.0, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 3.225873747625197);
    //     assert_relative_eq!(zmax, 1.6089873115477644);

    //     let (tmax, zmax) = neuron
    //         .max_potential_in_window(&input_spike_train, (5.0, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmax, 5.0);
    //     assert_relative_eq!(zmax, 0.8471776842735802);
    // }

    // #[test]
    // fn test_neuron_min_potential_deriv_in_window() {
    //     // Empty window
    //     let neuron = AlphaNeuron::new_empty(0);
    //     let spike_train = LoopSpikeTrain::new_empty(10.0);
    //     let input_spike_train = InLoopSpikeTrain::new_from(&neuron.inputs, &spike_train);
    //     assert_eq!(
    //         neuron.min_potential_deriv_in_window(&input_spike_train, (8.0, 1.0)),
    //         None
    //     );
    //     assert_eq!(
    //         neuron.min_potential_deriv_in_window(&input_spike_train, (3.0, 2.0)),
    //         None
    //     );

    //     // Without any input spike
    //     let neuron = AlphaNeuron::new_empty(0);
    //     let spike_train = LoopSpikeTrain::new_empty(10.0);
    //     let input_spike_train = InLoopSpikeTrain::new_from(&neuron.inputs, &spike_train);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (0.0, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 0.0);
    //     assert_relative_eq!(zmin, 0.0);

    //     // With a single input spike
    //     let inputs = vec![Input {
    //         source_id: 0,
    //         weight: 1.0,
    //         delay: 1.0,
    //     }];
    //     let neuron = AlphaNeuron::new_with_inputs(42, inputs);
    //     let spikes = vec![Spike {
    //         source_id: 0,
    //         time: 0.0,
    //     }];
    //     let spike_train = LoopSpikeTrain::build(spikes, 10.0).unwrap();
    //     let input_spike_train = InLoopSpikeTrain::new_from(&neuron.inputs, &spike_train);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (0.0, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 3.0);
    //     assert_relative_eq!(zmin, -0.36787944117144233);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (2.5, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 3.0);
    //     assert_relative_eq!(zmin, -0.36787944117144233);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (70.0, 80.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 73.0);
    //     assert_relative_eq!(zmin, -0.36787944117144233);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (1.25, 2.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 2.0);
    //     assert_relative_eq!(zmin, 0.0);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (2.0, 3.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 3.0);
    //     assert_relative_eq!(zmin, -0.36787944117144233);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (3.0, 4.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 3.0);
    //     assert_relative_eq!(zmin, -0.36787944117144233);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (0.0, 1.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 0.0);
    //     assert_relative_eq!(zmin, -0.002683701023220095);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (8.0, 9.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 8.0);
    //     assert_relative_eq!(zmin, -0.014872513059998151);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (8.0, 14.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 13.0);
    //     assert_relative_eq!(zmin, -0.36787944117144233);

    //     // With multiple input spikes
    //     let inputs = vec![
    //         Input {
    //             source_id: 0,
    //             weight: 1.0,
    //             delay: 1.0,
    //         },
    //         Input {
    //             source_id: 1,
    //             weight: 1.0,
    //             delay: 2.5,
    //         },
    //         Input {
    //             source_id: 2,
    //             weight: 1.0,
    //             delay: 4.0,
    //         },
    //         Input {
    //             source_id: 3,
    //             weight: -1.0,
    //             delay: 3.5,
    //         },
    //     ];
    //     let neuron = AlphaNeuron::new_with_inputs(42, inputs);
    //     let spikes = vec![
    //         Spike {
    //             source_id: 0,
    //             time: 0.0,
    //         },
    //         Spike {
    //             source_id: 1,
    //             time: 0.0,
    //         },
    //         Spike {
    //             source_id: 2,
    //             time: 0.0,
    //         },
    //         Spike {
    //             source_id: 3,
    //             time: 0.0,
    //         },
    //     ];
    //     let spike_train = LoopSpikeTrain::build(spikes, 10.0).unwrap();
    //     let input_spike_train = InLoopSpikeTrain::new_from(&neuron.inputs, &spike_train);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (0.0, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 3.5);
    //     assert_relative_eq!(zmin, -3.0547065498182806);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (2.0, 4.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 3.5);
    //     assert_relative_eq!(zmin, -3.0547065498182806);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (3.0, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 3.5);
    //     assert_relative_eq!(zmin, -3.0547065498182806);

    //     let (tmin, zmin) = neuron
    //         .min_potential_deriv_in_window(&input_spike_train, (5.0, 10.0))
    //         .unwrap();
    //     assert_relative_eq!(tmin, 5.7286993406927635);
    //     assert_relative_eq!(zmin, -0.3215556408466008);
    // }

    // #[cfg(test)]
    // mod tests {
    //     use approx::assert_relative_eq;
    //     use core::panic;

    //     use super::*;

    //     #[test]
    //     fn test_memorize_empty_periodic_spike_train() {
    //         let period = 100.0;
    //         let lim_weights = (-1.0, 1.0);
    //         let max_level = 0.0;
    //         let min_slope = 0.0;
    //         let half_width = 0.5;
    //         let objective = Objective::L2Norm;

    //         let spike_train: Vec<Spike> = vec![
    //             Spike::new(1, 1.0),
    //             Spike::new(1, 3.0),
    //             Spike::new(1, 25.0),
    //             Spike::new(2, 5.0),
    //             Spike::new(2, 77.0),
    //             Spike::new(2, 89.0),
    //         ];

    //         let neuron = Neuron::new(0);

    //         let spike_train: Vec<f64> = vec![];
    //         let mut input_spike_train: Vec<InputSpike> = vec![
    //             InputSpike::new(0, 1.0, 3.0),
    //             InputSpike::new(0, 1.0, 5.0),
    //             InputSpike::new(0, 1.0, 6.0),
    //             InputSpike::new(1, 1.0, 8.0),
    //             InputSpike::new(1, 1.0, 27.0),
    //             InputSpike::new(1, 1.0, 30.0),
    //             InputSpike::new(2, 1.0, 5.5),
    //             InputSpike::new(2, 1.0, 77.5),
    //             InputSpike::new(2, 1.0, 89.5),
    //         ];

    //         assert_eq!(
    //             neuron
    //                 .memorize_periodic_spike_train(
    //                     &spike_train,
    //                     &mut input_spike_train,
    //                     period,
    //                     lim_weights,
    //                     max_level,
    //                     min_slope,
    //                     half_width,
    //                     objective,
    //                 )
    //                 .unwrap(),
    //             vec![0.0, 0.0, 0.0]
    //         );

    //         // let expected_connections = vec![
    //         //     vec![Connection::build(0, 0, 0, 0.0, 1.0).unwrap()],
    //         //     vec![
    //         //         Connection::build(1, 0, 1, 0.0, 2.0).unwrap(),
    //         //         Connection::build(2, 0, 1, 0.0, 5.0).unwrap(),
    //         //     ],
    //         //     vec![Connection::build(3, 0, 2, 0.0, 0.5).unwrap()],
    //         // ];

    //         // assert_eq!(connections, expected_connections);
    //     }

    #[test]
    fn test_memorize_single_spike_periodic_spike_train() {
        let spike_train = MultiChannelSpikeTrain::new_from(vec![
            Spike::new(0, 1.55),
            Spike::new(1, 1.0),
            Spike::new(2, 1.5),
            Spike::new(3, 2.0),
            Spike::new(4, 3.5),
        ]);

        let mut neuron = AlphaNeuron::new_empty(0, 0);
        neuron.add_input(0, f64::NAN, 0.0);
        neuron.add_input(1, f64::NAN, 0.0);
        neuron.add_input(2, f64::NAN, 0.0);
        neuron.add_input(3, f64::NAN, 0.0);
        neuron.add_input(4, f64::NAN, 0.0);

        let time_template = TimeTemplate::new_cyclic_from(&vec![1.55], 0.25, 100.0);

        let input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(1, -99.0, f64::NAN),
            AlphaInputSpike::new(2, -98.5, f64::NAN),
            AlphaInputSpike::new(0, -98.45, f64::NAN),
            AlphaInputSpike::new(3, -98.0, f64::NAN),
            AlphaInputSpike::new(4, -96.5, f64::NAN),
            AlphaInputSpike::new(1, 1.0, f64::NAN),
            AlphaInputSpike::new(2, 1.5, f64::NAN),
            AlphaInputSpike::new(0, 1.55, f64::NAN),
            AlphaInputSpike::new(3, 2.0, f64::NAN),
            AlphaInputSpike::new(4, 3.5, f64::NAN),
            AlphaInputSpike::new(1, 101.0, f64::NAN),
            AlphaInputSpike::new(2, 101.5, f64::NAN),
            AlphaInputSpike::new(0, 101.55, f64::NAN),
            AlphaInputSpike::new(3, 102.0, f64::NAN),
            AlphaInputSpike::new(4, 103.5, f64::NAN),
        ]);
        neuron
            .memorize(
                vec![time_template],
                vec![input_spike_train],
                (-5.0, 5.0),
                0.5,
                0.0,
                Objective::L2,
            )
            .expect("Memorization failed");

        // let input_spike_train = AlphaInputSpikeTrain::new_from(&neuron.inputs, &spike_train);
        neuron.init_input_spike_train(&spike_train);
        assert_relative_eq!(neuron.next_spike(0.0).unwrap().time, 1.55);
        // panic!("{} fires at {}", neuron.id, neuron.next_spike(0.0).unwrap().time);

        // let mut connections = vec![
        //     vec![Connection::build(0, 0, 0, 1.0, 0.0).unwrap()],
        //     vec![Connection::build(1, 1, 0, 1.0, 0.0).unwrap()],
        //     vec![Connection::build(2, 2, 0, 1.0, 0.0).unwrap()],
        //     vec![Connection::build(3, 3, 0, 1.0, 0.0).unwrap()],
        //     vec![Connection::build(4, 4, 0, 1.0, 0.0).unwrap()],
        // ];

        // let _weights = neuron(
        //         &spike_train,
        //         &mut input_spike_train,
        //         period,
        //         lim_weights,
        //         max_level,
        //         min_slope,
        //         half_width,
        //         Objective::None,
        //     )
        //     .expect("Memorization failed");

        // let weights = neuron
        //     .memorize_periodic_spike_train(
        //         &spike_train,
        //         &mut input_spike_train,
        //         period,
        //         lim_weights,
        //         max_level,
        //         min_slope,
        //         half_width,
        //         Objective::L2Norm,
        //     )
        //     .expect("L2-memorization failed");

        // assert_eq!(
        //     weights,
        //     vec![
        //         -1.40501505062582,
        //         0.8276421729855583,
        //         2.2129265840338137,
        //         -1.2119262246876459,
        //         -0.0
        //     ]
        // );
        // assert_relative_eq!(weights[1], 1.157671, epsilon = 1e-6);
        // assert_relative_eq!(weights[2], 0.011027, epsilon = 1e-6);
        // assert_relative_eq!(weights[3], -0.415485, epsilon = 1e-6);
        // assert_relative_eq!(weights[4], 0.0);

        // let weights = neuron
        //     .memorize_periodic_spike_train(
        //         &spike_train,
        //         &mut input_spike_train,
        //         period,
        //         lim_weights,
        //         max_level,
        //         min_slope,
        //         half_width,
        //         Objective::L1Norm,
        //     )
        //     .expect("L1-memorization failed");
        // assert_eq!(
        //     weights,
        //     vec![
        //         -2.092002954197055,
        //         0.8276421729856879,
        //         2.2129265840329473,
        //         -0.41548472074177123,
        //         0.0
        //     ]
        // );
    }
}
