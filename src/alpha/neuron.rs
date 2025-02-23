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
// use crate::core::spikes::MultiChannelCyclicSpikeTrain;
use crate::core::utils::{TimeInterval, TimeIntervalUnion, TimeValuePair};
use crate::core::FIRING_THRESHOLD;
use crate::error::SNNError;

const RIGHT_LIMIT: f64 = 1e-12;

/// A spiking neuron model that uses alpha-shaped synaptic kernels to process incoming spikes.
///
/// The alpha kernel h(t) models the post-synaptic potential in response to input spikes and has the form:
/// ```text
/// h(t) = (t/τ)exp(1 - t/τ) for t ≥ 0
///      = 0              for t < 0
/// ```
/// where τ is the time constant that determines the kernel's temporal scale.
/// The current implementation only supports τ = 1.
///
/// # Properties
/// - Continuous and differentiable for all t > 0
/// - Reaches maximum value of 1 at t = τ
/// - Natural model for rise and decay of post-synaptic potentials
/// - Zero response to negative time differences (causality)
///
/// # Implementation Notes
/// - Uses Lambert W function for efficient threshold crossing detection
/// - Maintains numerical stability through careful handling of exponentials
/// - Supports noisy thresholds through optional Gaussian sampling
///
/// # Fields
/// - `id`: Unique identifier for the neuron
/// - `inputs`: Vector of synaptic inputs with weights and delays
/// - `threshold`: Firing threshold for action potential generation
/// - `ftimes`: Vector of past firing times
/// - `input_spike_train`: Collection of incoming spikes with alpha-kernel processing
/// - `threshold_sampler`: Optional noise generator for the threshold
/// - `rng`: Random number generator for stochastic components
#[derive(Debug, Clone, Serialize)]
pub struct AlphaNeuron {
    // The neuron ID.
    id: usize,
    // The neuron inputs.
    inputs: Vec<Input>,
    // The neuron firing threshold.
    threshold: f64,
    /// The neuron firing times.
    ftimes: Vec<f64>,
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
            ftimes: Vec<f64>,
            input_spike_train: AlphaInputSpikeTrain,
        }

        let data = AlphaNeuronData::deserialize(deserializer)?;
        Ok(AlphaNeuron {
            id: data.id,
            inputs: data.inputs,
            threshold: data.threshold,
            ftimes: data.ftimes,
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
            ftimes: vec![],
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
            ftimes: vec![],
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

    fn init_threshold_sampler(&mut self, sigma: f64, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
        self.threshold_sampler = Normal::new(FIRING_THRESHOLD, sigma).unwrap();
    }

    fn sample_threshold(&mut self) {
        self.threshold = self.threshold_sampler.sample(&mut self.rng);
    }

    fn ftimes_ref(&self) -> &Vec<f64> {
        self.ftimes.as_ref()
    }

    fn ftimes_mut(&mut self) -> &mut Vec<f64> {
        self.ftimes.as_mut()
    }

    /// A reference to the vector of inputs of the neuron
    fn inputs(&self) -> &[Input] {
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

    fn clear_inputs(&mut self) {
        self.inputs.clear();
    }

    /// Initialize the input spikes of the neuron from the provided spike train.
    fn init_input_spike_train(&mut self, times: &Vec<Vec<f64>>) {
        self.input_spike_train = Self::InputSpikeTrain::new_from(self.inputs(), times);
    }

    /// Drain the input spikes which are irrelevant after the provided time.
    fn drain_input_spike_train(&mut self, time: f64) {
        let pos = self.input_spike_train.find_before(time).unwrap_or(0);
        self.input_spike_train.input_spikes.drain(..pos);
    }

    /// Receive and process the spikes from the input channels.
    /// The spikes are provided as a vector of optional firing times.
    /// There is at most one spike per channel.
    fn receive_spikes(&mut self, ftimes: &Vec<Option<f64>>) {
        let mut new_input_spikes: Vec<AlphaInputSpike> = self
            .inputs_iter()
            .filter_map(|input| {
                ftimes[input.source_id].map(|ft| {
                    Self::InputSpike::new(input.source_id, ft + input.delay, input.weight)
                })
            })
            .collect();
        new_input_spikes.sort_by(|s1, s2| s1.partial_cmp(s2).unwrap());
        self.input_spike_train.insert_sorted(new_input_spikes);
    }
}

/// An input spike through an alpha-shaped synapse.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlphaInputSpike {
    /// The ID of the input along which the spike is received.
    pub input_id: usize,
    /// The time at which the spike is received.
    pub time: f64,
    /// The weight of the connection along which is the spike is received.
    pub weight: f64,
    /// The coefficient sum_{j <= i} w_j * exp(1-(s_i - s_j))
    pub a: f64,
    /// The coefficient sum_{j <= i} w_j * s_j * exp(1-(s_i - s_j))
    pub b: f64,
}

/// Implement the `PartialEq` trait for `AlphaInputSpike` based on the time and input_id.
impl PartialEq for AlphaInputSpike {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.input_id == other.input_id
    }
}

/// Implement the `PartialOrd` trait for `AlphaInputSpike` to allow sorting by time.
impl PartialOrd for AlphaInputSpike {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.time.partial_cmp(&other.time)
    }
}

impl InputSpike for AlphaInputSpike {
    /// A new input spike through an alpha-shaped synapse.
    fn new(input_id: usize, time: f64, weight: f64) -> Self {
        AlphaInputSpike {
            input_id,
            time,
            weight,
            a: f64::NAN,
            b: f64::NAN,
        }
    }

    /// The time at which the input spike is received.
    fn time(&self) -> f64 {
        self.time
    }

    /// The weight of the connection along which the spike is received.
    fn weight(&self) -> f64 {
        self.weight
    }

    /// The ID of the input along which the spike is received.
    fn input_id(&self) -> usize {
        self.input_id
    }

    /// The alpha-shaped kernel.
    fn kernel(&self, dt: f64) -> f64 {
        if dt > 0.0 {
            dt * (1.0 - dt).exp()
        } else {
            0.0
        }
    }

    /// The derivative of the alpha-shaped kernel.
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
    /// A new input spike train.
    fn new(mut input_spikes: Vec<AlphaInputSpike>) -> Self {
        input_spikes.sort_by(|input_spike_1, input_spike_2| {
            input_spike_1.partial_cmp(&input_spike_2).unwrap()
        });
        let mut input_spike_train = AlphaInputSpikeTrain { input_spikes };
        input_spike_train.compute_ab(0);
        input_spike_train
    }

    /// Find the position of the last input spike (strictly) before time, if any.
    fn find_before(&self, time: f64) -> Option<usize> {
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

    /// Compute the coefficients a and b of the input spikes.
    fn compute_ab(&mut self, start: usize) {
        if self.input_spikes.is_empty() {
            return;
        }

        if start == 0 {
            self.input_spikes[0].a = self.input_spikes[0].weight * E;
            self.input_spikes[0].b = self.input_spikes[0].weight * self.input_spikes[0].time * E;

            (start + 1..self.input_spikes.len()).for_each(|i| {
                self.input_spikes[i].a = self.input_spikes[i - 1].a
                    * (self.input_spikes[i - 1].time - self.input_spikes[i].time).exp()
                    + self.input_spikes[i].weight * E;
                self.input_spikes[i].b = self.input_spikes[i - 1].b
                    * (self.input_spikes[i - 1].time - self.input_spikes[i].time).exp()
                    + self.input_spikes[i].weight * self.input_spikes[i].time * E;
            });
        } else {
            (start..self.input_spikes.len()).for_each(|i| {
                self.input_spikes[i].a = self.input_spikes[i - 1].a
                    * (self.input_spikes[i - 1].time - self.input_spikes[i].time).exp()
                    + self.input_spikes[i].weight * E;
                self.input_spikes[i].b = self.input_spikes[i - 1].b
                    * (self.input_spikes[i - 1].time - self.input_spikes[i].time).exp()
                    + self.input_spikes[i].weight * self.input_spikes[i].time * E;
            });
        }
    }
}

impl InputSpikeTrain for AlphaInputSpikeTrain {
    /// An input spike through an alpha-shaped synapse.
    type InputSpike = AlphaInputSpike;

    /// A new empty input spike train.
    fn new_empty() -> Self {
        AlphaInputSpikeTrain {
            input_spikes: vec![],
        }
    }

    /// A new input spike train from a collection of inputs and firing times.
    fn new_from(inputs: &[Input], ftimes: &Vec<Vec<f64>>) -> Self {
        let input_spikes: Vec<Self::InputSpike> = inputs
            .iter()
            .enumerate()
            .flat_map(|(i, input)| {
                ftimes[input.source_id]
                    .iter()
                    .map(move |ft| AlphaInputSpike::new(i, ft + input.delay, input.weight))
            })
            .collect();

        Self::new(input_spikes)
    }

    /// A new input spike train from a collection of inputs and (cyclic) firing times.
    /// The firing times are periodically repeated until being negligeable on the provided interval.
    fn new_cyclic_from(
        inputs: &[Input],
        ftimes: &Vec<Vec<f64>>,
        period: f64,
        interval: &TimeInterval,
    ) -> Self {
        match interval {
            TimeInterval::Empty => Self::new_empty(),
            TimeInterval::Closed { start, end } => {
                let min_time = *start + lambert_wm1(-MIN_INPUT_VALUE / E);
                let input_spikes: Vec<Self::InputSpike> = inputs
                    .iter()
                    .enumerate()
                    .flat_map(|(input_id, input)| {
                        ftimes[input.source_id]
                            .iter()
                            .map(|time| {
                                let time = time + input.delay;
                                time - ((time - end) / period).ceil() * period
                            })
                            .flat_map(move |time| {
                                (0..).map_while(move |i| {
                                    let input_time = time - i as f64 * period;
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

                Self::new(input_spikes)
            }
        }
    }

    /// Update the input spikes after a change in the input weights.
    fn apply_weight_change(&mut self, inputs: &[Input]) -> Result<(), SNNError> {
        self.input_spikes.iter_mut().try_for_each(|input_spike| {
            let input = inputs
                .get(input_spike.input_id)
                .ok_or(SNNError::OutOfBounds(
                    "Input spike train and inputs are not aligned".to_string(),
                ))?;
            input_spike.weight = input.weight;
            Ok(())
        })?;

        self.compute_ab(0);

        Ok(())
    }

    /// Insert a collection of (sorted) input spikes into the neuron's input spike train.
    /// Spike insertions maintain sorted order with O(n) worst-case complexity.
    fn insert_sorted(&mut self, new_input_spikes: Vec<Self::InputSpike>) {
        if new_input_spikes.is_empty() {
            return;
        } else {
            // Collect the insertion indices of all new input spikes and keep track of the first insertion index
            let indices: Vec<usize> = new_input_spikes
                .iter()
                .enumerate()
                .map(|(offset, new_input_spike)| {
                    match self.input_spikes.binary_search_by(|input_spike| {
                        input_spike.time.partial_cmp(&new_input_spike.time).unwrap()
                    }) {
                        Ok(pos) => {
                            self.input_spikes[pos..]
                                .iter()
                                .enumerate()
                                .take_while(|(_, input_spike)| {
                                    input_spike.time == new_input_spike.time
                                })
                                .map(|(i, _)| pos + i + 1)
                                .last()
                                .unwrap()
                                + offset
                        }
                        Err(pos) => pos + offset,
                    }
                })
                .collect();
            let start = *indices.first().unwrap();

            // Create space at the end of the input spikes vector
            self.input_spikes.resize_with(
                self.input_spikes.len() + new_input_spikes.len(),
                Default::default,
            );

            // Insert the new input spikes at the pre-computed indices
            std::iter::once(self.input_spikes.len())
                .chain(indices.into_iter().rev())
                .tuple_windows()
                .zip(new_input_spikes.into_iter().enumerate().rev())
                .for_each(|((next_pos, pos), (offset, new_input_spike))| {
                    (pos + 1..next_pos).rev().for_each(|i| {
                        self.input_spikes[i].input_id = self.input_spikes[i - offset - 1].input_id;
                        self.input_spikes[i].time = self.input_spikes[i - offset - 1].time;
                        self.input_spikes[i].weight = self.input_spikes[i - offset - 1].weight;
                    });

                    self.input_spikes[pos].input_id = new_input_spike.input_id;
                    self.input_spikes[pos].time = new_input_spike.time;
                    self.input_spikes[pos].weight = new_input_spike.weight;
                });

            self.compute_ab(start);
        }
    }

    /// An iterator over the input spikes.
    fn iter(&self) -> impl Iterator<Item = &Self::InputSpike> + '_ {
        self.input_spikes.iter()
    }

    /// A mutable iterator over the input spikes.
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::InputSpike> + '_ {
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

    /// Evaluate the potential at a given time.
    fn potential(&self, time: f64) -> f64 {
        match self.find_before(time) {
            Some(pos) => {
                (time * self.input_spikes[pos].a - self.input_spikes[pos].b)
                    * (self.input_spikes[pos].time - time).exp()
            }
            None => 0.0,
        }
    }

    /// Evaluate the derivative of the potential at a given time.
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
    use crate::core::optim::{Objective, TimeTemplate};

    #[test]
    fn test_input_spike_train_new_cyclic_from() {
        let inputs = vec![
            Input::new(0, 0.5, 1.0),
            Input::new(1, 1.0, 3.0),
            Input::new(2, 1.0, 0.5),
            Input::new(1, -0.5, 2.0),
        ];
        let ftimes = vec![vec![0.0, 22.0], vec![10.0, 47.0], vec![34.5, 98.5]];
        let period = 100.0;
        let interval = TimeInterval::new(3.0, 103.0);

        let input_spike_train =
            AlphaInputSpikeTrain::new_cyclic_from(&inputs, &ftimes, period, &interval);

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
    fn test_input_spike_train_insert_sorted() {
        let mut input_spike_train = AlphaInputSpikeTrain::new(vec![]);
        input_spike_train.insert_sorted(vec![
            AlphaInputSpike::new(0, 3.0, 1.0),
            AlphaInputSpike::new(0, 5.0, 1.0),
            AlphaInputSpike::new(0, 7.0, 1.0),
        ]);
        assert_relative_eq!(input_spike_train.input_spikes[0].time, 3.0);
        assert_relative_eq!(input_spike_train.input_spikes[1].time, 5.0);
        assert_relative_eq!(input_spike_train.input_spikes[2].time, 7.0);
        assert_relative_eq!(
            input_spike_train.input_spikes[1].a,
            input_spike_train.input_spikes[..=1]
                .iter()
                .map(|input_spike| input_spike.weight * (1.0 - (5.0 - input_spike.time)).exp())
                .sum::<f64>()
        );
        assert_relative_eq!(
            input_spike_train.input_spikes[1].b,
            input_spike_train.input_spikes[..=1]
                .iter()
                .map(|input_spike| input_spike.weight
                    * input_spike.time
                    * (1.0 - (5.0 - input_spike.time)).exp())
                .sum::<f64>()
        );

        let mut input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(0, 3.0, 1.0),
            AlphaInputSpike::new(0, 5.0, 1.0),
            AlphaInputSpike::new(0, 7.0, 1.0),
        ]);
        input_spike_train.insert_sorted(vec![]);
        assert_relative_eq!(input_spike_train.input_spikes[0].time, 3.0);
        assert_relative_eq!(input_spike_train.input_spikes[1].time, 5.0);
        assert_relative_eq!(input_spike_train.input_spikes[2].time, 7.0);
        assert_relative_eq!(
            input_spike_train.input_spikes[1].a,
            input_spike_train.input_spikes[..=1]
                .iter()
                .map(|input_spike| input_spike.weight * (1.0 - (5.0 - input_spike.time)).exp())
                .sum::<f64>()
        );
        assert_relative_eq!(
            input_spike_train.input_spikes[1].b,
            input_spike_train.input_spikes[..=1]
                .iter()
                .map(|input_spike| input_spike.weight
                    * input_spike.time
                    * (1.0 - (5.0 - input_spike.time)).exp())
                .sum::<f64>()
        );

        let mut input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(0, 1.0, 1.0),
            AlphaInputSpike::new(0, 5.0, 1.0),
            AlphaInputSpike::new(0, 9.0, 1.0),
        ]);
        input_spike_train.insert_sorted(vec![
            AlphaInputSpike::new(0, 3.0, 1.0),
            AlphaInputSpike::new(0, 7.0, 1.0),
        ]);
        assert_relative_eq!(input_spike_train.input_spikes[0].time, 1.0);
        assert_relative_eq!(input_spike_train.input_spikes[1].time, 3.0);
        assert_relative_eq!(input_spike_train.input_spikes[2].time, 5.0);
        assert_relative_eq!(input_spike_train.input_spikes[3].time, 7.0);
        assert_relative_eq!(input_spike_train.input_spikes[4].time, 9.0);
        assert_relative_eq!(
            input_spike_train.input_spikes[2].a,
            input_spike_train.input_spikes[..=2]
                .iter()
                .map(|input_spike| input_spike.weight * (1.0 - (5.0 - input_spike.time)).exp())
                .sum::<f64>()
        );
        assert_relative_eq!(
            input_spike_train.input_spikes[2].b,
            input_spike_train.input_spikes[..=2]
                .iter()
                .map(|input_spike| input_spike.weight
                    * input_spike.time
                    * (1.0 - (5.0 - input_spike.time)).exp())
                .sum::<f64>()
        );

        let mut input_spike_train = AlphaInputSpikeTrain::new(vec![
            AlphaInputSpike::new(0, 1.0, 1.0),
            AlphaInputSpike::new(0, 5.0, 1.0),
            AlphaInputSpike::new(0, 9.0, 1.0),
        ]);
        input_spike_train.insert_sorted(vec![
            AlphaInputSpike::new(0, 2.0, 1.0),
            AlphaInputSpike::new(0, 3.0, 1.0),
            AlphaInputSpike::new(0, 11.0, 1.0),
        ]);
        assert_relative_eq!(input_spike_train.input_spikes[0].time, 1.0);
        assert_relative_eq!(input_spike_train.input_spikes[1].time, 2.0);
        assert_relative_eq!(input_spike_train.input_spikes[2].time, 3.0);
        assert_relative_eq!(input_spike_train.input_spikes[3].time, 5.0);
        assert_relative_eq!(input_spike_train.input_spikes[4].time, 9.0);
        assert_relative_eq!(input_spike_train.input_spikes[5].time, 11.0);
        assert_relative_eq!(
            input_spike_train.input_spikes[3].a,
            input_spike_train.input_spikes[..=3]
                .iter()
                .map(|input_spike| input_spike.weight * (1.0 - (5.0 - input_spike.time)).exp())
                .sum::<f64>()
        );
        assert_relative_eq!(
            input_spike_train.input_spikes[3].b,
            input_spike_train.input_spikes[..=3]
                .iter()
                .map(|input_spike| input_spike.weight
                    * input_spike.time
                    * (1.0 - (5.0 - input_spike.time)).exp())
                .sum::<f64>()
        );
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
    fn test_memorize_single_spike_periodic_spike_train_l1() {
        let mut neuron = AlphaNeuron::new_empty(0, 0);
        neuron.push_input(0, f64::NAN, 0.0);
        neuron.push_input(1, f64::NAN, 0.0);
        neuron.push_input(2, f64::NAN, 0.0);
        neuron.push_input(3, f64::NAN, 0.0);
        neuron.push_input(4, f64::NAN, 0.0);

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
                Objective::L1,
            )
            .expect("Memorization failed");

        neuron.init_input_spike_train(&vec![
            vec![1.55],
            vec![1.0],
            vec![1.5],
            vec![2.0],
            vec![3.5],
        ]);
        assert_relative_eq!(neuron.next_spike(0.0).unwrap(), 1.55, epsilon = 1e-9);
    }

    #[test]
    fn test_memorize_single_spike_periodic_spike_train_l2() {
        let mut neuron = AlphaNeuron::new_empty(0, 0);
        neuron.push_input(0, f64::NAN, 0.0);
        neuron.push_input(1, f64::NAN, 0.0);
        neuron.push_input(2, f64::NAN, 0.0);
        neuron.push_input(3, f64::NAN, 0.0);
        neuron.push_input(4, f64::NAN, 0.0);

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

        neuron.init_input_spike_train(&vec![
            vec![1.55],
            vec![1.0],
            vec![1.5],
            vec![2.0],
            vec![3.5],
        ]);
        assert_relative_eq!(neuron.next_spike(0.0).unwrap(), 1.55, epsilon = 1e-9);
    }

    #[test]
    fn test_memorize_single_spike_periodic_spike_train_linfinity() {
        let mut neuron = AlphaNeuron::new_empty(0, 0);
        neuron.push_input(0, f64::NAN, 0.0);
        neuron.push_input(1, f64::NAN, 0.0);
        neuron.push_input(2, f64::NAN, 0.0);
        neuron.push_input(3, f64::NAN, 0.0);
        neuron.push_input(4, f64::NAN, 0.0);

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
                Objective::LInfinity,
            )
            .expect("Memorization failed");

        neuron.init_input_spike_train(&vec![
            vec![1.55],
            vec![1.0],
            vec![1.5],
            vec![2.0],
            vec![3.5],
        ]);
        assert_relative_eq!(neuron.next_spike(0.0).unwrap(), 1.55, epsilon = 1e-9);
    }
}
