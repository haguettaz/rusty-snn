//! Neuron-related traits and structures.
use embed_doc_image::embed_doc_image;
use grb::prelude::*;
use log;
use serde::{Deserialize, Serialize};

use crate::core::spikes::{MultiChannelCyclicSpikeTrain, MultiChannelSpikeTrain};
use crate::core::utils::{TimeInterval, TimeIntervalUnion, TimeValuePair};

pub const MIN_INPUT_VALUE: f64 = 1e-12;

use super::optim;
use super::optim::{Objective, TimeTemplate};
use super::spikes::Spike;
use crate::core::REFRACTORY_PERIOD;
use crate::error::SNNError;

/// A spiking neuron whose behavior is depicted in the following block diagram:
///
/// ![A Foobaring][neuron]
///
#[embed_doc_image("neuron", "images/neuron.svg")]
pub trait Neuron: Sync + Send {
    type InputSpike: InputSpike;
    type InputSpikeTrain: InputSpikeTrain<InputSpike = Self::InputSpike>;

    /// Get the neuron's ID.
    fn id(&self) -> usize;

    /// Get the firing threshold of the neuron.
    fn threshold(&self) -> f64;

    /// Init the threshold noise sampler.
    fn init_threshold_sampler(&mut self, sigma: f64);

    /// Sample the firing threshold of the neuron, i.e., apply a random deviation to the nominal threshold.
    fn sample_threshold(&mut self);

    /// Get a reference to the firing times of the neuron.
    fn firing_times_ref(&self) -> &Vec<f64>;

    /// Get a mutable reference to the firing times of the neuron.
    fn firing_times_mut(&mut self) -> &mut Vec<f64>;

    /// Get a reference to the last firing time of the neuron, if any.
    fn last_firing_time(&self) -> Option<&f64> {
        self.firing_times_ref().last()
    }

    /// Add a firing time to the neuron.
    fn push_firing_time(&mut self, time: f64) {
        self.firing_times_mut().push(time);
    }

    /// Make the neuron fire at a given time and apply a threshold deviation.
    fn fire(&mut self, time: f64) {
        self.push_firing_time(time);
        self.sample_threshold();
    }

    /// Get the neuron's potential at a given time.
    fn potential(&self, time: f64) -> f64 {
        self.input_spike_train().potential(time)
    }

    /// A reference to the vector of inputs of the neuron
    fn inputs(&self) -> &Vec<Input>;

    /// A mutable reference to the vector of inputs of the neuron
    fn inputs_mut(&mut self) -> &mut Vec<Input>;

    /// Get a reference to the vector of inputs of the neuron
    fn inputs_iter(&self) -> impl Iterator<Item = &Input> + '_;

    /// Get a reference to the inputs of the neuron.
    fn inputs_iter_mut(&mut self) -> impl Iterator<Item = &mut Input> + '_;

    /// A reference to the ith input.
    fn input_ref(&self, i: usize) -> Option<&Input>;

    /// A mutable reference to a specific input.
    fn input_mut(&mut self, i: usize) -> Option<&mut Input>;

    /// Add an input to the neuron.
    fn add_input(&mut self, source_id: usize, weight: f64, delay: f64) {
        // let input = Input::new(self.num_inputs(), source_id, weight, delay);
        let input = Input::new(source_id, weight, delay);
        self.inputs_mut().push(input);
    }

    /// Get the number of inputs to the neuron.
    fn num_inputs(&self) -> usize {
        self.inputs().len()
    }

    /// A reference to the input spike train of a specific channel.
    fn input_spike_train(&self) -> &Self::InputSpikeTrain;

    /// A mutable reference to the input spike train of a specific channel.
    fn input_spike_train_mut(&mut self) -> &mut Self::InputSpikeTrain;

    // /// Clear the input spike train of the neuron.
    // fn clear_input_spike_train(&mut self);

    // /// Extend the input spike train of the neuron with new input spikes.
    // fn extend_input_spike_train(&mut self, new_input_spike_train: Self::InputSpikeTrain);

    /// Initialize the input spike train of the neuron from the provided spike train.
    fn init_input_spike_train(&mut self, spike_train: &MultiChannelSpikeTrain);

    /// Update the input spikes of the neuron from the provided spike train.
    /// The input spikes are updated by removing all input spikes which are irrelevant from the provided time.
    fn update_input_spikes(
        &mut self,
        time: f64,
        spike_train: &MultiChannelSpikeTrain,
    ) -> Result<(), SNNError>;

    /// Returns the next firing time of the neuron, if any.
    fn next_spike(&self, mut start: f64) -> Option<Spike> {
        if let Some(last) = self.last_firing_time() {
            start = last + REFRACTORY_PERIOD;
        }

        self.input_spike_train()
            .next_potential_threshold_crossing(start, self.threshold())
            .map(|time| Spike {
                neuron_id: self.id(),
                time,
            })
    }

    /// Compute the optimal synaptic weights of the neuron to reproduce the provided spike trains when fed with specific (sorted) input spikes trains.
    /// The input spikes trains fully determine the neuron's behavior on the provided time templates.
    /// The optimization is performed using the Gurobi solver; the optimization objective can be either None, L1, or L2.
    fn memorize(
        &mut self,
        time_templates: Vec<TimeTemplate>,
        mut input_spike_trains: Vec<Self::InputSpikeTrain>,
        lim_weights: (f64, f64),
        max_level: f64,
        min_slope: f64,
        // half_width: f64,
        objective: Objective,
    ) -> Result<(), SNNError> {
        // Initialize the Gurobi environment and model for optimization
        let mut model =
            optim::set_grb_model(format!("neuron_{}", self.id()).as_str(), "gurobi.log")?;
        let weights = optim::set_grb_vars(&mut model, self.num_inputs(), lim_weights)?;
        optim::set_grb_objective(&mut model, &weights, objective)?;

        // Set equality constraints for each firing time
        for (time_template, input_spike_train) in
            time_templates.iter().zip(input_spike_trains.iter())
        {
            for time in time_template.firing_times.iter() {
                self.add_threshold_crossing_potential_cstr(
                    &mut model,
                    &weights,
                    &input_spike_train,
                    *time,
                )?;
            }
        }

        for it in 0..optim::MAX_ITER {
            // For fixed constraints, determine the optimal weights
            log::trace!("Neuron {}: Optimize weights...", self.id());
            model.optimize().map_err(|e| {
                SNNError::OptimizationError(format!(
                    "Error while optimizing neuron {}: {}",
                    self.id(),
                    e
                ))
            })?;

            // If the model is infeasible, return an error
            let status = model.status().map_err(|e| {
                SNNError::OptimizationError(format!(
                    "Error while optimizing neuron {}: {}",
                    self.id(),
                    e
                ))
            })?;

            if Status::Optimal != status {
                return Err(SNNError::InfeasibleMemorization(format!(
                    "Error while optimizing neuron {}",
                    self.id()
                )));
            }

            // Update the input weights
            for (input_id, weight) in weights.iter().enumerate() {
                let weight = model
                    .get_obj_attr(grb::attribute::VarDoubleAttr::X, weight)
                    .map_err(|e| {
                        SNNError::OptimizationError(format!(
                            "Error while optimizing neuron {}: {}",
                            self.id(),
                            e
                        ))
                    })?;
                let input = self.input_mut(input_id).ok_or_else(|| {
                    SNNError::OutOfBounds(format!("Input ID {} out of bounds", input_id))
                })?;
                input.weight = weight;
            }
            // Update the input spike trains
            input_spike_trains
                .iter_mut()
                .try_for_each(|input_spike_train| input_spike_train.update_from(self.inputs()))?;

            // Refine the constraints to ensure the neuron's behavior is consistent with the provided spike trains and time templates
            log::trace!("Neuron {}: Refine constraints...", self.id());
            let new_cstrs = self.refine_constraints(
                &mut model,
                &weights,
                &input_spike_trains,
                &time_templates,
                max_level,
                min_slope,
            )?;
            log::trace!("Neuron {}: {} new constraints added", self.id(), new_cstrs);

            if new_cstrs == 0 {
                // Get the total number of constraints and the objective value
                match objective {
                    Objective::None => {
                        let num_cstrs = model
                            .get_attr(grb::attribute::ModelIntAttr::NumConstrs)
                            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
                        log::info!(
                            "Neuron {}: Optimization succeeded in {} iterations! All {} (time) constraints are satisfied.",
                            self.id(),
                            it,
                            num_cstrs
                        )
                    }
                    Objective::L2 => {
                        let num_cstrs = model
                            .get_attr(grb::attribute::ModelIntAttr::NumConstrs)
                            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
                        let obj_val = model
                            .get_attr(grb::attribute::ModelDoubleAttr::ObjVal)
                            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
                        log::info!(
                            "Neuron {}: Optimization succeeded in {} iterations!! All {} (time) constraints are satisfied for a final cost of {}.",
                            self.id(),
                            it,
                            num_cstrs, obj_val
                        )
                    }
                    Objective::L1 => {
                        let num_cstrs = model
                            .get_attr(grb::attribute::ModelIntAttr::NumConstrs)
                            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
                        let obj_val = model
                            .get_attr(grb::attribute::ModelDoubleAttr::ObjVal)
                            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
                        log::info!(
                            "Neuron {}: Optimization succeeded in {} iterations! All {} (time) constraints are satisfied for a final cost of {}.",
                            self.id(),
                            it,
                            num_cstrs - (self.num_inputs() as i32 * 2),
                            obj_val,
                        )
                    }
                }
                return Ok(());
            }
        }

        Err(SNNError::ConvergenceError(format!(
            "Neuron {}: optimization failed... Maximum number of iterations reached",
            self.id()
        )))
    }

    /// Add an potential threshold-crossing constraint to the model at a given time.
    fn add_threshold_crossing_potential_cstr(
        &self,
        model: &mut Model,
        weights: &Vec<Var>,
        input_spike_train: &Self::InputSpikeTrain,
        time: f64,
    ) -> Result<(), SNNError> {
        let mut expr = grb::expr::LinExpr::new();
        for input_spike in input_spike_train.input_spikes_iter() {
            expr.add_term(
                input_spike.kernel(time - input_spike.time()),
                weights[input_spike.input_id()],
            );
        }
        model
            .add_constr(
                format!("eq_{}", time).as_str(),
                c!(expr == self.threshold()),
            )
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

        log::debug!(
            "Neuron {}: new potential threshold crossing constraint added at time={}",
            self.id(),
            time
        );
        Ok(())
    }

    /// Add a maximum potential constraint to the model at a given time.
    fn add_max_potential_cstr(
        &self,
        model: &mut Model,
        weights: &Vec<Var>,
        input_spike_train: &Self::InputSpikeTrain,
        time: f64,
        max_value: f64,
    ) -> Result<(), SNNError> {
        let mut expr = grb::expr::LinExpr::new();
        for input_spike in input_spike_train.input_spikes_iter() {
            expr.add_term(
                input_spike.kernel(time - input_spike.time()),
                weights[input_spike.input_id()],
            );
        }

        model
            .add_constr(format!("max_{}", time).as_str(), c!(expr <= max_value))
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

        log::debug!(
            "Neuron {}: new maximum potential constraint added at time={}",
            self.id(),
            time
        );
        Ok(())
    }

    /// Add a minimum potential derivative constraint to the model at a given time.
    fn add_min_potential_deriv_cstr(
        &self,
        model: &mut Model,
        weights: &Vec<Var>,
        input_spike_train: &Self::InputSpikeTrain,
        time: f64,
        min_value: f64,
    ) -> Result<(), SNNError> {
        let mut expr = grb::expr::LinExpr::new();
        for input_spike in input_spike_train.input_spikes_iter() {
            expr.add_term(
                input_spike.kernel_deriv(time - input_spike.time()),
                weights[input_spike.input_id()],
            );
        }

        model
            .add_constr(format!("min_{}", time).as_str(), c!(expr >= min_value))
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

        log::debug!(
            "Neuron {}: new minimum potential derivative constraint added at time={}",
            self.id(),
            time
        );
        Ok(())
    }

    /// Refine the constraints of the model and return the number of new constraints added.
    fn refine_constraints(
        &self,
        model: &mut Model,
        weights: &Vec<Var>,
        input_spike_trains: &Vec<Self::InputSpikeTrain>,
        time_templates: &Vec<TimeTemplate>,
        max_level: f64,
        min_slope: f64,
    ) -> Result<usize, SNNError> {
        for (time_template, input_spike_train) in
            time_templates.iter().zip(input_spike_trains.iter())
        {
            // 1. Check for maximum level constraint
            let pairs = input_spike_train
                .max_potential_in_windows(&time_template.silence_regions, optim::MAX_NEW_CSTRS);
            for pair in pairs.iter() {
                if pair.value > max_level + optim::FEASIBILITY_TOL {
                    self.add_max_potential_cstr(
                        model,
                        &weights,
                        input_spike_train,
                        pair.time,
                        max_level,
                    )?;
                };
            }

            // 2. Check for minimum slope constraint
            let pairs = input_spike_train.min_potential_deriv_in_windows(
                &time_template.active_regions,
                optim::MAX_NEW_CSTRS,
            );
            for pair in pairs.iter() {
                if pair.value < min_slope - optim::FEASIBILITY_TOL {
                    log::debug!("Neuron {}: adding min slope constraint at {}, current value is {} (should be > {})", self.id(), pair.time, pair.value, min_slope);
                    self.add_min_potential_deriv_cstr(
                        model,
                        &weights,
                        &input_spike_train,
                        pair.time,
                        min_slope,
                    )?;
                };
            }
        }

        let prev_num_cstrs = model
            .get_attr(grb::attribute::ModelIntAttr::NumConstrs)
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

        model
            .update()
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

        let num_cstrs = model
            .get_attr(grb::attribute::ModelIntAttr::NumConstrs)
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

        let added_cstrs = (num_cstrs - prev_num_cstrs) as usize;
        Ok(added_cstrs)
    }
}

/// An input connection to a neuron.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Input {
    // /// The input id.
    // pub input_id: usize,
    /// The ID of the neuron producing the spike.
    pub source_id: usize,
    /// The weight of the synapse along which the spikes are received.
    pub weight: f64,
    /// The delay of the synapse along which the spikes are received.
    pub delay: f64,
}

impl Input {
    // pub fn new(input_id: usize, source_id: usize, weight: f64, delay: f64) -> Self {
    pub fn new(source_id: usize, weight: f64, delay: f64) -> Self {
        Input {
            // input_id,
            source_id,
            weight,
            delay,
        }
    }
}

/// An input spike along a specific input.
pub trait InputSpike: PartialOrd {
    fn input_id(&self) -> usize;
    fn time(&self) -> f64;
    fn weight(&self) -> f64;
    fn kernel(&self, dt: f64) -> f64;
    fn kernel_deriv(&self, dt: f64) -> f64;
    fn new(input_id: usize, time: f64, weight: f64) -> Self;
}

pub trait InputSpikeTrain {
    type InputSpike: InputSpike;

    /// Create an empty input spike train to a neuron.
    fn new_empty() -> Self;

    /// Create an input spike train to a neuron from a cyclic spike train.
    /// Input spikes are periodically extended to cover the whole provided time interval.
    /// Hence, every spike has a non-neglible effect on the neuron's behavior at some point in the interval.
    fn new_cyclic_from(
        inputs: &Vec<Input>,
        spike_train: &MultiChannelCyclicSpikeTrain,
        interval: &TimeInterval,
    ) -> Self;

    /// Create an input spike train to a neuron from a cyclic spike train.
    /// Input spikes are periodically extended to cover the whole provided time interval.
    /// Hence, every spike has a non-neglible effect on the neuron's behavior at some point in the interval.
    fn new_from(inputs: &Vec<Input>, spike_train: &MultiChannelSpikeTrain) -> Self;

    fn input_spikes_iter(&self) -> impl Iterator<Item = &Self::InputSpike> + '_;
    fn input_spikes_iter_mut(&mut self) -> impl Iterator<Item = &mut Self::InputSpike> + '_;

    fn len(&self) -> usize {
        self.input_spikes_iter().count()
    }

    fn update_from(&mut self, inputs: &Vec<Input>) -> Result<(), SNNError>;
    fn insert(&mut self, new_input_spike: Self::InputSpike) -> usize;
    fn merge(&mut self, new_input_spikes: Self);

    /// Returns the next potential threshold crossing time after the provided time.
    /// TODO: Default implementation with [Brent’s method](https://en.wikipedia.org/wiki/Brent%27s_method) for root finding,
    /// e.g., see [this crate](https://argmin-rs.github.io/argmin/argmin/solver/brent/index.html).
    fn next_potential_threshold_crossing(&self, time: f64, threshold: f64) -> Option<f64>;

    /// Returns the k largest maximum potentials in the provided time windows.
    fn max_potential_in_windows(
        &self,
        windows: &TimeIntervalUnion,
        k: usize,
    ) -> Vec<TimeValuePair<f64>>;

    /// Returns the k smallest minimum potential derivatives in the provided time windows.
    fn min_potential_deriv_in_windows(
        &self,
        windows: &TimeIntervalUnion,
        k: usize,
    ) -> Vec<TimeValuePair<f64>>;

    /// Default implementation of the potential.
    fn potential(&self, time: f64) -> f64 {
        self.input_spikes_iter()
            .map(|input_spike| input_spike.kernel(time - input_spike.time()) * input_spike.weight())
            .sum()
    }

    /// Default implementation of the potential derivative.
    fn potential_deriv(&self, time: f64) -> f64 {
        self.input_spikes_iter()
            .map(|input_spike| {
                input_spike.kernel_deriv(time - input_spike.time()) * input_spike.weight()
            })
            .sum()
    }
}
