//! Neuron-related module.
use embed_doc_image::embed_doc_image;
use grb::prelude::*;
use log;
use serde::{Deserialize, Serialize};

use crate::core::utils::{TimeInterval, TimeIntervalUnion, TimeValuePair};

/// The minimum value of an input spike contribution before it is considered negligible.
pub const MIN_INPUT_VALUE: f64 = 1e-12;

use super::optim;
use super::optim::{Objective, TimeTemplate};
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
    fn init_threshold_sampler(&mut self, sigma: f64, seed: u64);

    /// Sample the firing threshold of the neuron, i.e., apply a random deviation to the nominal threshold.
    fn sample_threshold(&mut self);

    /// Get a reference to the firing times of the neuron.
    fn ftimes_ref(&self) -> &Vec<f64>;

    /// Get a mutable reference to the firing times of the neuron.
    fn ftimes_mut(&mut self) -> &mut Vec<f64>;

    /// Get a reference to the last firing time of the neuron, if any.
    fn last_ftime(&self) -> Option<&f64> {
        self.ftimes_ref().last()
    }

    /// Add a firing time to the neuron.
    fn push_ftime(&mut self, time: f64) {
        self.ftimes_mut().push(time);
    }

    /// Make the neuron fire at a given time and apply a threshold deviation.
    fn fire(&mut self, time: f64) {
        self.push_ftime(time);
        self.sample_threshold();
    }

    /// Get the neuron's potential at a given time.
    fn potential(&self, time: f64) -> f64 {
        self.input_spike_train().potential(time)
    }

    /// Returns a slice of inputs of the neuron.
    fn inputs(&self) -> &[Input];

    /// Returns a mutable reference to the vector of inputs of the neuron
    fn inputs_mut(&mut self) -> &mut Vec<Input>;

    /// Returns an iterator over the inputs of the neuron.
    fn inputs_iter(&self) -> impl Iterator<Item = &Input> + '_;

    /// Returns a mutable iterator over the inputs of the neuron.
    fn inputs_iter_mut(&mut self) -> impl Iterator<Item = &mut Input> + '_;

    /// Returns a reference to a specific input.
    fn input_ref(&self, i: usize) -> Option<&Input>;

    /// Returns a mutable reference to a specific input.
    fn input_mut(&mut self, i: usize) -> Option<&mut Input>;

    /// Push an input to the neuron.
    fn push_input(&mut self, source_id: usize, weight: f64, delay: f64) {
        let input = Input::new(source_id, weight, delay);
        self.inputs_mut().push(input);
    }

    /// Get the number of inputs to the neuron.
    fn num_inputs(&self) -> usize {
        self.inputs().len()
    }

    /// Clear the inputs to the neuron.
    fn clear_inputs(&mut self) {
        self.inputs_mut().clear();
    }

    /// A reference to the input spike train of a specific channel.
    fn input_spike_train(&self) -> &Self::InputSpikeTrain;

    /// A mutable reference to the input spike train of a specific channel.
    fn input_spike_train_mut(&mut self) -> &mut Self::InputSpikeTrain;

    /// Initialize the input spike train of the neuron from the provided spike train.
    fn init_input_spike_train(&mut self, spike_train: &Vec<Vec<f64>>);

    /// Clean the input spikes of the neuron, by draining all spikes which are irrelevant after the provided time.
    fn drain_input_spike_train(&mut self, time: f64);

    /// Receive and process the spikes from the input channels.
    /// The spikes are provided as a vector of optional firing times.
    /// There is at most one spike per channel.
    fn receive_spikes(&mut self, ftimes: &Vec<Option<f64>>);

    /// Returns the next time at which the neuron will fire after the given start time, if any.
    /// This method enforces the refractory period.
    ///
    /// # Important
    /// Multiple spikes cannot occur back-to-back in a single call, as each spike must be
    /// separated by at least the refractory period. This means that if the neuron has fired
    /// recently, the earliest possible next spike will be forcibly delayed until after the
    /// refractory period, regardless of input strength.
    ///
    /// # Arguments
    /// * `start` - The time from which to start looking for the next spike
    ///
    /// # Returns
    /// * `Some(time)` - The time of the next spike if one is found
    /// * `None` - If no future spike is detected
    fn next_spike(&self, mut start: f64) -> Option<f64> {
        if let Some(last) = self.last_ftime() {
            start = last + REFRACTORY_PERIOD;
        }

        self.input_spike_train()
            .next_potential_threshold_crossing(start, self.threshold())
    }

    /// Compute the optimal synaptic weights of the neuron to reproduce the provided spike trains when fed with specific (sorted) input spikes trains.
    /// The input spikes trains fully determine the neuron's behavior on the provided time templates.
    /// The optimization is performed using the Gurobi solver; the objective function is specified by the user.
    fn memorize(
        &mut self,
        time_templates: Vec<TimeTemplate>,
        mut input_spike_trains: Vec<Self::InputSpikeTrain>,
        lim_weights: (f64, f64),
        max_level: f64,
        min_slope: f64,
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
            for time in time_template.ftimes.iter() {
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
            log::trace!("Neuron {}: optimize weights...", self.id());
            model.optimize().map_err(|e| {
                SNNError::OptimizationError(format!(
                    "Neuron {}: error while optimizing the model ({})",
                    self.id(),
                    e
                ))
            })?;

            // Needs to properly handle the status, see https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html
            let status = model.status().map_err(|e| {
                SNNError::InvalidOperation(format!(
                    "Neuron {}: error while checking the model status ({})",
                    self.id(),
                    e
                ))
            })?;
            match status {
                Status::Optimal => {}
                Status::InfOrUnbd | Status::Infeasible | Status::Unbounded => {
                    log::error!("Neuron {}: the model is infeasible", self.id(),);
                    return Err(SNNError::OptimizationError(format!(
                        "Neuron {}: the model is infeasible",
                        self.id(),
                    )));
                }
                _ => {
                    log::error!(
                        "Neuron {}: unexpected error while optimizing... ({:?}). This is typically due to numerical issues encountered at the edge of feasibility.",
                        self.id(),
                        status
                    );
                    return Err(SNNError::OptimizationError(format!(
                        "Neuron {}: unexpected error while optimizing... ({:?}). This is typically due to numerical issues encountered at the edge of feasibility.",
                        self.id(),
                        status
                    )));
                }
            };

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
                .try_for_each(|input_spike_train| {
                    input_spike_train.apply_weight_change(self.inputs())
                })?;

            // Refine the constraints to ensure the neuron's behavior is consistent with the provided spike trains and time templates
            log::trace!("Neuron {}: refine constraints...", self.id());
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
                return match objective {
                    Objective::None => {
                        let num_cstrs = model
                            .get_attr(grb::attribute::ModelIntAttr::NumConstrs)
                            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;
                        log::info!(
                            "Neuron {}: Optimization succeeded in {} iterations! All {} (time) constraints are satisfied.",
                            self.id(),
                            it,
                            num_cstrs
                        );
                        Ok(())
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
                        );
                        Ok(())
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
                        );
                        Ok(())
                    }
                    _ => Err(SNNError::NotImplemented(
                        "Objective not implemented".to_string(),
                    )),
                };
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
        for input_spike in input_spike_train.iter() {
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
        for input_spike in input_spike_train.iter() {
            expr.add_term(
                input_spike.kernel(time - input_spike.time()),
                weights[input_spike.input_id()],
            );
        }

        model
            .add_constr(format!("max_{}", time).as_str(), c!(expr <= max_value))
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

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
        for input_spike in input_spike_train.iter() {
            expr.add_term(
                input_spike.kernel_deriv(time - input_spike.time()),
                weights[input_spike.input_id()],
            );
        }

        model
            .add_constr(format!("min_{}", time).as_str(), c!(expr >= min_value))
            .map_err(|e| SNNError::OptimizationError(e.to_string()))?;

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
                if pair.value > max_level + optim::REF_FEASIBILITY_TOL {
                    self.add_max_potential_cstr(
                        model,
                        &weights,
                        input_spike_train,
                        pair.time,
                        max_level,
                    )?;
                    log::debug!(
                        "Neuron {}: new maximum potential constraint added at time={} (constraint is violated by {})",
                        self.id(),
                        pair.time,
                        max_level - pair.value
                    );
                };
            }

            // 2. Check for minimum slope constraint
            let pairs = input_spike_train.min_potential_deriv_in_windows(
                &time_template.active_regions,
                optim::MAX_NEW_CSTRS,
            );
            for pair in pairs.iter() {
                if pair.value < min_slope - optim::REF_FEASIBILITY_TOL {
                    self.add_min_potential_deriv_cstr(
                        model,
                        &weights,
                        &input_spike_train,
                        pair.time,
                        min_slope,
                    )?;
                    log::debug!(
                        "Neuron {}: new minimum potential derivative constraint added at time={} (constraint is violated by {})",
                        self.id(),
                        pair.time,
                        pair.value - min_slope
                    );
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

/// A collection of input spikes to a neuron.
pub trait InputSpikeTrain {
    type InputSpike: InputSpike;

    /// An empty input spike train to a neuron.
    fn new_empty() -> Self;

    /// Returns a collection of input spikes (an input spike train) to a neuron from a collection of inputs and (cyclic) firing times (time of emission).
    /// Only the periodically extended input spikes which are non negligible in the provided time interval are kept.
    fn new_cyclic_from(
        inputs: &[Input],
        ftimes: &Vec<Vec<f64>>,
        period: f64,
        interval: &TimeInterval,
    ) -> Self;

    /// Returns a collection of input spikes (an input spike train) to a neuron from a collection of inputs and firing times (time of emission).
    fn new_from(inputs: &[Input], ftimes: &Vec<Vec<f64>>) -> Self;

    /// An iterator over the input spikes.
    fn iter(&self) -> impl Iterator<Item = &Self::InputSpike> + '_;

    /// A mutable iterator over the input spikes.
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::InputSpike> + '_;

    /// The number of input spikes in the spike train.
    fn len(&self) -> usize {
        self.iter().count()
    }

    /// Update the input spikes after a change in the input weights.
    fn apply_weight_change(&mut self, inputs: &[Input]) -> Result<(), SNNError>;

    /// Insert a collection of (sorted) input spikes into the input spike train.
    fn insert_sorted(&mut self, new_input_spike: Vec<Self::InputSpike>);

    /// Returns the next potential threshold crossing time after the provided time.
    /// In future version, one could use a default implementation using [Brent’s method](https://en.wikipedia.org/wiki/Brent%27s_method) for root finding,
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

    /// Evaluate the neuron's potential at a given time.
    fn potential(&self, time: f64) -> f64 {
        self.iter()
            .map(|input_spike| input_spike.kernel(time - input_spike.time()) * input_spike.weight())
            .sum()
    }

    /// Evaluate the derivative of the neuron's potential at a given time.
    fn potential_deriv(&self, time: f64) -> f64 {
        self.iter()
            .map(|input_spike| {
                input_spike.kernel_deriv(time - input_spike.time()) * input_spike.weight()
            })
            .sum()
    }
}
