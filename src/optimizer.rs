//! This module contains the simulation program and interval structures.
//!
use super::error::SNNError;
use super::spike_train::SpikeTrain;

use itertools::izip;
use nalgebra::{DMatrix, DVector};

use super::neuron::{FIRING_THRESHOLD, REFRACTORY_PERIOD};

// /// Minimum potential derivative around firing times
// pub const MIN_SLOPE: f64 = 2.0 * FIRING_THRESHOLD / REFRACTORY_PERIOD;
// pub const MIN_GAP: f64 = FIRING_THRESHOLD;
// pub const ACTIVITY_WINDOW: f64 = 0.2 * REFRACTORY_PERIOD;


pub struct OptimConfig {
    activity_window: f64,
    slope_min: f64,
    gap_min: f64,
    weight_min: f64,
    weight_max: f64,
    l2_norm: bool,
}

impl OptimConfig {
    pub fn new(
        activity_window: f64,
        slope_min: f64,
        gap_min: f64,
        weight_min: f64,
        weight_max: f64,
        l2_norm: bool,
    ) -> Self {
        Self {
            activity_window,
            slope_min,
            gap_min,
            weight_min,
            weight_max,
            l2_norm,
        }
    }

    pub fn slope_min(&self) -> f64 {
        self.slope_min
    }

    pub fn gap_min(&self) -> f64 {
        self.gap_min
    }

    pub fn activity_window(&self) -> f64 {
        self.activity_window
    }

    pub fn weight_min(&self) -> f64 {
        self.weight_min
    }

    pub fn weight_max(&self) -> f64 {
        self.weight_max
    }

    pub fn l2_norm(&self) -> bool {
        self.l2_norm
    }
}



// // use std::cmp::Ordering;
// // use std::collections::HashSet;
// // use std::error::Error;
// // use std::fmt;

// // /// Represents a half-open time interval [start, end).
// // #[derive(Debug, PartialEq)]
// // pub struct TimeInterval {
// //     start: f64,
// //     end: f64,
// // }

// // /// Implement the partial ordering for time intervals.
// // /// The ordering is based on the start and end times of the intervals.
// // /// A time interval is less than another if it ends before the other starts, and vice versa.
// // /// Overlapping intervals are not ordered.
// // impl PartialOrd for TimeInterval {
// //     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
// //         if self.end <= other.start {
// //             Some(Ordering::Less)
// //         } else if self.start >= other.end {
// //             Some(Ordering::Greater)
// //         } else {
// //             None
// //         }
// //     }
// // }

// // impl TimeInterval {
// //     /// Create a time interval with the specified parameters.
// //     pub fn build(start: f64, end: f64) -> Result<Self, SimulationError> {
// //         if !(start.is_finite() && end.is_finite()) {
// //             return Err(SimulationError::HalfInfiniteTimeInterval);
// //         }

// //         if start >= end {
// //             return Err(SimulationError::EmptyTimeInterval);
// //         }

// //         Ok(TimeInterval { start, end })
// //     }

// //     /// Returns the start time of the time interval.
// //     pub fn start(&self) -> f64 {
// //         self.start
// //     }

// //     /// Returns the end time of the time interval.
// //     pub fn end(&self) -> f64 {
// //         self.end
// //     }
// // }

// /// Represents a simulation program with a time interval, neuron control, and threshold noise.
// #[derive(Debug, PartialEq)]
// pub struct OptimizationProgram<'a> {
//     spike_trains: &'a [SpikeTrain],
//     min_slope: f64,
//     min_gap: f64,
//     activivity_window: f64,
//     norm: Option<usize>,
// }

// impl OptimizationProgram<'_> {
//     /// Create a simulation interval with the specified parameters.
//     /// The function returns an error for invalid simulation intervals or control.
//     pub fn build<'a>(
//         spike_trains: &'a [SpikeTrain],
//         norm: Option<usize>,
//     ) -> Result<OptimizationProgram<'a>, SNNError> {
//         todo!()
//     }

//     pub fn norm(&self) -> Option<usize> {
//         self.norm
//     }
// }

// pub struct Constraint {
//     cn: DVector<f64>,
//     yn: f64,
// }

// pub struct MinNormOptimizer {
//     num_variables: usize,
//     norm: Option<usize>,
//     constraints: Vec<Constraint>,
// }

// impl MinNormOptimizer {
//     pub fn new(num_variables: usize, norm: Option<usize>) -> Self {
//         Self {
//             num_variables,
//             norm,
//             constraints: Vec::new(),
//         }
//     }

//     // Add a linear constraint cn x <= yn to the optimization problem.
//     pub fn add_linear_leq(&mut self, cn: DVector<f64>, yn: f64) {
//         self.constraints.push(Constraint { cn, yn });
//     }

//     // Add a linear constraint cn x == yn to the optimization problem.
//     // This effectively removes a free variable from the optimization problem.
//     pub fn add_linear_eq(&mut self, cn: DVector<f64>, yn: f64) {
//         // first, get the first non-zero element of cn, associated with a free variable
//         // all past and future leq constraints will be adapted to this equality constraint
//         todo!();
//     }

//     /// Determine the solution of the inequality constrained min-norm problem using the dual forward filtering backward deciding algorithm
//     /// See Li and Loeliger, 2025.
//     /// How to check for convergence? How to set gamma to enforce constraint satisfaction?
//     pub fn dual_forward_filtering_backward_deciding(&self) -> Result<DVector<f64>, &'static str> {
//         // Allocate memory for primal and dual quantities
//         let mut x: DVector<f64> = DVector::zeros(self.num_variables);

//         let mut mfx: DVector<f64> = DVector::zeros(self.num_variables);
//         let mut vfx: DMatrix<f64> = DMatrix::identity(self.num_variables, self.num_variables);
//         let mut cmfx: Vec<f64> = Vec::with_capacity(self.constraints.len());
//         let mut cvfx: Vec<DVector<f64>> = Vec::with_capacity(self.constraints.len());

//         let mut xitx: DVector<f64> = DVector::zeros(self.num_variables);
//         let mut mby: Vec<f64> = Vec::with_capacity(self.constraints.len());
//         let mut vby: Vec<f64> = Vec::with_capacity(self.constraints.len());

//         loop {
//             // Primal Forward Filtering
//             mfx = DVector::zeros(self.num_variables);
//             vfx = DMatrix::identity(self.num_variables, self.num_variables);
//             for (constraint, mbyn, vbyn, cmfxn, cvfxn) in
//                 izip!(self.constraints.iter(), mby.iter(), vby.iter(), cmfx.iter_mut(), cvfx.iter_mut())
//             {
//                 constraint.cn.mul_to(&vfx, cvfxn);
//                 *cmfxn = constraint.cn.dot(&mfx);
//                 let gain = 1.0 / (vbyn + constraint.cn.dot(&cvfxn));
//                 let residual = mbyn - *cmfxn;
//                 vfx.syger(-gain, &cvfxn, &cvfxn, 1.0);
//                 mfx.axpy(gain * residual, &cvfxn, 1.0);
//             }

//             // Dual Backward Deciding and NUP Updating
//             xitx = DVector::zeros(self.num_variables);
//             for (constraint, mbyn, vbyn, cmfxn, cvfxn) in
//                 izip!(self.constraints.iter(), mby.iter_mut(), vby.iter_mut(), cmfx.iter(), cvfx.iter()).rev()
//             {
//                 let wfyn = 1.0 / constraint.cn.dot(&cvfxn);
//                 let xifyn = cmfxn - xitx.dot(&cvfxn);

//                 // Dual deciding
//                 let xityn = match xifyn > constraint.yn * wfyn {
//                     true => xifyn - constraint.yn * wfyn,
//                     false => 0.0,
//                 };
//                 xitx.axpy(xityn, &constraint.cn, 1.0);

//                 // NUP updating
//                 todo!();
//             }

//             // Final Estimation
//             x = -xitx;
//         }

//         Ok(x)
//     }

//     /// Determine the solution of the inequality constrained min-norm problem using the gurobi solver.
//     pub fn gurobi_callback(&self) -> Result<DVector<f64>, &'static str> {
//         todo!();
//     }

//     pub fn solve() {
//         todo!()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
}
