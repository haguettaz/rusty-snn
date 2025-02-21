//! Error module for the Rusty SNN library.
//!

use thiserror::Error;

/// Error types for the library.
#[derive(Debug, Error, PartialEq)]
pub enum SNNError {
    #[error("Out of bounds access: {0}")]
    OutOfBounds(String),
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    #[error("Optimization error: {0}")]
    OptimizationError(String),
    #[error("Convergence error: {0}")]
    ConvergenceError(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("I/O error: {0}")]
    IOError(String),
}