use serde::{Deserialize, Serialize};
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Input {
    source_id: usize,
    weight: f64,
    delay: f64,
    kernel: Kernel,
    firing_times: Vec<f64>,
}

impl Input {
    pub fn build(source_id: usize, weight: f64, delay: f64, order: i32, beta: f64) -> Input {
        if delay <= 0.0 {
            panic!("Delay must be positive.");
        }

        Input {
            source_id: source_id,
            weight: weight,
            delay: delay,
            kernel: Kernel::build(order, beta),
            firing_times: Vec::new(),
        }
    }

    pub fn add_firing_time(&mut self, time: f64) {
        self.firing_times.push(time);
    }

    pub fn apply(&self, time: f64) -> f64 {
        self.firing_times
            .iter()
            .filter(|&&ft| ft + self.delay < time)
            .map(|ft| self.weight * self.kernel.apply(time - ft - self.delay))
            .sum()
    }

    pub fn firing_times(&self) -> &Vec<f64> {
        &self.firing_times
    }

    pub fn weight(&self) -> f64 {
        self.weight
    }

    pub fn delay(&self) -> f64 {
        self.delay
    }

    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    pub fn source_id(&self) -> usize {
        self.source_id
    }
}

#[derive(Debug, Serialize, Deserialize)]
/// Implements a normalized function kernel for synaptic response.
/// The kernel is defined as: γ * t^n * exp(-βt) for t > 0 where:
/// - n is the order
/// - β (beta) is the time constant
/// - γ (gamma) is the normalization factor
#[derive(PartialEq, Clone)]
pub struct Kernel {
    order: i32,
    beta: f64,
    gamma: f64,
}

impl Kernel {
    pub fn build(order: i32, beta: f64) -> Kernel {
        if order <= 0 {
            panic!("Order must be positive.");
        }
        if beta <= 0.0 {
            panic!("Beta must be positive.");
        }

        let ln_gamma2: f64 = (1..=2 * order).fold(
            ((2 * order + 1) as f64) * (2.0 * beta as f64).ln(),
            |acc, n| acc - (n as f64).ln(),
        );
        let gamma = (0.5 * ln_gamma2).exp();

        Kernel { order, beta, gamma }
    }

    pub fn apply(&self, time: f64) -> f64 {
        self.gamma * time.powi(self.order) * (-self.beta * time).exp()
    }

    pub fn order(&self) -> i32 {
        self.order
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    #[test]
    fn test_input() {
        let input = Input::build(0, 1.0, 1.0, 1, 1.0);
        assert_eq!(input.source_id, 0);
        assert_eq!(input.delay, 1.0);
        assert_eq!(input.weight, 1.0);
        assert_eq!(input.kernel.order, 1);
        assert_eq!(input.kernel.beta, 1.0);
        assert_eq!(input.firing_times.len(), 0);
    }

    #[test]
    #[should_panic(expected = "Delay must be positive.")]
    fn test_input_rejects_negative_delay() {
        Input::build(0, 1.0, -1.0, 1, 1.0);
    }

    #[test]
    fn test_kernel() {
        let kernel = Kernel::build(1, 1.0);
        assert_eq!(kernel.order, 1);
        assert_eq!(kernel.beta, 1.0);
        assert!((kernel.gamma - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_apply() {
        let kernel = Kernel::build(1, 1.0);
        assert!((kernel.apply(0.0)).abs() < 1e-10);
        assert!((kernel.apply(1.0) - 2.0 / E).abs() < 1e-10);
    }

    #[test]
    fn test_input_apply() {
        let mut input = Input::build(0, 1.0, 1.0, 1, 1.0);
        input.add_firing_time(0.0);
        input.add_firing_time(1.0);
        assert!((input.apply(0.0)).abs() < 1e-10);
        assert!((input.apply(1.0)).abs() < 1e-10);
        assert!((input.apply(2.0) - 2.0 / E).abs() < 1e-10);
    }

    #[test]
    fn test_input_clone() {
        let input = Input::build(0, 1.0, 1.0, 1, 1.0);
        let cloned = input.clone();
        assert_eq!(input, cloned);
    }

    #[test]
    fn test_kernel_clone() {
        let kernel = Kernel::build(1, 1.0);
        let cloned = kernel.clone();
        assert_eq!(kernel, cloned);
    }

    #[test]
    fn test_kernel_numerical_stability() {
        let kernel = Kernel::build(1, 1.0);  // High order, small beta
        assert!((kernel.apply(100.0)).abs() < 1e-10);  // Should decay to ~0
    }
}
