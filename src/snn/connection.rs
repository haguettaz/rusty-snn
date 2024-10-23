use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Input {
    source_id: usize,
    delay: f64,
    weight: f64,
    kernel: Kernel,
    firing_times: Vec<f64>,
}

impl Input {
    pub fn new(source_id: usize, delay: f64, weight: f64, order: i32, beta: f64) -> Input {
        Input {
            source_id: source_id,
            delay: delay,
            weight: weight,
            kernel: Kernel::new(order, beta),
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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Kernel {
    order: i32,
    beta: f64,
    gamma: f64,
}

impl Kernel {
    // Creates a new (energy-normalized) kernel with a given delay, weight, order, and beta (=time constant).
    pub fn new(order: i32, beta: f64) -> Kernel {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    #[test]
    fn test_input() {
        let input = Input::new(0, 1.0, 1.0, 1, 1.0);
        assert_eq!(input.source_id, 0);
        assert_eq!(input.delay, 1.0);
        assert_eq!(input.weight, 1.0);
        assert_eq!(input.kernel.order, 1);
        assert_eq!(input.kernel.beta, 1.0);
        assert_eq!(input.firing_times.len(), 0);
    }

    #[test]
    fn test_kernel() {
        let kernel = Kernel::new(1, 1.0);
        assert_eq!(kernel.order, 1);
        assert_eq!(kernel.beta, 1.0);
        assert!((kernel.gamma - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_apply() {
        let kernel = Kernel::new(1, 1.0);
        assert!((kernel.apply(0.0)).abs() < 1e-10);
        assert!((kernel.apply(1.0) - 2.0 / E).abs() < 1e-10);
    }

    #[test]
    fn test_input_apply() {
        let mut input = Input::new(0, 1.0, 1.0, 1, 1.0);
        input.add_firing_time(0.0);
        input.add_firing_time(1.0);
        assert!((input.apply(0.0)).abs() < 1e-10);
        assert!((input.apply(1.0)).abs() < 1e-10);
        assert!((input.apply(2.0) - 2.0 / E).abs() < 1e-10);
    }
}
