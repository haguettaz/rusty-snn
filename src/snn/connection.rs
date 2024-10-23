pub struct Input {
    source_id: usize,
    delay: f64,
    kernel: Kernel,
    firing_times: Vec<f64>,
}

impl Input {
    pub fn new(
        source_id: usize,
        delay: f64,
        weight: f64,
        order: i32,
        beta: f64,
    ) -> Input {
        Input {
            source_id: source_id,
            delay: delay,
            kernel: Kernel::new(weight, order, beta),
            firing_times: Vec::new(),
        }
    }

    pub fn add_firing_time(&mut self, time: f64) {
        self.firing_times.push(time + self.delay);
    }

    pub fn apply(&self, time: f64) -> f64 {
        self.firing_times.iter().map(|ft| self.kernel.apply(time - ft)).sum()
    }
}

pub struct Kernel {
    weight: f64,
    order: i32,
    beta: f64,
    gamma: f64,
}

impl Kernel {
    // Creates a new (energy-normalized) kernel with a given delay, weight, order, and beta (=time constant).
    pub fn new(weight: f64, order: i32, beta: f64) -> Kernel {
        let ln_gamma2: f64 = (1..=2 * order).fold(
            ((2 * order + 1) as f64) * (2.0 * beta as f64).ln(),
            |acc, n| acc - (n as f64).ln(),
        );
        let gamma = (0.5 * ln_gamma2).exp();

        Kernel {
            weight,
            order,
            beta,
            gamma,
        }
    }

    pub fn apply(&self, time: f64) -> f64 {
        self.weight
            * self.gamma
            * time.powi(self.order)
            * (- self.beta * time).exp()
    }
}
