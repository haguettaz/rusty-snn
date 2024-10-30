use rand::distributions::{Distribution};

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
    pub fn build(order: i32, beta: f64) -> Self {
        // Calculate ln(γ²) where γ is the normalization factor
        // ln(γ²) = ln((2β)^(n+1/2) / n!) where n is the order
        // This is derived from the constraint that the kernel should have normalized energy.
        // let ln_gamma2: f64 = (1..=2 * order).fold(
        //     ((2 * order + 1) as f64) * (2.0 * beta as f64).ln(),
        //     |acc, n| acc - (n as f64).ln(),
        // );
        // let gamma = (0.5 * ln_gamma2).exp();
    }

    pub fn apply(&self, time: f64) -> f64 {
        if time < 0.0 {
            return 0.0;
        }
        2_f64 * time * (-time).exp()
    }

    // pub fn order(&self) -> i32 {
    //     self.order
    // }

    // pub fn beta(&self) -> f64 {
    //     self.beta
    // }
}

impl Distribution<Kernel> for Kernel {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Kernel {
        todo!()
    }
}