use crate::core::neuron::Input;
use crate::core::spike::Spike;
use crate::core::metrics::*;

#[derive(Debug)]
pub struct AlphaLinearJitterPropagator {
    trans: Vec<Vec<f64>>,
}

impl AlphaLinearJitterPropagator {
    fn kernel(dt: f64) -> f64 {
        if dt <= 0.0 {
            0.0
        } else {
            (1.0 - dt) * (1.0 - dt).exp()
        }
    }

    pub fn new(connections: Vec<&Vec<Input>>, times: &Vec<Vec<f64>>, period: f64) -> Self {
        let mut spikes = times
            .iter()
            .enumerate()
            .flat_map(|(id, ctimes)| {
                ctimes.iter().map(move |time| Spike {
                    time: *time,
                    neuron_id: id,
                })
            })
            .collect::<Vec<Spike>>();
        spikes.sort_by(|spike_1, spike_2| spike_1.partial_cmp(spike_2).unwrap());

        let mut trans: Vec<Vec<f64>> = vec![vec![0.0; spikes.len()]; spikes.len()];
        for (j, target_spike) in spikes.iter().enumerate() {
            for (i, source_spike) in spikes.iter().enumerate() {
                trans[j][i] = connections[target_spike.neuron_id]
                    .iter()
                    .filter_map(|input| {
                        if input.source_id == source_spike.neuron_id {
                            Some(
                                input.weight
                                    * Self::kernel(
                                        (target_spike.time - source_spike.time - input.delay)
                                            .rem_euclid(period),
                                    ),
                            )
                        } else {
                            None
                        }
                    })
                    .sum();
            }

            let a: f64 = trans[j].iter().sum();
            mult_scalar_in(&mut trans[j], 1.0 / a);
        }

        Self { trans }
    }
}

impl RealLinearOperator for AlphaLinearJitterPropagator {
    /// The dimension of the linear operator.
    fn dim(&self) -> usize {
        self.trans.len()
    }

    /// Apply the jitter linear transformation to the jitter vector (in-place).
    fn apply_in(&self, jitter: &mut Vec<f64>) {
        let mean_jitter = mean(jitter);
        for i in 0..self.dim() {
            jitter[i] = inner(&self.trans[i], jitter);
        }
        sub_scalar_in(jitter, mean_jitter);
    }
}

#[cfg(test)]
mod tests {
    use core::panic;

    use super::*;
    use approx::assert_relative_eq;

    const SEED: u64 = 42;

    #[test]
    fn test_jitter_spectral_radius_real() {
        let inputs_0 = vec![
            Input::new(1, 1.0, 0.0),
            Input::new(2, 1.0, 0.0),
            Input::new(3, 1.0, 0.0),
            Input::new(4, 1.0, 0.0),
        ];
        let inputs_1 = vec![
            Input::new(0, 1.0, 0.0),
            Input::new(2, 1.0, 0.0),
            Input::new(3, 1.0, 0.0),
            Input::new(4, 1.0, 0.0),
        ];
        let inputs_2 = vec![
            Input::new(0, 1.0, 0.0),
            Input::new(1, 1.0, 0.0),
            Input::new(3, 1.0, 0.0),
            Input::new(4, 1.0, 0.0),
        ];
        let inputs_3 = vec![
            Input::new(0, 1.0, 0.0),
            Input::new(1, 1.0, 0.0),
            Input::new(2, 1.0, 0.0),
            Input::new(4, 1.0, 0.0),
        ];
        let inputs_4 = vec![
            Input::new(0, 1.0, 0.0),
            Input::new(1, 1.0, 0.0),
            Input::new(2, 1.0, 0.0),
            Input::new(3, 1.0, 0.0),
        ];
        let connections = vec![&inputs_0, &inputs_1, &inputs_2, &inputs_3, &inputs_4];
        let times = vec![vec![0.0], vec![2.0], vec![4.0], vec![6.0], vec![8.0]];
        let period = 10.0;

        let linear_jitter_propagator = AlphaLinearJitterPropagator::new(connections, &times, period);
        let phi = linear_jitter_propagator.spectral_radius(SEED).unwrap();
        assert_relative_eq!(phi, 0.000786348, max_relative = 1e-2);
    }

    #[test]
    fn test_jitter_spectral_radius_complex() {
        let inputs_0 = vec![
            Input::new(2, 1.0, 0.0),
            Input::new(3, 1.0, 0.0),
            Input::new(4, 1.0, 0.0),
        ];
        let inputs_1 = vec![
            Input::new(0, 1.0, 0.0),
            Input::new(3, 1.0, 0.0),
            Input::new(4, 1.0, 0.0),
        ];
        let inputs_2 = vec![
            Input::new(0, 1.0, 0.0),
            Input::new(1, 1.0, 0.0),
            Input::new(4, 1.0, 0.0),
        ];
        let inputs_3 = vec![
            Input::new(0, 1.0, 0.0),
            Input::new(1, 1.0, 0.0),
            Input::new(2, 1.0, 0.0),
        ];
        let inputs_4 = vec![
            Input::new(1, 1.0, 0.0),
            Input::new(2, 1.0, 0.0),
            Input::new(3, 1.0, 0.0),
        ];
        let connections = vec![&inputs_0, &inputs_1, &inputs_2, &inputs_3, &inputs_4];

        let times = vec![vec![0.0], vec![2.0], vec![4.0], vec![6.0], vec![8.0]];
        let period = 10.0;

        let linear_jitter_propagator = AlphaLinearJitterPropagator::new(connections, &times, period);
        let phi = linear_jitter_propagator.spectral_radius(SEED).unwrap();

        assert_relative_eq!(phi, 0.0009247, max_relative = 1e-2);
    }
}
