use core::f64;

use log::{debug, info};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error::SNNError;

use super::connection::Connection;
use super::spike_train::Spike;

/// The number of power iterations to compute the jitter eigenvalue.
const MAX_ITER_SPECTRAL_RADIUS: usize = 100;
const REL_TOL_RAD: f64 = 1e-6;
const TOL_ZERO: f64 = 1e-12;

/// Represents a spike train associated with a specific neuron.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
struct JitSpike {
    /// The ID of the neuron producing the spike.
    pub source_id: usize,
    /// The time at which the spike is produced.
    pub time: f64,
    /// The jitter of the spike.
    pub jitter: f64,
}

fn jitter_linear_transform(
    jit_spike_train: &mut Vec<JitSpike>,
    num_spikes: usize,
    connections: &Vec<Vec<Connection>>,
    period: f64,
) {
    let mean_jitter = jit_spike_train
        .iter()
        .fold(0.0, |acc, jit_spike| acc + jit_spike.jitter)
        / num_spikes as f64;

    for i in 0..num_spikes {
        let (a, b) = jit_spike_train
            .iter()
            .fold((0.0, 0.0), |(acc_a, acc_b), src_jit_spike| {
                let tmp = connections[jit_spike_train[i].source_id]
                    .iter()
                    .filter(|input| input.source_id() == src_jit_spike.source_id)
                    .fold(0.0, |acc, input| {
                        let dt = (jit_spike_train[i].time - input.delay() - src_jit_spike.time)
                            .rem_euclid(period);
                        acc + input.weight() * (1_f64 - dt) * (1_f64 - dt).exp()
                    });

                (acc_a + tmp, acc_b + src_jit_spike.jitter * tmp)
            });

        jit_spike_train[i].jitter = b / a;
    }
    for i in 0..num_spikes {
        jit_spike_train[i].jitter -= mean_jitter;
    }
}

fn jitter_rescale(jit_spike_train: &mut Vec<JitSpike>, gamma: f64) -> Result<(), SNNError> {
    if gamma == 0.0 {
        return Err(SNNError::InvalidParameters(
            "0 is not a valid rescaling factor".to_string(),
        ));
    }
    for i in 0..jit_spike_train.len() {
        jit_spike_train[i].jitter /= gamma;
    }
    Ok(())
}

fn jitter_random<R: Rng>(spike_train: &Vec<Spike>, rng: &mut R) -> Vec<JitSpike> {
    let mut jit_spike_train: Vec<JitSpike> = spike_train
        .iter()
        .map(|spike| {
            // let jitter = rng.gen::<f64>();
            JitSpike {
                source_id: spike.source_id(),
                time: spike.time(),
                jitter: rng.gen::<f64>(),
            }
        })
        .collect();
    jit_spike_train.sort_by(|jit_spike_1, jit_spike_2| {
        jit_spike_1.time.partial_cmp(&jit_spike_2.time).unwrap()
    });
    jit_spike_train
}

fn jitter_normalize(jit_spike_train: &mut Vec<JitSpike>) -> Result<(), SNNError> {
    let norm = jitter_inner(jit_spike_train, jit_spike_train).sqrt();
    jitter_rescale(jit_spike_train, norm)
    // if norm == 0.0 {
    //     return Err(SNNError::JitterNormalizationError(
    //         "Impossible to normalize the zero vector".to_string(),
    //     ));
    // }

    // for i in 0..jit_spike_train.len() {
    //     jit_spike_train[i].jitter /= norm;
    // }
    // Ok(())
}

fn jitter_inner(jit_spike_train_1: &Vec<JitSpike>, jit_spike_train_2: &Vec<JitSpike>) -> f64 {
    jit_spike_train_1
        .iter()
        .zip(jit_spike_train_2.iter())
        .fold(0.0, |acc, (jit_spike_1, jit_spike_2)| {
            acc + jit_spike_1.jitter * jit_spike_2.jitter
        })
}

// fn jitter_gram_schmidt(
//     jit_spike_train: &mut Vec<JitSpike>,
//     prev_jit_spike_train: &Vec<JitSpike>,
// ) -> Result<(), SNNError> {
//     let c = jitter_inner(jit_spike_train, prev_jit_spike_train);
//     if (c.abs() - 1_f64).abs() < TOL_COLINEAR {
//         return Err(SNNError::JitterGramSchmidtError(
//             "Error during Gram-Schmidt orthogonalization: the dominant eigenspace is one-dimensional ".to_string()
//         ));
//     }

//     for i in 0..jit_spike_train.len() {
//         jit_spike_train[i].jitter -= c * prev_jit_spike_train[i].jitter;
//     }

//     jitter_normalize(jit_spike_train)
// }

fn jitter_copy(jit_spike_train: &Vec<JitSpike>, copied_jit_spike_train: &mut Vec<JitSpike>) {
    for i in 0..jit_spike_train.len() {
        copied_jit_spike_train[i].jitter = jit_spike_train[i].jitter;
    }
}

// fn jitter_power_iteration(
//     jit_spike_train: &mut Vec<JitSpike>,
//     num_spikes: usize,
//     connections: &Vec<Vec<Connection>>,
//     period: f64,
// ) -> Result<(), SNNError> {
//     jitter_linear_transform(jit_spike_train, num_spikes, connections, period);
//     jitter_normalize(jit_spike_train)
// }

pub fn jitter_spectral_radius<R: Rng>(
    connections: &Vec<Vec<Connection>>,
    spike_train: &Vec<Spike>,
    period: f64,
    rng: &mut R,
) -> Result<f64, SNNError> {
    // Init the jitter vectors at random (should have non-zero components in the dominant eigenspace) and two consecutive iterations
    let mut jit_spike_train: Vec<JitSpike> = jitter_random(spike_train, rng);
    jitter_normalize(&mut jit_spike_train)?;
    let mut next_jit_spike_train: Vec<JitSpike> = jit_spike_train.clone();
    jitter_linear_transform(&mut next_jit_spike_train, spike_train.len(), connections, period);
    let mut next_next_jit_spike_train: Vec<JitSpike> = next_jit_spike_train.clone();
    jitter_linear_transform(&mut next_next_jit_spike_train, spike_train.len(), connections, period);

    let mut spectral_radius: f64;
    let mut prev_spectral_radius = f64::NAN;
    
    // Repeat until convergence or maximum number of iterations is reached
    for it in 0..MAX_ITER_SPECTRAL_RADIUS {
        // // Do a few power iterations to obtain an approximation of the dominant eigenspace
        // for _ in 0..10 {
        //     jitter_copy(&prev_jit_spike_train, &mut jit_spike_train);
        //     jitter_power_iteration(&mut jit_spike_train, spike_train.len(), connections, period)?;
        //     jitter_copy(&jit_spike_train, &mut prev_jit_spike_train);
        // }
        
        let jit_01 = jitter_inner(&jit_spike_train, &next_jit_spike_train);
        let jit_12 = jitter_inner(&next_jit_spike_train, &next_next_jit_spike_train);
        let jit_02 = jitter_inner(&jit_spike_train, &next_next_jit_spike_train);
        // let jit_00 = jitter_inner(&jit_spike_train, &jit_spike_train);
        let jit_11 = jitter_inner(&next_jit_spike_train, &next_jit_spike_train);
        // let jit_22 = jitter_inner(&next_next_jit_spike_train, &next_next_jit_spike_train);
        
        let gamma = (jit_11 - jit_01 * jit_01).abs();
        if gamma < TOL_ZERO {
            spectral_radius = jit_01.abs();
            debug!("Iter {}: one-dimensional eigenspace detected ({} < {}) with spectral radius {} (previous was {})", it, gamma, TOL_ZERO, spectral_radius, prev_spectral_radius);
        }
        else {
            spectral_radius = ((jit_01 * jit_12 - jit_02 * jit_11).abs() / gamma).sqrt();
            debug!("Iter {}: two-dimensional eigenspace detected ({} > {}) with spectral radius {} (previous was {})", it, gamma, TOL_ZERO, spectral_radius, prev_spectral_radius);
        }

        let rel_diff = (spectral_radius - prev_spectral_radius).abs() / spectral_radius.max(prev_spectral_radius);
        if rel_diff < REL_TOL_RAD {
            info!("Converged to spectral radius in {} iterations", it);
            return Ok(spectral_radius);
        }
        prev_spectral_radius = spectral_radius;

        jitter_copy(&next_jit_spike_train, &mut jit_spike_train);
        jitter_rescale(&mut jit_spike_train, jit_11.sqrt())?;
        jitter_copy(&next_next_jit_spike_train, &mut next_jit_spike_train);
        jitter_rescale(&mut next_jit_spike_train, jit_11.sqrt())?;
        jitter_copy(&next_jit_spike_train, &mut next_next_jit_spike_train);
        jitter_linear_transform(&mut next_next_jit_spike_train, spike_train.len(), connections, period);
    }

    Err(SNNError::ConvergenceError(
        "Maximum number of iterations reached before the spectral radius has converged".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_jitter_spectral_radius_real() {
        let period = 10.0;
        let mut rng = rand::thread_rng();

        let connections = vec![
            vec![
                Connection::build(1, 0, 1.0, 0.0).unwrap(),
                Connection::build(2, 0, 1.0, 0.0).unwrap(),
                Connection::build(3, 0, 1.0, 0.0).unwrap(),
                Connection::build(4, 0, 1.0, 0.0).unwrap(),
            ],
            vec![
                Connection::build(0, 1, 1.0, 0.0).unwrap(),
                Connection::build(2, 1, 1.0, 0.0).unwrap(),
                Connection::build(3, 1, 1.0, 0.0).unwrap(),
                Connection::build(4, 1, 1.0, 0.0).unwrap(),
            ],
            vec![
                Connection::build(0, 2, 1.0, 0.0).unwrap(),
                Connection::build(1, 2, 1.0, 0.0).unwrap(),
                Connection::build(3, 2, 1.0, 0.0).unwrap(),
                Connection::build(4, 2, 1.0, 0.0).unwrap(),
            ],
            vec![
                Connection::build(0, 3, 1.0, 0.0).unwrap(),
                Connection::build(1, 3, 1.0, 0.0).unwrap(),
                Connection::build(2, 3, 1.0, 0.0).unwrap(),
                Connection::build(4, 3, 1.0, 0.0).unwrap(),
            ],
            vec![
                Connection::build(0, 4, 1.0, 0.0).unwrap(),
                Connection::build(1, 4, 1.0, 0.0).unwrap(),
                Connection::build(2, 4, 1.0, 0.0).unwrap(),
                Connection::build(3, 4, 1.0, 0.0).unwrap(),
            ],
        ];

        let spike_train = vec![
            Spike::new(0, 0.0),
            Spike::new(1, 2.0),
            Spike::new(2, 4.0),
            Spike::new(3, 6.0),
            Spike::new(4, 8.0),
        ];
        let phi = jitter_spectral_radius(&connections, &spike_train, period, &mut rng).unwrap();
        assert_relative_eq!(phi, 0.000786348, max_relative = 1e-2);
    }

    #[test]
    fn test_jitter_spectral_radius_complex() {
        let period = 10.0;
        let mut rng = rand::thread_rng();

        let connections = vec![
            vec![
                Connection::build(2, 0, 1.0, 0.0).unwrap(),
                Connection::build(3, 0, 1.0, 0.0).unwrap(),
                Connection::build(4, 0, 1.0, 0.0).unwrap(),
            ],
            vec![
                Connection::build(0, 1, 1.0, 0.0).unwrap(),
                Connection::build(3, 1, 1.0, 0.0).unwrap(),
                Connection::build(4, 1, 1.0, 0.0).unwrap(),
            ],
            vec![
                Connection::build(0, 2, 1.0, 0.0).unwrap(),
                Connection::build(1, 2, 1.0, 0.0).unwrap(),
                Connection::build(4, 2, 1.0, 0.0).unwrap(),
            ],
            vec![
                Connection::build(0, 3, 1.0, 0.0).unwrap(),
                Connection::build(1, 3, 1.0, 0.0).unwrap(),
                Connection::build(2, 3, 1.0, 0.0).unwrap(),
            ],
            vec![
                Connection::build(1, 4, 1.0, 0.0).unwrap(),
                Connection::build(2, 4, 1.0, 0.0).unwrap(),
                Connection::build(3, 4, 1.0, 0.0).unwrap(),
            ],
        ];

        let spike_train = vec![
            Spike::new(0, 0.0),
            Spike::new(1, 2.0),
            Spike::new(2, 4.0),
            Spike::new(3, 6.0),
            Spike::new(4, 8.0),
        ];
        let phi = jitter_spectral_radius(&connections, &spike_train, period, &mut rng).unwrap();
        assert_relative_eq!(phi, 0.0009247, max_relative = 1e-2);
    }
}
