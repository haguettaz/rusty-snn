//! Builder modulex with utilities for generating random networks.

use super::network::Network;
use super::neuron::Neuron;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::distributions::{Uniform, Distribution};

#[derive(Debug, PartialEq)]
pub enum NetworkSamplerError {
    /// Error for invalid topology, e.g., incompatible with the number of connections and neurons.
    TopologyError(String),
    /// Error for invalid weights, e.g., minimum weight greater than maximum weight.
    DelaysError(String),
    /// Error for invalid delays, e.g., minimum delay greater than maximum delay.
    WeightsError(String),
}

#[derive(Debug, PartialEq)]
pub enum Topology {
    /// Random topology
    Random,
    /// Fixed in-degree topology, i.e., each neuron has the same number of inputs.
    Fin,
    /// Fixed out-degree topology, i.e., each neuron has the same number of outputs.
    Fout,
    /// Fixed in-degree and out-degree topology, i.e., each neuron has the same number of inputs and outputs.
    FinFout,
}

#[derive(Debug, PartialEq)]
pub struct NetworkSampler {
    /// the number of neurons in the network
    num_neurons: usize,
    /// the number of connections in the network
    num_connections: usize,
    /// the minimum and maximum weights of the connections
    lim_weights: (f64, f64),
    /// the minimum and maximum delays of the connections
    lim_delays: (f64, f64),
    /// the distribution of network structures
    topology: Topology,
}

impl NetworkSampler {
    /// Create a new NetworkSampler instance.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use rusty_snn::network::builder::{NetworkSampler, Topology};
    /// let sampler = NetworkSampler::new(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::Random);
    /// ```
    pub fn new(
        num_neurons: usize,
        num_connections: usize,
        lim_weights: (f64, f64),
        lim_delays: (f64, f64),
        topology: Topology,
    ) -> Result<Self, NetworkSamplerError> {
        if !matches!(topology, Topology::Random) && num_connections % num_neurons != 0 {
            return Err(NetworkSamplerError::TopologyError("The provided topology is not compatible with the number of connections and neurons.".into()));
        }

        if lim_weights.0 > lim_weights.1 {
            return Err(NetworkSamplerError::WeightsError(
                "The minimum weight must be less than the maximum weight.".into(),
            ));
        }

        if lim_delays.0 > lim_delays.1 {
            return Err(NetworkSamplerError::DelaysError(
                "The minimum delay must be less than the maximum weight.".into(),
            ));
        }

        if lim_delays.0 < 0.0 {
            return Err(NetworkSamplerError::DelaysError(
                "Delays must be non-negative.".into(),
            ));
        }

        Ok(NetworkSampler {
            num_neurons,
            num_connections,
            lim_weights,
            lim_delays,
            topology,
        })
    }

    /// Sample a network from the distribution.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use rusty_snn::network::builder::{NetworkSampler, Topology};
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    /// 
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let sampler = NetworkSampler::new(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::Random).unwrap();
    /// let network = sampler.sample(&mut rng);
    /// ```
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Network {
        let mut neurons = Vec::new();
        for id in 0..self.num_neurons {
            neurons.push(Neuron::new(id, 1.0));
        }

        let mut network = Network::new(neurons);

        let (src_vec, tgt_vec) = match self.topology {
            Topology::Random => {
                let src_vec: Vec<usize> = (0..self.num_connections)
                    .map(|_| rng.gen_range(0..self.num_neurons))
                    .collect();
                let tgt_vec: Vec<usize> = (0..self.num_connections)
                    .map(|_| rng.gen_range(0..self.num_neurons))
                    .collect();
                (src_vec, tgt_vec)
            }
            Topology::Fin => {
                let src_vec: Vec<usize> = (0..self.num_connections)
                    .map(|_| rng.gen_range(0..self.num_neurons))
                    .collect();
                let tgt_vec: Vec<usize> = (0..self.num_connections)
                    .map(|i| i % self.num_neurons)
                    .collect();
                (src_vec, tgt_vec)
            }
            Topology::Fout => {
                let src_vec: Vec<usize> = (0..self.num_connections)
                    .map(|i| i % self.num_neurons)
                    .collect();
                let tgt_vec: Vec<usize> = (0..self.num_connections)
                    .map(|_| rng.gen_range(0..self.num_neurons))
                    .collect();
                (src_vec, tgt_vec)
            }
            Topology::FinFout => {
                let mut src_vec: Vec<usize> = (0..self.num_connections)
                    .map(|i| i % self.num_neurons)
                    .collect();
                src_vec.shuffle(rng);
                let tgt_vec: Vec<usize> = (0..self.num_connections)
                    .map(|i| i % self.num_neurons)
                    .collect();
                (src_vec, tgt_vec)
            }
        };

        let weight_dist = Uniform::new_inclusive(self.lim_weights.0, self.lim_weights.1);
        let delay_dist = Uniform::new_inclusive(self.lim_delays.0, self.lim_delays.1);

        for (src_id, tgt_id) in src_vec.into_iter().zip(tgt_vec.into_iter()) {
            let weight = weight_dist.sample(rng);
            let delay = delay_dist.sample(rng);
            network.add_connection(src_id, tgt_id, weight, delay);
        }

        network
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn validate_num_outputs(network: &Network, num_outputs: usize) {
        for neuron in network.neurons().iter() {
            assert_eq!(
                network
                    .neurons()
                    .iter()
                    .flat_map(|n| n.inputs())
                    .filter(|i| i.source_id() == neuron.id())
                    .count(),
                num_outputs
            );
        }
    }

    fn validate_num_inputs(network: &Network, num_inputs: usize) {
        for neuron in network.neurons().iter() {
            assert_eq!(neuron.inputs().len(), num_inputs);
        }
    }

    #[test]
    fn test_sampler_random() {
        assert_eq!(
            NetworkSampler::new(3, 12, (0.1, -0.1), (0.1, 10.0), Topology::Random),
            Err(NetworkSamplerError::WeightsError(
                "The minimum weight must be less than the maximum weight.".into()
            ))
        );

        assert_eq!(
            NetworkSampler::new(3, 12, (-0.1, 0.1), (10.0, 0.1), Topology::Random),
            Err(NetworkSamplerError::DelaysError(
                "The minimum delay must be less than the maximum weight.".into()
            ))
        );

        let mut rng = StdRng::seed_from_u64(42);
        let network_sampler =
            NetworkSampler::new(3, 8, (-0.1, 0.1), (0.1, 10.0), Topology::Random).unwrap();
        let network = network_sampler.sample(&mut rng);

        assert_eq!(network.num_neurons(), 3);
        assert_eq!(network.num_connections(), 8);
    }

    #[test]
    fn test_sampler_fin() {
        assert_eq!(
            NetworkSampler::new(3, 12, (0.1, -0.1), (0.1, 10.0), Topology::Fin),
            Err(NetworkSamplerError::WeightsError(
                "The minimum weight must be less than the maximum weight.".into()
            ))
        );

        assert_eq!(
            NetworkSampler::new(3, 12, (-0.1, 0.1), (10.0, 0.1), Topology::Fin),
            Err(NetworkSamplerError::DelaysError(
                "The minimum delay must be less than the maximum weight.".into()
            ))
        );

        assert_eq!(
            NetworkSampler::new(3, 8, (-0.1, 0.1), (10.0, 0.1), Topology::Fin),
            Err(NetworkSamplerError::TopologyError(
                "The provided topology is not compatible with the number of connections and neurons."
                    .into()
            ))
        );

        let mut rng = StdRng::seed_from_u64(42);
        let network_sampler =
            NetworkSampler::new(3, 12, (-0.1, 0.1), (0.1, 10.0), Topology::Fin).unwrap();
        let network = network_sampler.sample(&mut rng);

        assert_eq!(network.num_neurons(), 3);
        assert_eq!(network.num_connections(), 12);

        validate_num_inputs(&network, 4);
    }

    #[test]
    fn test_sampler_fout() {
        assert_eq!(
            NetworkSampler::new(3, 12, (0.1, -0.1), (0.1, 10.0), Topology::Fout),
            Err(NetworkSamplerError::WeightsError(
                "The minimum weight must be less than the maximum weight.".into()
            ))
        );

        assert_eq!(
            NetworkSampler::new(3, 12, (-0.1, 0.1), (10.0, 0.1), Topology::Fout),
            Err(NetworkSamplerError::DelaysError(
                "The minimum delay must be less than the maximum weight.".into()
            ))
        );

        assert_eq!(
            NetworkSampler::new(3, 8, (-0.1, 0.1), (10.0, 0.1), Topology::Fout),
            Err(NetworkSamplerError::TopologyError(
                "The provided topology is not compatible with the number of connections and neurons."
                    .into()
            ))
        );

        let mut rng = StdRng::seed_from_u64(42);
        let network_sampler =
            NetworkSampler::new(3, 12, (-0.1, 0.1), (0.1, 10.0), Topology::Fout).unwrap();
        let network = network_sampler.sample(&mut rng);

        assert_eq!(network.num_neurons(), 3);
        assert_eq!(network.num_connections(), 12);

        validate_num_outputs(&network, 4);
    }

    #[test]
    fn test_sampler_fin_fout() {
        assert_eq!(
            NetworkSampler::new(3, 12, (0.1, -0.1), (0.1, 10.0), Topology::FinFout),
            Err(NetworkSamplerError::WeightsError(
                "The minimum weight must be less than the maximum weight.".into()
            ))
        );

        assert_eq!(
            NetworkSampler::new(3, 12, (-0.1, 0.1), (10.0, 0.1), Topology::FinFout),
            Err(NetworkSamplerError::DelaysError(
                "The minimum delay must be less than the maximum weight.".into()
            ))
        );

        assert_eq!(
            NetworkSampler::new(3, 8, (-0.1, 0.1), (10.0, 0.1), Topology::FinFout),
            Err(NetworkSamplerError::TopologyError(
                "The provided topology is not compatible with the number of connections and neurons."
                    .into()
            ))
        );

        let mut rng = StdRng::seed_from_u64(42);
        let network_sampler =
            NetworkSampler::new(3, 12, (-0.1, 0.1), (0.1, 10.0), Topology::FinFout).unwrap();
        let network = network_sampler.sample(&mut rng);

        assert_eq!(network.num_neurons(), 3);
        assert_eq!(network.num_connections(), 12);

        validate_num_inputs(&network, 4);
        validate_num_outputs(&network, 4);
    }
}