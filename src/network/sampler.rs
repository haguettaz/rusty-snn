//! Builder module with utilities for generating random networks.

use super::network::Network;
use super::neuron::Neuron;
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Debug, PartialEq)]
pub enum NetworkSamplerError {
    /// Error for invalid topology, e.g., incompatible with the number of connections and neurons.
    TopologyError,
    /// Error for invalid delays, e.g., minimum delay greater than maximum delay.
    DelaysError,
    /// Error for invalid weights, e.g., minimum weight greater than maximum weight.
    WeightsError,
}

impl std::fmt::Display for NetworkSamplerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NetworkSamplerError::TopologyError => write!(
                f,
                "The provided topology is not compatible with the number of connections and neurons.",
            ),
            NetworkSamplerError::DelaysError => write!(
                f,
                "The provided interval of delays is invalid.",
            ),
            NetworkSamplerError::WeightsError => write!(
                f,
                "The provided interval of weights is invalid.",
            ),
        }
    }
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
    /// # Arguments
    ///
    /// * `num_neurons` - The number of neurons in the network
    /// * `num_connections` - The number of connections in the network
    /// * `lim_weights` - The minimum and maximum weights of the connections
    /// * `lim_delays` - The minimum and maximum delays of the connections
    /// * `topology` - The distribution of network structures
    pub fn new(
        num_neurons: usize,
        num_connections: usize,
        lim_weights: (f64, f64),
        lim_delays: (f64, f64),
        topology: Topology,
    ) -> Result<Self, NetworkSamplerError> {
        if !matches!(topology, Topology::Random) && num_connections % num_neurons != 0 {
            return Err(NetworkSamplerError::TopologyError);
        }

        if lim_weights.0 > lim_weights.1 {
            return Err(NetworkSamplerError::WeightsError);
        }

        if lim_delays.0 > lim_delays.1 {
            return Err(NetworkSamplerError::DelaysError);
        }

        if lim_delays.0 < 0.0 {
            return Err(NetworkSamplerError::DelaysError);
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
    /// ```rust
    /// use rusty_snn::network::sampler::{NetworkSampler, Topology};
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

        let weight_dist = Uniform::new_inclusive(self.lim_weights.0, self.lim_weights.1);
        let delay_dist = Uniform::new_inclusive(self.lim_delays.0, self.lim_delays.1);

        let (src_vec, tgt_vec) = self.sample_src_tgt(rng);

        for (src_id, tgt_id) in src_vec.into_iter().zip(tgt_vec.into_iter()) {
            let weight = weight_dist.sample(rng);
            let delay = delay_dist.sample(rng);
            if let Err(e) = network.add_connection(src_id, tgt_id, weight, delay) {
                eprintln!("Failed to add connection: {:?}", e);
            }
        }

        network
    }

    pub fn sample_src_tgt<R: Rng>(&self, rng: &mut R) -> (Vec<usize>, Vec<usize>) {
        match self.topology {
            Topology::Random => self.sample_src_tgt_random(rng),
            Topology::Fin => self.sample_src_tgt_fin(rng),
            Topology::Fout => self.sample_src_tgt_fout(rng),
            Topology::FinFout => self.sample_src_tgt_fin_fout(rng),
        }
    }

    pub fn sample_src_tgt_random<R: Rng>(&self, rng: &mut R) -> (Vec<usize>, Vec<usize>) {
        let mut source_ids = Vec::with_capacity(self.num_connections);
        let mut target_ids = Vec::with_capacity(self.num_connections);

        let dist = Uniform::new(0, self.num_neurons);

        for _ in 0..self.num_connections {
            source_ids.push(dist.sample(rng));
            target_ids.push(dist.sample(rng));
        }

        (source_ids, target_ids)
    }

    pub fn sample_src_tgt_fin<R: Rng>(&self, rng: &mut R) -> (Vec<usize>, Vec<usize>) {
        let mut source_ids = Vec::with_capacity(self.num_connections);
        let mut target_ids = Vec::with_capacity(self.num_connections);

        let dist = Uniform::new(0, self.num_neurons);

        for i in 0..self.num_connections {
            source_ids.push(dist.sample(rng));
            target_ids.push(i % self.num_neurons);
        }

        (source_ids, target_ids)
    }

    pub fn sample_src_tgt_fout<R: Rng>(&self, rng: &mut R) -> (Vec<usize>, Vec<usize>) {
        let mut source_ids = Vec::with_capacity(self.num_connections);
        let mut target_ids = Vec::with_capacity(self.num_connections);

        let dist = Uniform::new(0, self.num_neurons);

        for i in 0..self.num_connections {
            source_ids.push(i % self.num_neurons);
            target_ids.push(dist.sample(rng));
        }

        (source_ids, target_ids)
    }

    pub fn sample_src_tgt_fin_fout<R: Rng>(&self, rng: &mut R) -> (Vec<usize>, Vec<usize>) {
        let mut source_ids = Vec::with_capacity(self.num_connections);
        let mut target_ids = Vec::with_capacity(self.num_connections);

        for i in 0..self.num_connections {
            source_ids.push(i % self.num_neurons);
            target_ids.push(i % self.num_neurons);
        }
        target_ids.shuffle(rng);

        (source_ids, target_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_sampler_new() {
        assert_eq!(
            NetworkSampler::new(3, 8, (-0.25, 0.25), (1.0, 8.0), Topology::Fin),
            Err(NetworkSamplerError::TopologyError)
        );

        assert_eq!(
            NetworkSampler::new(3, 8, (-0.25, 0.25), (1.0, 8.0), Topology::Fout),
            Err(NetworkSamplerError::TopologyError)
        );

        assert_eq!(
            NetworkSampler::new(3, 8, (-0.25, 0.25), (1.0, 8.0), Topology::FinFout),
            Err(NetworkSamplerError::TopologyError)
        );

        let mut rng = StdRng::seed_from_u64(42);
        let network_sampler =
            NetworkSampler::new(3, 8, (-0.1, 0.1), (0.1, 10.0), Topology::Random).unwrap();
        let network = network_sampler.sample(&mut rng);

        assert_eq!(network.num_neurons(), 3);
        assert_eq!(network.num_connections(), 8);
    }

    #[test]
    fn test_src_tgt_random() {
        let mut rng = StdRng::seed_from_u64(42);
        let network_sampler =
            NetworkSampler::new(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::Random).unwrap();
        let (src_vec, tgt_vec) = network_sampler.sample_src_tgt(&mut rng);

        assert_eq!(src_vec.len(), 100);
        assert_eq!(tgt_vec.len(), 100);
    }

    #[test]
    fn test_src_tgt_fin() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut vec = (0..10).cycle().take(100).collect::<Vec<usize>>();
        vec.sort();

        let network_sampler =
            NetworkSampler::new(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::Fin).unwrap();

        let (src_vec, mut tgt_vec) = network_sampler.sample_src_tgt(&mut rng);
        assert_eq!(src_vec.len(), 100);
        tgt_vec.sort();
        assert_eq!(tgt_vec, vec);
    }

    #[test]
    fn test_src_tgt_fout() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut vec = (0..10).cycle().take(100).collect::<Vec<usize>>();
        vec.sort();

        let network_sampler =
            NetworkSampler::new(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::Fout).unwrap();

        let (mut src_vec, tgt_vec) = network_sampler.sample_src_tgt(&mut rng);
        src_vec.sort();
        assert_eq!(src_vec, vec);
        assert_eq!(tgt_vec.len(), 100);
    }

    #[test]
    fn test_src_tgt_fint_fout() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut vec = (0..10).cycle().take(100).collect::<Vec<usize>>();
        vec.sort();

        let network_sampler =
            NetworkSampler::new(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::FinFout).unwrap();

        let (mut src_vec, mut tgt_vec) = network_sampler.sample_src_tgt(&mut rng);
        src_vec.sort();
        assert_eq!(src_vec, vec);
        tgt_vec.sort();
        assert_eq!(tgt_vec, vec);
    }
}
