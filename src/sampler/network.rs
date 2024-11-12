//! This module provides a network sampler that generates random networks with a given topology.
//!
//! # Examples
//!
//! ```rust
//! use rusty_snn::sampler::network::{NetworkSampler, Topology};
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! // Set the random number generator seed
//! let mut rng = StdRng::seed_from_u64(42);
//!
//! // Create a network sampler to generate networks with 10 neurons, 100 connections, weights in the range (-0.1, 0.1), delays in the range (0.1, 10.0), and fixed-in-degree topology.
//! let sampler = NetworkSampler::build(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::Fin).unwrap();
//!
//! // Sample a network from the distribution
//! let network = sampler.sample(&mut rng);
//!
//! assert_eq!(network.num_neurons(), 10);
//! assert_eq!(network.num_connections(), 100);
//! assert_eq!(network.num_inputs(0), 10);
//! ```

use crate::core::connection::Connection;
use crate::core::network::Network;

use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;
use std::error::Error;
use std::fmt;

#[derive(Debug, PartialEq)]
pub enum Topology {
    /// Random topology
    Random,
    /// Fixed in-degree topology, i.e., each neuron has the same number of inputs.
    /// Requires `num_connections & num_neurons == 0`.
    Fin,
    /// Fixed out-degree topology, i.e., each neuron has the same number of outputs.
    /// Requires `num_connections & num_neurons == 0`.
    Fout,
    /// Fixed in-degree and out-degree topology, i.e., each neuron has the same number of inputs and outputs.
    /// Requires `num_connections & num_neurons == 0`.
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
    /// Create a network sampler with the specified parameters.
    /// The function returns an error for invalid delay values or incompatible topologies.
    pub fn build(
        num_neurons: usize,
        num_connections: usize,
        lim_weights: (f64, f64),
        lim_delays: (f64, f64),
        topology: Topology,
    ) -> Result<Self, NetworkSamplerError> {
        if !matches!(topology, Topology::Random) && num_connections % num_neurons != 0 {
            return Err(NetworkSamplerError::IncompatibleTopology);
        }

        if lim_delays.0 <= 0.0 {
            return Err(NetworkSamplerError::InvalidDelay);
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
    /// use rusty_snn::sampler::network::{NetworkSampler, Topology};
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let sampler = NetworkSampler::build(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::Random).unwrap();
    /// let network = sampler.sample(&mut rng);
    /// ```
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Network {
        // let neurons = (0..self.num_neurons)
        //     .map(|id| Neuron::new(id, 1.0))
        //     .collect();

        let weight_dist = Uniform::new_inclusive(self.lim_weights.0, self.lim_weights.1);
        let delay_dist = Uniform::new_inclusive(self.lim_delays.0, self.lim_delays.1);

        let mut connections = Vec::with_capacity(self.num_connections);
        let (source_vec, target_vec) = self.sample_src_tgt(rng);
        for (source_id, target_id) in source_vec.into_iter().zip(target_vec.into_iter()) {
            let weight = weight_dist.sample(rng);
            let delay = delay_dist.sample(rng);
            connections.push(Connection::build(source_id, target_id, weight, delay).unwrap());
        }

        Network::new(connections)
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

#[derive(Debug, PartialEq)]
pub enum NetworkSamplerError {
    /// Error for invalid delay value.
    InvalidDelay,
    /// Error for incompatibility between the topology and the number of connections and neurons.
    IncompatibleTopology,
}

impl fmt::Display for NetworkSamplerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NetworkSamplerError::InvalidDelay => write!(f, "Invalid delay value: must be non-negative"),
            NetworkSamplerError::IncompatibleTopology => write!(f, "The connectivity topology is not compatible with the number of connections and neurons"),
        }
    }
}

impl Error for NetworkSamplerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_sampler_build() {
        assert_eq!(
            NetworkSampler::build(3, 8, (-0.25, 0.25), (0.0, 8.0), Topology::Random),
            Err(NetworkSamplerError::InvalidDelay)
        );

        assert_eq!(
            NetworkSampler::build(3, 8, (-0.25, 0.25), (1.0, 8.0), Topology::Fin),
            Err(NetworkSamplerError::IncompatibleTopology)
        );

        assert_eq!(
            NetworkSampler::build(3, 8, (-0.25, 0.25), (1.0, 8.0), Topology::Fout),
            Err(NetworkSamplerError::IncompatibleTopology)
        );

        assert_eq!(
            NetworkSampler::build(3, 8, (-0.25, 0.25), (1.0, 8.0), Topology::FinFout),
            Err(NetworkSamplerError::IncompatibleTopology)
        );

        let mut rng = StdRng::seed_from_u64(42);
        let network_sampler =
            NetworkSampler::build(3, 8, (-0.1, 0.1), (0.1, 10.0), Topology::Random).unwrap();
        let network = network_sampler.sample(&mut rng);

        assert_eq!(network.num_neurons(), 3);
        assert_eq!(network.num_connections(), 8);
    }

    #[test]
    fn test_src_tgt_random() {
        let mut rng = StdRng::seed_from_u64(42);
        let network_sampler =
            NetworkSampler::build(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::Random).unwrap();
        let (source_vec, target_vec) = network_sampler.sample_src_tgt(&mut rng);

        assert_eq!(source_vec.len(), 100);
        assert_eq!(target_vec.len(), 100);
    }

    #[test]
    fn test_src_tgt_fin() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut vec = (0..10).cycle().take(100).collect::<Vec<usize>>();
        vec.sort();

        let network_sampler =
            NetworkSampler::build(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::Fin).unwrap();

        let (source_vec, mut target_vec) = network_sampler.sample_src_tgt(&mut rng);
        assert_eq!(source_vec.len(), 100);
        target_vec.sort();
        assert_eq!(target_vec, vec);
    }

    #[test]
    fn test_src_tgt_fout() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut vec = (0..10).cycle().take(100).collect::<Vec<usize>>();
        vec.sort();

        let network_sampler =
            NetworkSampler::build(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::Fout).unwrap();

        let (mut source_vec, target_vec) = network_sampler.sample_src_tgt(&mut rng);
        source_vec.sort();
        assert_eq!(source_vec, vec);
        assert_eq!(target_vec.len(), 100);
    }

    #[test]
    fn test_src_tgt_fint_fout() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut vec = (0..10).cycle().take(100).collect::<Vec<usize>>();
        vec.sort();

        let network_sampler =
            NetworkSampler::build(10, 100, (-0.1, 0.1), (0.1, 10.0), Topology::FinFout).unwrap();

        let (mut source_vec, mut target_vec) = network_sampler.sample_src_tgt(&mut rng);
        source_vec.sort();
        assert_eq!(source_vec, vec);
        target_vec.sort();
        assert_eq!(target_vec, vec);
    }
}
