use rand::seq::SliceRandom;
use rand::Rng;

pub trait IOSampler {
    fn iter(&mut self) -> Box<dyn Iterator<Item = (usize, usize)>>;
}

pub struct IOBalancedSampler {
    pub num_neurons: usize,
    pub num_connections: usize,
}

impl IOSampler for IOBalancedSampler {
    fn iter(&mut self) -> Box<dyn Iterator<Item = (usize, usize)>> {
        let mut rng = rand::thread_rng();
        let mut source_ids : Vec<usize> = (0..self.num_neurons).flat_map(|id| std::iter::repeat(id).take(self.num_connections / self.num_neurons)).collect();
        source_ids.shuffle(&mut rng);
        let target_ids : Vec<usize> = (0..self.num_neurons).flat_map(|id| std::iter::repeat(id).take(self.num_connections / self.num_neurons)).collect();
        Box::new(source_ids.into_iter().zip(target_ids.into_iter()))
    }
}

pub struct InBalancedSampler {
    pub num_neurons: usize,
    pub num_connections: usize,
}

impl IOSampler for InBalancedSampler {
    fn iter(&mut self) -> Box<dyn Iterator<Item = (usize, usize)>> {
        let mut rng = rand::thread_rng();
        let source_ids : Vec<usize> = (0..self.num_connections).map(|_| rng.gen_range(0..self.num_neurons)).collect();
        let target_ids : Vec<usize> = (0..self.num_neurons).flat_map(|id| std::iter::repeat(id).take(self.num_connections / self.num_neurons)).collect();
        Box::new(source_ids.into_iter().zip(target_ids.into_iter()))
    }
}

pub struct OutBalancedSampler {
    pub num_neurons: usize,
    pub num_connections: usize,
}

impl IOSampler for OutBalancedSampler {
    fn iter(&mut self) -> Box<dyn Iterator<Item = (usize, usize)>> {
        let mut rng = rand::thread_rng();
        let mut source_ids : Vec<usize> = (0..self.num_neurons).flat_map(|id| std::iter::repeat(id).take(self.num_connections / self.num_neurons)).collect();
        source_ids.shuffle(&mut rng);
        let mut target_ids : Vec<usize> = (0..self.num_connections).map(|_| rng.gen_range(0..self.num_neurons)).collect();
        target_ids.sort();
        Box::new(source_ids.into_iter().zip(target_ids.into_iter()))
    }
}

pub struct UnbalancedSampler {
    pub num_neurons: usize,
    pub num_connections: usize,
}

impl IOSampler for UnbalancedSampler {
    fn iter(&mut self) -> Box<dyn Iterator<Item = (usize, usize)>> {
        let mut rng = rand::thread_rng();
        let source_ids : Vec<usize> = (0..self.num_connections).map(|_| rng.gen_range(0..self.num_neurons)).collect();
        let mut target_ids : Vec<usize> = (0..self.num_connections).map(|_| rng.gen_range(0..self.num_neurons)).collect::<Vec<usize>>();
        target_ids.sort();
        Box::new(source_ids.into_iter().zip(target_ids.into_iter()))
    }
}

pub enum IOSamplerEnum {
    IOBalanced(IOBalancedSampler),
    InBalanced(InBalancedSampler),
    OutBalanced(OutBalancedSampler),
    Unbalanced(UnbalancedSampler),
}

impl IOSampler for IOSamplerEnum {
    fn iter(&mut self) -> Box<dyn Iterator<Item = (usize, usize)>> {
        match self {
            IOSamplerEnum::IOBalanced(sampler) => sampler.iter(),
            IOSamplerEnum::InBalanced(sampler) => sampler.iter(),
            IOSamplerEnum::OutBalanced(sampler) => sampler.iter(),
            IOSamplerEnum::Unbalanced(sampler) => sampler.iter(),
        }
    }
}

// test if their are sorted by target_id
// test if the number of connections is correct
// test if the number of source_ids is correct in OutBalancedSampler
// test if the number of target_ids is correct in InBalancedSampler
// test if the number of source_ids is correct in IOBalancedSampler
// test if the number of target_ids is correct in IOBalancedSampler

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_balanced_sampler() {
        let mut sampler = IOBalancedSampler {
            num_neurons: 10,
            num_connections: 100,
        };

        for (source_id, target_id) in sampler.iter() {
            assert!(source_id < 10);
            assert!(target_id < 10);
        }
        for id in 0..10 {
            assert_eq!(sampler.iter().filter(|(src, _)| *src == id).count(), 10);
            assert_eq!(sampler.iter().filter(|(_, tgt)| *tgt == id).count(), 10);
        }
        assert!(sampler.iter().collect::<Vec<(usize, usize)>>().len() == 100);
    }

    #[test]
    fn test_in_balanced_sampler() {
        let mut sampler = InBalancedSampler {
            num_neurons: 10,
            num_connections: 100,
        };

        for (source_id, target_id) in sampler.iter() {
            assert!(source_id < 10);
            assert!(target_id < 10);
        }
        for id in 0..10 {
            assert_eq!(sampler.iter().filter(|(_, tgt)| *tgt == id).count(), 10);
        }
        assert!(sampler.iter().collect::<Vec<(usize, usize)>>().len() == 100);
    }
    #[test]
    fn test_out_balanced_sampler() {
        let mut sampler = OutBalancedSampler {
            num_neurons: 10,
            num_connections: 100,
        };

        for (source_id, target_id) in sampler.iter() {
            assert!(source_id < 10);
            assert!(target_id < 10);
        }
        for id in 0..10 {
            assert_eq!(sampler.iter().filter(|(src, _)| *src == id).count(), 10);
        }
        assert!(sampler.iter().collect::<Vec<(usize, usize)>>().len() == 100);
    }
    #[test]
    fn test_unbalanced_sampler() {
        let mut sampler = UnbalancedSampler {
            num_neurons: 10,
            num_connections: 100,
        };

        for (source_id, target_id) in sampler.iter() {
            assert!(source_id < 10);
            assert!(target_id < 10);
        }
        assert!(sampler.iter().collect::<Vec<(usize, usize)>>().len() == 100);
    }
}