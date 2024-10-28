// use rand::distributions::{WeightedError, WeightedIndex};

// pub fn prob_num_spikes(period: f64, firing_rate: f64) -> Result<WeightedIndex<f64>, WeightedError> {
//     let max_num_spikes = if period == period.floor() {
//         period as usize - 1
//     } else {
//         period as usize
//     };

//     let log_weights:Vec<f64> = (0..=max_num_spikes).scan(0.0, |state, n| {
//         if n > 0 {
//             *state += (n as f64).ln();
//         }
//         Some((n as f64 - 1.0) * (firing_rate * (period - n as f64)).ln() - *state)
//     }).collect();

//     // to avoid overflow when exponentiating, normalize the log probabilities by subtracting the maximum value
//     let max = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
//     WeightedIndex::new(log_weights.iter().map(|log_p| (log_p - max).exp()))
    
//     // let weights: Vec<f64> = log_weights.iter().map(|log_p| (log_p - max).exp()).collect(); 
//     // normalize the weights to sum to 1
//     // let norm: f64 = weights.iter().sum();
//     // WeightedIndex::new(weights.iter().map(|p| p / norm))
    
// }

// // pub fn exp_num_spikes(period: f64, firing_rate: f64) -> f64 {
// //     // calculate the expected number of spikes in a period from WeightedIndex
// //     let prob = prob_num_spikes(period, firing_rate);

// // }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_prob_num_spikes() {
//         // test probabilities for a few cases
//         assert!(prob_num_spikes(1.0, 1000.0).into_iter().zip([1.0]).all(|(a, b)| (a - b).abs() < 1e-10));
//         assert!(prob_num_spikes(1.001, 1000.0).into_iter().zip([0.0009980039920159682, 0.999001996007984]).all(|(a, b)| (a - b).abs() < 1e-10));
//         assert!(prob_num_spikes(5.0, 0.5).into_iter().zip([0.17227456258411844, 0.4306864064602961, 0.3230148048452221, 0.07178106774338269, 0.002243158366980709]).all(|(a, b)| (a - b).abs() < 1e-10));

//         // test sum of probabilities is 1
//         assert!((prob_num_spikes(100.0, 0.1).iter().sum::<f64>() - 1.0).abs() < 1e-10);
//         assert!((prob_num_spikes(100.0, 1.0).iter().sum::<f64>() - 1.0).abs() < 1e-10);
//         assert!((prob_num_spikes(10000.0, 0.1).iter().sum::<f64>() - 1.0).abs() < 1e-10);
        
//         // test length of output
//         assert_eq!(prob_num_spikes(100.1, 0.1).len(), 101);
//         assert_eq!(prob_num_spikes(100.0, 0.1).len(), 100);
//         assert_eq!(prob_num_spikes(99.9, 0.1).len(), 100);


//     }

//     // #[test]
//     // fn test_exp_num_spikes() {
//     //     assert!((exp_num_spikes(50.0, 0.1) - 4.182098894692317).abs() < 1e-12);
//     //     assert!((exp_num_spikes(50.0, 0.2) - 7.22532574319544).abs() < 1e-12);
//     //     assert!((exp_num_spikes(50.0, 0.5) - 13.010466052671848).abs() < 1e-12);
//     // }
// }