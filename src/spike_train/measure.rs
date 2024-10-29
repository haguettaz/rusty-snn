use super::spike_train::{PeriodicSpikeTrain, SpikeTrain};
use itertools::Itertools;

// Warning: border cases are not handled here, e.g.,
pub fn similarity_measure(
    ref_spike_train: &PeriodicSpikeTrain,
    sim_spike_train: &SpikeTrain,
) -> (f64, f64) {
    if ref_spike_train.period() != sim_spike_train.duration() {
        panic!("The period of the reference spike train must be equal to the duration of the simulated spike train.");
    }

    // create an iterator over every possible lag between the reference and simulated spike trains
    let lag_iter = ref_spike_train
        .firing_times()
        .iter()
        .zip_eq(sim_spike_train.firing_times().iter())
        .flat_map(|(ref_times, sim_times)| {
            ref_times
                .iter()
                .cartesian_product(sim_times.iter())
                .map(|(ref_t, sim_t)| sim_t - ref_t)
        });

    // if lag_iter is empty, count the number of channels which are pairs of empty spike trains
    if lag_iter.clone().next().is_none() {
        let score: f64 = ref_spike_train
            .firing_times()
            .iter()
            .zip_eq(sim_spike_train.firing_times().iter())
            .filter(|(ref_times, sim_times)| ref_times.is_empty() && sim_times.is_empty())
            .count() as f64
            / ref_spike_train.num_channels() as f64;
        return (score, score);
    }

    let distance_iter = lag_iter.map(move |lag| {
        // iterate over the channels of the reference and simulated spike trains
        ref_spike_train
            .firing_times()
            .iter()
            .zip_eq(sim_spike_train.firing_times().iter())
            // for each pair of channels, calculate the precision and recall
            .map(
                move |(ref_times, sim_times)| match (ref_times.is_empty(), sim_times.is_empty()) {
                    (true, true) => (1.0, 1.0),
                    (true, false) => (0.0, 0.0),
                    (false, true) => (0.0, 0.0),
                    (false, false) => {
                        let tmp = ref_times
                            .iter()
                            .map(|&ref_t| {
                                sim_times
                                    .iter()
                                    .filter_map(|&sim_t| {
                                        let abs_dist = (sim_t - ref_t - lag).abs();
                                        let mod_dist =
                                            abs_dist.min(ref_spike_train.period() - abs_dist);
                                        if mod_dist < 0.5 {
                                            Some(1.0 - 2.0 * mod_dist)
                                        } else {
                                            None
                                        }
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>();

                        let p = tmp / ref_times.len() as f64;
                        let r = tmp / sim_times.len() as f64;
                        (p, r)
                    }
                },
            )
            .reduce(|(acc_p, acc_r), (p, r)| (acc_p + p, acc_r + r))
            .unwrap()
    });
    let (precision, recall) = distance_iter
        .reduce(|(max_p, max_r), (p, r)| (max_p.max(p), max_r.max(r)))
        .unwrap_or((0.0, 0.0));

    (
        precision / ref_spike_train.num_channels() as f64,
        recall / ref_spike_train.num_channels() as f64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_measure() {
        let sim_spike_train =
            SpikeTrain::build(vec![vec![2.0, 3.0, 4.0], vec![1.2, 3.2]], 1.0, 5.0).unwrap();
        let ref_spike_train =
            PeriodicSpikeTrain::build(vec![vec![1.0, 2.0, 3.0], vec![0.2, 2.2]], 4.0).unwrap();
        assert_eq!(
            similarity_measure(&ref_spike_train, &sim_spike_train,),
            (1.0, 1.0)
        );

        let sim_spike_train =
            SpikeTrain::build(vec![vec![2.0, 3.0, 4.0], vec![]], 1.0, 5.0).unwrap();
        let ref_spike_train = PeriodicSpikeTrain::build(vec![vec![], vec![1.2, 3.2]], 4.0).unwrap();
        assert_eq!(
            similarity_measure(&ref_spike_train, &sim_spike_train,),
            (0.0, 0.0)
        );

        let sim_spike_train = SpikeTrain::build(vec![vec![], vec![]], 1.0, 5.0).unwrap();
        let ref_spike_train = PeriodicSpikeTrain::build(vec![vec![], vec![]], 4.0).unwrap();
        assert_eq!(
            similarity_measure(&ref_spike_train, &sim_spike_train,),
            (1.0, 1.0)
        );

        let sim_spike_train = SpikeTrain::build(vec![vec![], vec![2.0]], 1.0, 5.0).unwrap();
        let ref_spike_train = PeriodicSpikeTrain::build(vec![vec![1.0], vec![]], 4.0).unwrap();
        assert_eq!(
            similarity_measure(&ref_spike_train, &sim_spike_train,),
            (0.0, 0.0)
        );

        let sim_spike_train =
            SpikeTrain::build(vec![vec![2.0, 3.0, 4.0], vec![], vec![], vec![]], 1.0, 5.0).unwrap();
        let ref_spike_train =
            PeriodicSpikeTrain::build(vec![vec![], vec![], vec![1.2, 3.2], vec![1.2, 3.2]], 4.0)
                .unwrap();
        assert_eq!(
            similarity_measure(&ref_spike_train, &sim_spike_train,),
            (0.25, 0.25)
        );
    }
}
