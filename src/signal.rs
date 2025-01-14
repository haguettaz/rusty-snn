// to be changed to kernel.rs with the Kernel trait having required methods eval, eval derivative, crossing, max, min, etc.

use itertools::Itertools;
use lambert_w::lambert_w0;
use log::debug;

use super::spike_train::InSpike;
use super::INSPIKE_MIN;

/// Returns the neuron potential at the given time, based on all its input spikes.
pub fn potential(inspikes: &[InSpike], time: f64) -> f64 {
    inspikes
        .iter()
        .fold(0.0, |acc, inspike| acc + inspike.signal(time))
}

/// Returns the neuron potential at the given time, based on all its input spikes and their periodic extension.
/// The result only make sense if the contribution of a spike is negligible after the prescribed period (see POTENTIAL_TOLERANCE).
pub fn periodic_potential(inspikes: &[InSpike], time: f64, period: f64) -> f64 {
    inspikes.iter().fold(0.0, |acc, inspike| {
        acc + inspike.periodic_signal(time, period)
    })
}

/// Returns the neuron potential at the given time, based on all its input spikes and their periodic extension.
/// The result only make sense if the contribution of a spike is negligible after the prescribed period (see POTENTIAL_TOLERANCE).
pub fn periodic_potential_derivative(inspikes: &[InSpike], time: f64, period: f64) -> f64 {
    inspikes.iter().fold(0.0, |acc, inspike| {
        acc + inspike.periodic_signal_derivative(time, period)
    })
}

// Returns the position of the last inspike before (or equal to) time, if any.
// If several inspike appears exactly at time, returns any of them.
fn find_pos_left(inspikes: &[InSpike], time: f64) -> Option<usize> {
    match inspikes.binary_search_by(|inspike| {
        if inspike.time() > time {
            std::cmp::Ordering::Greater
        } else if inspike.time() < time {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(pos) => Some(pos),
        Err(pos) => {
            if pos > 0 {
                Some(pos - 1)
            } else {
                None
            }
        }
    }
}

// Returns the position of the last inspike whose contribution at time is negligible.
// If no such inspike exists, returns the position of the last inspike.
fn find_pos_neg(inspikes: &Vec<InSpike>, time: f64) -> Option<usize> {
    match find_pos_left(inspikes, time - 1_f64) {
        Some(end_pos) => {
            match inspikes[..=end_pos].binary_search_by(|inspike| {
                if inspike.kernel(time) > INSPIKE_MIN {
                    std::cmp::Ordering::Greater
                } else if inspike.kernel(time) < INSPIKE_MIN {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            }) {
                Ok(pos) | Err(pos) => {
                    if pos > 0 {
                        Some(pos - 1)
                    } else {
                        None
                    }
                }
            }
        }
        None => return None,
    }
}

/// Returns the next firing time of the neuron, if any.
pub fn crossing_potential(
    start: f64,
    firing_threshold: f64,
    inspikes: &Vec<InSpike>,
) -> Option<f64> {
    if inspikes.is_empty() {
        if firing_threshold == 0.0 {
            return Some(start);
        } else {
            return None;
        }
    }

    // Search the position of the last inspike before the start time
    let pos_start = find_pos_left(inspikes, start).unwrap_or(0);
    // Search the position of the last inspike whose contribution at start time is negligible.
    let pos_neg = find_pos_neg(inspikes, start).unwrap_or(0);

    for (i, inspike) in inspikes[pos_start..].iter().enumerate() {
        let (a, b) =
            inspikes[pos_neg..=pos_start + i]
                .iter()
                .fold((0.0, 0.0), |(mut a, mut b), item| {
                    a += item.weight() * (item.time() - inspike.time()).exp();
                    b += item.weight() * item.time() * (item.time() - inspike.time()).exp();
                    (a, b)
                });

        if a == 0.0 {
            if firing_threshold < b {
                let firing_time = inspike.time() + 1.0 + (-b / firing_threshold).ln();
                if firing_time >= start
                    && firing_time > inspike.time()
                    && ((pos_start + i + 1 >= inspikes.len())
                        || (firing_time <= inspikes[pos_start + i + 1].time()))
                {
                    return Some(firing_time);
                }
            }
        } else {
            if firing_threshold <= a * (inspike.time() - b / a).exp() {
                let firing_time = b / a
                    - lambert_w0(-firing_threshold / a * (b / a - 1.0 - inspike.time()).exp());
                if firing_time >= start
                    && firing_time > inspike.time()
                    && ((pos_start + i + 1 >= inspikes.len())
                        || (firing_time <= inspikes[pos_start + i + 1].time()))
                {
                    return Some(firing_time);
                }
            }
        }
    }
    None
}

/// Returns the maximum potential and the associated time in the prescribed interval.
/// The two following assumptions are made:
/// 1. The input spikes repeat themselfs with the provided period
/// 2. The contribution of an input spike on the neuron potential fades quickly away; after the provided period, its effect is negligible (see POTENTIAL_TOLERANCE).
/// The function returns None if the interval of interest is empty, i.e., start > end, or too long, i.e., end - start > period.
pub fn max_periodic_potential(
    inspikes: &Vec<InSpike>,
    start: f64,
    end: f64,
    period: f64,
) -> Option<(f64, f64)> {
    if start > end {
        debug!("The provided interval is empty [{}, {}]", start, end);
        return None;
    }
    if end - start > period {
        debug!("The provided interval is too long [{}, {}]", start, end);
        return None;
    }

    if inspikes.is_empty() {
        return Some((start, 0_f64));
    }

    // Init the global maximum and the associated time with the greatest of the two endpoints
    let (mut tmax, mut zmax) = (start, periodic_potential(inspikes, start, period));
    let tmp_zmax = periodic_potential(inspikes, end, period);
    if tmp_zmax > zmax {
        (tmax, zmax) = (end, tmp_zmax);
    }

    if inspikes.len() == 1 {
        let time = match end != inspikes[0].time() {
            true => inspikes[0].time() - ((inspikes[0].time() - end) / period).ceil() * period,
            false => end,
        };
        let tmp_tmax = match inspikes[0].weight() > 0.0 {
            true => time + 1.0,
            false => time,
        };
        if tmp_tmax < end && tmp_tmax > start {
            let tmp_zmax = periodic_potential(inspikes, tmp_tmax, period);
            if tmp_zmax > zmax {
                (tmax, zmax) = (tmp_tmax, tmp_zmax);
            }
        }
        return Some((tmax, zmax));
    }

    (tmax, zmax) = inspikes
        .iter()
        .circular_tuple_windows()
        .map(|(inspike, next_inspike)| {
            let time = inspike.time() - ((inspike.time() - end) / period).ceil() * period;
            let next_time =
                next_inspike.time() - ((next_inspike.time() - time) / period).floor() * period;
            let weight = inspike.weight();
            (weight, time, next_time)
        })
        .filter(|(_, _, next_time)| next_time >= &start)
        .map(|(weight, time, next_time)| {
            let tmax = match weight > 0.0 {
                true => {
                    let (a, b) = inspikes
                        .iter()
                        .map(|tmp_inspike| {
                            let tmp_weight = tmp_inspike.weight();
                            let tmp_time = tmp_inspike.time()
                                - ((tmp_inspike.time() - time) / period).ceil() * period;
                            (tmp_weight, tmp_time)
                        })
                        .fold((0.0, 0.0), |(acc_a, acc_b), (tmp_weight, tmp_time)| {
                            (
                                acc_a + tmp_weight * (tmp_time - time).exp(),
                                acc_b + tmp_weight * tmp_time * (tmp_time - time).exp(),
                            )
                        });
                    1.0 + b / a
                }
                false => time,
            };
            if tmax < next_time && tmax > start && tmax < end {
                (tmax, periodic_potential(inspikes, tmax, period))
            } else {
                (f64::NAN, f64::NEG_INFINITY)
            }
        })
        .fold((tmax, zmax), |(acc_t, acc_z), (t, z)| {
            if z > acc_z {
                (t, z)
            } else {
                (acc_t, acc_z)
            }
        });

    Some((tmax, zmax))
}

/// Returns the minimum potential derivative and the associated time in the prescribed interval.
/// The two following assumptions are made:
/// 1. The input spikes repeat themselfs with the provided period
/// 2. The contribution of an input spike on the neuron potential derivative fades quickly away; after the provided period, its effect is negligible (see POTENTIAL_TOLERANCE).
/// The function returns None if the interval of interest is empty, i.e., start > end, or too long, i.e., end - start > period.
pub fn min_periodic_potential_derivative(
    inspikes: &Vec<InSpike>,
    start: f64,
    end: f64,
    period: f64,
) -> Option<(f64, f64)> {
    if start > end {
        debug!("The provided interval is empty [{}, {}]", start, end);
        return None;
    }
    if end - start > period {
        debug!("The provided interval is too long [{}, {}]", start, end);
        return None;
    }

    if inspikes.is_empty() {
        return Some((start, 0_f64));
    }

    // Init the global minimum and the associated time with the lowest of the two endpoints
    let (mut tmin, mut zpmin) = (
        start,
        periodic_potential_derivative(inspikes, start, period),
    );
    let tmp_zpmin = periodic_potential_derivative(inspikes, end, period);
    if tmp_zpmin < zpmin {
        (tmin, zpmin) = (end, tmp_zpmin);
    }

    if inspikes.len() == 1 {
        let weight = inspikes[0].weight();
        let time = inspikes[0].time() - ((inspikes[0].time() - end) / period).ceil() * period;
        let tmp_tmin = match weight > 0.0 {
            true => time + 2.0,
            false => time,
        };
        if tmp_tmin < end && tmp_tmin > start {
            let tmp_zpmin = periodic_potential_derivative(inspikes, tmp_tmin, period);
            if tmp_zpmin < zpmin {
                (tmin, zpmin) = (tmp_tmin, tmp_zpmin);
            }
        }
        return Some((tmin, zpmin));
    }

    (tmin, zpmin) = inspikes
        .iter()
        .circular_tuple_windows()
        .map(|(inspike, next_inspike)| {
            let time = inspike.time() - ((inspike.time() - end) / period).ceil() * period;
            let next_time =
                next_inspike.time() - ((next_inspike.time() - time) / period).floor() * period;
            let weight = inspike.weight();
            (weight, time, next_time)
        })
        .filter(|(_, _, next_time)| next_time >= &start)
        .map(|(weight, time, next_time)| {
            let t = match weight > 0.0 {
                true => {
                    let (a, b) = inspikes
                        .iter()
                        .map(|tmp_inspike| {
                            let tmp_weight = tmp_inspike.weight();
                            let tmp_time = tmp_inspike.time()
                                - ((tmp_inspike.time() - time) / period).ceil() * period;
                            (tmp_weight, tmp_time)
                        })
                        .fold((0.0, 0.0), |(acc_a, acc_b), (tmp_weight, tmp_time)| {
                            (
                                acc_a + tmp_weight * (tmp_time - time).exp(),
                                acc_b + tmp_weight * tmp_time * (tmp_time - time).exp(),
                            )
                        });
                    2.0 + b / a
                }
                false => time,
            };
            if t < next_time && t > start && t < end {
                (t, periodic_potential_derivative(inspikes, t, period))
            } else {
                (f64::NAN, f64::INFINITY)
            }
        })
        .fold((tmin, zpmin), |(acc_t, acc_z), (t, z)| {
            if z < acc_z {
                (t, z)
            } else {
                (acc_t, acc_z)
            }
        });

    Some((tmin, zpmin))
}

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_potential() {
        // no inspike
        let inspikes: Vec<InSpike> = vec![];
        assert_eq!(potential(&inspikes, 0.0), 0.0);
        assert_eq!(potential(&inspikes, 42.0), 0.0);

        // 1 inspike producing a spike
        let inspikes: Vec<InSpike> = vec![InSpike::new(0, 1.0, 1.0)];
        assert_eq!(potential(&inspikes, 0.0), 0.0);
        assert_eq!(potential(&inspikes, 1.0), 0.0);
        assert_eq!(potential(&inspikes, 2.0), 1.0);

        // 2 inspikes canceling each other
        let inspikes: Vec<InSpike> = vec![InSpike::new(0, 1.0, 1.0), InSpike::new(1, -1.0, 1.0)];
        assert_eq!(potential(&inspikes, 0.0), 0.0);
        assert_eq!(potential(&inspikes, 1.0), 0.0);
        assert_eq!(potential(&inspikes, 2.0), 0.0);

        // 4 inspikes producing a spike
        let inspikes = vec![
            InSpike::new(0, 1.0, 1.0),
            InSpike::new(1, -0.25, 2.5),
            InSpike::new(2, 1.0, 3.0),
            InSpike::new(3, 1.0, 4.0),
        ];
        assert_eq!(potential(&inspikes, 0.0), 0.0);
        assert_eq!(potential(&inspikes, 1.0), 0.0);
        assert_eq!(potential(&inspikes, 2.0), 1.0);
        assert_eq!(potential(&inspikes, 3.0), 0.5296687235053686);
        assert_eq!(potential(&inspikes, 4.0), 1.1785568523176007);
        assert_eq!(potential(&inspikes, 5.0), 1.795450805721572);

        // many zero-weight inspikes producing no spike
        let inspikes = vec![InSpike::new(0, 0.0, 1.0); 100];
        assert_eq!(potential(&inspikes, 0.0), 0.0);
        assert_eq!(potential(&inspikes, 42.0), 0.0);
    }

    #[test]
    fn test_find_pos_left() {
        let inspikes = vec![
            InSpike::new(0, 1.0, 1.0),
            InSpike::new(1, 1.0, 2.0),
            InSpike::new(4, 1.0, 5.0),
        ];
        assert_eq!(find_pos_left(&inspikes, 0.0), None);
        assert_eq!(find_pos_left(&inspikes, 1.0), Some(0));
        assert_eq!(find_pos_left(&inspikes, 2.0), Some(1));
        assert_eq!(find_pos_left(&inspikes, 2.5), Some(1));
        assert_eq!(find_pos_left(&inspikes, 3.0), Some(1));
        assert_eq!(find_pos_left(&inspikes, 5.0), Some(2));
        assert_eq!(find_pos_left(&inspikes, 6.0), Some(2));
    }

    #[test]
    fn test_find_pos_neg() {
        let inspikes = vec![
            InSpike::new(0, 1.0, 1.0),
            InSpike::new(1, 1.0, 20.0),
            InSpike::new(4, 1.0, 50.0),
        ];

        assert_eq!(find_pos_neg(&inspikes, 0.0), None);
        assert_eq!(find_pos_neg(&inspikes, 5.0), None);
        assert_eq!(find_pos_neg(&inspikes, 10.0), None);

        let pos = find_pos_neg(&inspikes, 50.0).unwrap();

        assert!(inspikes[pos].kernel(50.0) < INSPIKE_MIN);
        if pos < inspikes.len() - 1 {
            assert!(inspikes[pos + 1].kernel(50.0) >= INSPIKE_MIN);
        }

        let pos = find_pos_neg(&inspikes, 100.0).unwrap();
        assert!(inspikes[pos].kernel(100.0) < INSPIKE_MIN);
        if pos < inspikes.len() - 1 {
            assert!(inspikes[pos + 1].kernel(100.0) >= INSPIKE_MIN);
        }

        let pos = find_pos_neg(&inspikes, 1000.0).unwrap();
        assert!(inspikes[pos].kernel(1000.0) < INSPIKE_MIN);
        if pos < inspikes.len() - 1 {
            assert!(inspikes[pos + 1].kernel(1000.0) >= INSPIKE_MIN);
        }
    }

    #[test]
    fn test_periodic_potential() {
        // no inspike
        let inspikes: Vec<InSpike> = vec![];
        assert_eq!(periodic_potential(&inspikes, 0.0, 10.0), 0.0);
        assert_eq!(periodic_potential(&inspikes, 42.0, 10.0), 0.0);

        // 1 inspike producing a spike
        let inspikes: Vec<InSpike> = vec![InSpike::new(0, 1.0, 1.0)];
        assert_eq!(
            periodic_potential(&inspikes, 0.0, 10.0),
            0.003019163651122607
        );
        assert_eq!(periodic_potential(&inspikes, 1.0, 10.0), 0.0);
        assert_eq!(periodic_potential(&inspikes, 2.0, 10.0), 1.0);

        // 2 inspikes canceling each other
        let inspikes: Vec<InSpike> = vec![InSpike::new(0, 1.0, 1.0), InSpike::new(1, -1.0, 1.0)];
        assert_eq!(periodic_potential(&inspikes, 0.0, 10.0), 0.0);
        assert_eq!(periodic_potential(&inspikes, 1.0, 10.0), 0.0);
        assert_eq!(periodic_potential(&inspikes, 2.0, 10.0), 0.0);

        // 4 inspikes producing a spike
        let inspikes = vec![
            InSpike::new(0, 1.0, 1.0),
            InSpike::new(1, -0.25, 2.5),
            InSpike::new(2, 1.0, 3.0),
            InSpike::new(3, 1.0, 4.0),
        ];
        assert_relative_eq!(
            periodic_potential(&inspikes, 0.0, 10.0),
            0.0579792,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential(&inspikes, 1.0, 10.0),
            0.0234710,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential(&inspikes, 2.0, 10.0),
            1.0098310,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential(&inspikes, 3.0, 10.0),
            0.5326879,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential(&inspikes, 4.0, 10.0),
            1.1785569,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential(&inspikes, 5.0, 10.0),
            1.7954508,
            epsilon = 1e-6
        );

        // many zero-weight inspikes producing no spike
        let inspikes = vec![InSpike::new(0, 0.0, 1.0); 100];
        assert_relative_eq!(
            periodic_potential(&inspikes, 0.0, 10.0),
            0.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential(&inspikes, 42.0, 10.0),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_periodic_potential_derivative() {
        // no inspike
        let inspikes: Vec<InSpike> = vec![];
        assert_relative_eq!(periodic_potential_derivative(&inspikes, 0.0, 10.0), 0.0);
        assert_relative_eq!(periodic_potential_derivative(&inspikes, 42.0, 10.0), 0.0);

        // 1 inspike producing a spike
        let inspikes: Vec<InSpike> = vec![InSpike::new(0, 1.0, 1.0)];
        assert_relative_eq!(
            periodic_potential_derivative(&inspikes, 0.0, 10.0),
            -0.0026837,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential_derivative(&inspikes, 1.0, 10.0),
            2.7182818,
            epsilon = 1e-6
        );
        assert_relative_eq!(periodic_potential_derivative(&inspikes, 2.0, 10.0), 0.0);
        assert_relative_eq!(
            periodic_potential_derivative(&inspikes, 3.0, 10.0),
            -0.3678794,
            epsilon = 1e-6
        );

        // 2 inspikes canceling each other
        let inspikes: Vec<InSpike> = vec![InSpike::new(0, 1.0, 1.0), InSpike::new(1, -1.0, 1.0)];
        assert_relative_eq!(periodic_potential_derivative(&inspikes, 0.0, 10.0), 0.0);
        assert_relative_eq!(periodic_potential_derivative(&inspikes, 1.0, 10.0), 0.0);
        assert_relative_eq!(periodic_potential_derivative(&inspikes, 2.0, 10.0), 0.0);

        // 4 inspikes producing a spike
        let inspikes = vec![
            InSpike::new(0, 1.0, 1.0),
            InSpike::new(1, -0.25, 2.5),
            InSpike::new(2, 1.0, 3.0),
            InSpike::new(3, 1.0, 4.0),
        ];
        assert_relative_eq!(
            periodic_potential_derivative(&inspikes, 0.0, 10.0),
            -0.0488029,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential_derivative(&inspikes, 1.0, 10.0),
            2.6980632,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential_derivative(&inspikes, 2.0, 10.0),
            -0.0086345,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential_derivative(&inspikes, 3.0, 10.0),
            2.1416285,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential_derivative(&inspikes, 4.0, 10.0),
            2.5234276,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            periodic_potential_derivative(&inspikes, 5.0, 10.0),
            -0.4335668,
            epsilon = 1e-6
        );

        // many zero-weight inspikes producing no spike
        let inspikes = vec![InSpike::new(0, 0.0, 1.0); 100];
        assert_relative_eq!(periodic_potential_derivative(&inspikes, 0.0, 10.0), 0.0);
        assert_relative_eq!(periodic_potential_derivative(&inspikes, 42.0, 10.0), 0.0);
    }

    #[test]
    fn test_crossing_potential() {
        // no inspike
        let inspikes: Vec<InSpike> = vec![];
        assert_eq!(crossing_potential(0.0, 1.0, &inspikes), None);
        assert_eq!(crossing_potential(0.0, 0.0, &inspikes), Some(0.0));

        // 1 inspike producing a spike
        let inspikes: Vec<InSpike> = vec![InSpike::new(0, 1.0, 1.0)];
        assert_eq!(crossing_potential(0.0, 1.0, &inspikes), Some(2.0));

        // 2 inspikes canceling each other
        let inspikes: Vec<InSpike> = vec![InSpike::new(0, 1.0, 1.0), InSpike::new(1, -1.0, 1.0)];
        assert_eq!(crossing_potential(0.0, 1.0, &inspikes), None);

        // 4 inspikes producing a spike
        let inspikes = vec![
            InSpike::new(0, 1.0, 1.0),
            InSpike::new(1, -0.25, 1.5),
            InSpike::new(2, 1.0, 3.0),
            InSpike::new(3, 1.0, 4.0),
        ];
        assert_eq!(
            crossing_potential(0.0, 1.0, &inspikes),
            Some(3.2757576038986502)
        );

        // many zero-weight inspikes producing no spike
        let inspikes = vec![InSpike::new(0, 0.0, 1.0); 100];
        assert_eq!(crossing_potential(0.0, 1.0, &inspikes), None);

        // many inspikes producing no spike because of extreme firing threshold
        let inspikes = vec![InSpike::new(0, 0.0, 1.0); 100];
        assert_eq!(crossing_potential(0.0, f64::INFINITY, &inspikes), None);
    }

    #[test]
    fn test_max_periodic_potential() {
        let inspikes: Vec<InSpike> = vec![];

        assert_eq!(max_periodic_potential(&inspikes, 100.0, 0.0, 10.0), None);
        assert_eq!(max_periodic_potential(&inspikes, 0.0, 500.0, 10.0), None);

        // Without any input spike
        let (tmax, zmax) = max_periodic_potential(&inspikes, 0.0, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 0.0);
        assert_relative_eq!(zmax, 0.0);

        // With a single input spike
        let inspikes: Vec<InSpike> = vec![InSpike::new(0, 1.0, 1.0)];

        let (tmax, zmax) = max_periodic_potential(&inspikes, 0.0, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 2.0);
        assert_relative_eq!(zmax, 1.0);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 2.5, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 2.5);
        assert_relative_eq!(zmax, 0.909795, epsilon = 1e-6);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 70.0, 80.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 72.0);
        assert_relative_eq!(zmax, 1.0, epsilon = 1e-6);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 1.25, 2.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 2.0);
        assert_relative_eq!(zmax, 1.0);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 2.0, 3.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 2.0);
        assert_relative_eq!(zmax, 1.0);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 1.0, 3.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 2.0);
        assert_relative_eq!(zmax, 1.0);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 0.0, 1.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 0.0);
        assert_relative_eq!(zmax, 0.003019, max_relative = 1e-2);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 8.0, 11.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 8.0);
        assert_relative_eq!(zmax, 0.01735126523666451);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 8.0, 12.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 12.0);
        assert_relative_eq!(zmax, 1.0);

        // With multiple input spikes
        let inspikes: Vec<InSpike> = vec![
            InSpike::new(0, 1.0, 1.0),
            InSpike::new(1, 1.0, 2.5),
            InSpike::new(3, -1.0, 3.5),
            InSpike::new(2, 1.0, 4.0),
        ];

        let (tmax, zmax) = max_periodic_potential(&inspikes, 0.0, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 3.226, max_relative = 1e-2);
        assert_relative_eq!(zmax, 1.609, max_relative = 1e-2);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 2.0, 4.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 3.226, max_relative = 1e-2);
        assert_relative_eq!(zmax, 1.609, max_relative = 1e-2);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 3.0, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 3.226, max_relative = 1e-2);
        assert_relative_eq!(zmax, 1.609, max_relative = 1e-2);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 5.0, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 5.0);
        assert_relative_eq!(zmax, 0.847, max_relative = 1e-2);

        let (tmax, zmax) = max_periodic_potential(&inspikes, 500.0, 507.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 503.226, max_relative = 1e-2);
        assert_relative_eq!(zmax, 1.609, max_relative = 1e-2);
    }

    #[test]
    fn test_min_periodic_potential_derivative() {
        // let inputs: Vec<Input> = vec![];
        let inspikes: Vec<InSpike> = vec![];

        assert_eq!(
            min_periodic_potential_derivative(&inspikes, 100.0, 0.0, 10.0),
            None
        );
        assert_eq!(
            min_periodic_potential_derivative(&inspikes, 0.0, 500.0, 10.0),
            None
        );

        // Without any input spike
        let inspikes: Vec<InSpike> = vec![];

        let (tmax, zmax) = min_periodic_potential_derivative(&inspikes, 0.0, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmax, 0.0);
        assert_relative_eq!(zmax, 0.0);

        // With a single input spike
        let inspikes: Vec<InSpike> = vec![InSpike::new(0, 1.0, 1.0)];

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 0.0, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 3.0);
        assert_relative_eq!(zmin, -0.367879, epsilon = 1e-6);

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 3.5, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 3.5);
        assert_relative_eq!(zmin, -0.334695, epsilon = 1e-6);

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 70.0, 80.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 73.0);
        assert_relative_eq!(zmin, -0.367879, epsilon = 1e-6);

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 2.0, 3.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 3.0);
        assert_relative_eq!(zmin, -0.367879, epsilon = 1e-6);

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 3.0, 4.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 3.0);
        assert_relative_eq!(zmin, -0.367879, epsilon = 1e-6);

        // With multiple input spikes
        let inspikes: Vec<InSpike> = vec![
            InSpike::new(0, 1.0, 1.0),
            InSpike::new(1, 1.0, 2.5),
            InSpike::new(3, -1.0, 3.5),
            InSpike::new(2, 1.0, 4.0),
        ];

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 0.0, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 3.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.0547065, epsilon = 1e-6);

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 2.0, 4.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 3.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.0547065, epsilon = 1e-6);

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 3.0, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 3.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.0547065, epsilon = 1e-6);

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 5.0, 10.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 5.728699, epsilon = 1e-6);
        assert_relative_eq!(zmin, -0.321555, epsilon = 1e-6);

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 5.0, 11.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 5.728699, epsilon = 1e-6);
        assert_relative_eq!(zmin, -0.321555, epsilon = 1e-6);

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 8.0, 14.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 13.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.0547065, epsilon = 1e-6);

        let (tmin, zmin) = min_periodic_potential_derivative(&inspikes, 50.0, 60.0, 10.0).unwrap();
        assert_relative_eq!(tmin, 53.5, epsilon = 1e-6);
        assert_relative_eq!(zmin, -3.0547065, epsilon = 1e-6);
    }
}
