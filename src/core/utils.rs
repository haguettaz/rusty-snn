//! Utility functions and types.
use core::f64;
use std::cmp::Ordering;

/// A time-value pair to represent the value of a function at a given time.
#[derive(Debug, Clone)]
pub struct TimeValuePair<T: PartialOrd> {
    pub time: f64,
    pub value: T,
}

impl<T: PartialOrd> PartialEq for TimeValuePair<T> {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.value == other.value
    }
}

impl<T: PartialOrd> PartialOrd for TimeValuePair<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.value.partial_cmp(&other.value) {
            Some(Ordering::Equal) => self.time.partial_cmp(&other.time),
            other => other,
        }
    }
}

/// An open time interval.
#[derive(PartialEq, Debug, Clone)]
pub enum TimeInterval {
    /// A closed interval [start, end].
    Closed { start: f64, end: f64 },
    /// An empty time interval.
    Empty,
}

impl TimeInterval {
    pub fn new(start: f64, end: f64) -> Self {
        if start >= end {
            TimeInterval::Empty
        } else {
            TimeInterval::Closed { start, end }
        }
    }

    pub fn start(&self) -> Option<f64> {
        match self {
            TimeInterval::Closed { start, end: _ } => Some(*start),
            TimeInterval::Empty => None,
        }
    }

    pub fn end(&self) -> Option<f64> {
        match self {
            TimeInterval::Closed { start: _, end } => Some(*end),
            TimeInterval::Empty => None,
        }
    }

    pub fn contains(&self, time: f64) -> bool {
        match self {
            TimeInterval::Closed { start, end } => time >= *start && time <= *end,
            TimeInterval::Empty => false,
        }
    }

    pub fn length(&self) -> f64 {
        match self {
            TimeInterval::Closed { start, end } => end - start,
            TimeInterval::Empty => 0.0,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            TimeInterval::Empty => true,
            _ => false,
        }
    }

    pub fn intersect(&self, interval: TimeInterval) -> Self {
        match (self, interval) {
            (TimeInterval::Empty, _) => TimeInterval::Empty,
            (_, TimeInterval::Empty) => TimeInterval::Empty,
            (
                TimeInterval::Closed {
                    start: start1,
                    end: end1,
                },
                TimeInterval::Closed {
                    start: start2,
                    end: end2,
                },
            ) => TimeInterval::new(start1.max(start2), end1.min(end2)),
        }
    }
}

/// A union of open time intervals.
#[derive(PartialEq, Debug, Clone)]
pub enum TimeIntervalUnion {
    /// A union of right half-open intervals [start, end).
    ClosedUnion(Vec<TimeInterval>),
    /// An empty time interval.
    Empty,
}

impl TimeIntervalUnion {
    pub fn new_from(mut intervals: Vec<TimeInterval>) -> Self {
        intervals.sort_by(|interval_1, interval_2| match (interval_1, interval_2) {
            (
                TimeInterval::Closed {
                    start: start_0,
                    end: end_0,
                },
                TimeInterval::Closed {
                    start: start_1,
                    end: end_1,
                },
            ) => {
                if start_0 < start_1 {
                    Ordering::Less
                } else if start_0 > start_1 {
                    Ordering::Greater
                } else if end_0 < end_1 {
                    Ordering::Less
                } else if end_0 > end_1 {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }
            (TimeInterval::Closed { .. }, TimeInterval::Empty) => Ordering::Greater,
            (TimeInterval::Empty, TimeInterval::Closed { .. }) => Ordering::Less,
            (TimeInterval::Empty, TimeInterval::Empty) => Ordering::Equal,
        });

        let mut union_intervals = vec![];

        for interval in intervals {
            if let TimeInterval::Closed { start, end } = interval {
                if union_intervals.is_empty() {
                    union_intervals.push(interval);
                } else {
                    let last_interval = union_intervals.last().unwrap().clone();
                    if last_interval.contains(start) {
                        union_intervals.pop();
                        union_intervals.push(TimeInterval::new(
                            last_interval.start().unwrap(),
                            end.max(last_interval.end().unwrap()),
                        ));
                    } else {
                        union_intervals.push(interval);
                    }
                }
            }
        }

        if union_intervals.is_empty() {
            TimeIntervalUnion::Empty
        } else {
            TimeIntervalUnion::ClosedUnion(union_intervals)
        }
    }

    pub fn contains(&self, time: f64) -> bool {
        match self {
            TimeIntervalUnion::ClosedUnion(time_intervals) => time_intervals
                .iter()
                .any(|interval| interval.contains(time)),
            TimeIntervalUnion::Empty => false,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            TimeIntervalUnion::Empty => true,
            _ => false,
        }
    }

    pub fn iter(&self) -> std::slice::Iter<TimeInterval> {
        match self {
            TimeIntervalUnion::ClosedUnion(time_intervals) => time_intervals.iter(),
            TimeIntervalUnion::Empty => [].iter(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_contains() {
        let interval = TimeInterval::Closed {
            start: 0.0,
            end: 10.0,
        };
        assert_eq!(interval.contains(-1.0), false);
        assert_eq!(interval.contains(0.0), true);
        assert_eq!(interval.contains(5.0), true);
        assert_eq!(interval.contains(10.0), true);
        assert_eq!(interval.contains(12.0), false);

        let intervals = TimeIntervalUnion::new_from(vec![
            TimeInterval::Closed {
                start: 0.0,
                end: 0.5,
            },
            TimeInterval::Closed {
                start: 3.0,
                end: 5.0,
            },
        ]);
        assert_eq!(intervals.contains(-1.0), false);
        assert_eq!(intervals.contains(0.0), true);
        assert_eq!(intervals.contains(2.0), false);
        assert_eq!(intervals.contains(5.0), true);
        assert_eq!(intervals.contains(10.0), false);
        assert_eq!(intervals.contains(12.0), false);
    }
}
