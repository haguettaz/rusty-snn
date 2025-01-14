

pub fn mean<I>(iter: I) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    match iter
        .map(|val| (val, 1_usize))
        .reduce(|acc, val| (acc.0 + val.0, acc.1 + val.1))
    {
        Some((sum, count)) => Some(sum / (count as f64)),
        None => None,
    }
}

pub fn median<I>(iter: I) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    let mut values: Vec<f64> = iter.collect();
    let len = values.len();
    if len == 0 {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    match len % 2 == 0 {
        true => Some((values[len / 2] + values[len / 2 - 1]) / 2.0),
        false => Some(values[(len - 1) / 2]),
    }
}

pub fn find_mod_insert_position(arr: &[f64], value: f64, period: f64) -> usize {
    let value_mod = value.rem_euclid(period);
    arr.binary_search_by(|&x| {
        let x_mod = x.rem_euclid(period);
        x_mod.partial_cmp(&value_mod).unwrap()
    })
    .unwrap_or_else(|pos| pos)
}

pub fn find_insert_position(arr: &[f64], value: f64) -> usize {
    arr.binary_search_by(|&x| x.partial_cmp(&value).unwrap())
        .unwrap_or_else(|pos| pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_insert_position() {
        let arr = vec![1.0, 3.0, 8.0];
        assert_eq!(find_insert_position(&arr, 2.0), 1);
        assert_eq!(find_insert_position(&arr, 32.0), 3);
        assert_eq!(find_insert_position(&arr, 9.0), 3);
    }

    #[test]
    fn test_find_mod_insert_position() {
        let arr = vec![1.0, 3.0, 8.0];
        assert_eq!(find_mod_insert_position(&arr, 2.0, 10.0), 1);
        assert_eq!(find_mod_insert_position(&arr, 32.0, 10.0), 1);
        assert_eq!(find_mod_insert_position(&arr, 9.0, 10.0), 3);
    }
}
