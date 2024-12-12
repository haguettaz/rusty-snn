/// Calculate the minimum distance between two points on a circular modulo space.
pub fn mod_dist(x: f64, y: f64, modulo: f64) -> f64 {
    let diff1 = (x - y).rem_euclid(modulo);
    let diff2 = (y - x).rem_euclid(modulo);
    diff1.min(diff2)
}