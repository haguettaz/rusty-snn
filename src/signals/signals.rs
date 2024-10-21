// signals == spike trains
// needs to implement signal visualization
// needs to implement signal processing (similarity measure, sampling)
enum Signal {
    SpikeTrain(Vec<f64>), // spike train
    GradedSignal, // functional signal
}