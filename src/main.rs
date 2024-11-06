// use derivative::Derivative;
// use serde::{Deserialize, Serialize};
// use std::sync::mpsc::{Receiver, Sender, channel};

// #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
// pub struct Input {
//     firing_times: Vec<f64>,
//     source_id: usize,
//     delay: f64,
//     weight: f64,
// }

// #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
// pub struct Spike {
//     time: f64,
//     source_id: usize,
// }

// #[derive(Derivative, Serialize, Deserialize)]
// #[derivative(Debug, PartialEq)]
// pub struct Neuron {
//     id: usize,
//     threshold: f64,
//     firing_times: Vec<f64>,
//     inputs: Vec<Input>,
    
//     // #[serde(skip)]
//     #[derivative(Debug = "ignore", PartialEq = "ignore", Clone = "ignore")]
//     rx: Option<Receiver<Spike>>,
    
//     // #[serde(skip)]
//     #[derivative(Debug = "ignore", PartialEq = "ignore")]
//     txs: Option<Vec<Sender<Spike>>>,
// }

// impl Neuron {
//     pub fn new(id: usize, threshold: f64, inputs: Vec<Input>) -> Self {
//         let (tx, rx) = channel();
//         Neuron {
//             id,
//             threshold,
//             firing_times: Vec::new(),
//             inputs,
//             rx: Some(rx),
//             txs: Some(vec![tx]),
//         }
//     }

//     pub fn reinitialize_channels(&mut self) {
//         let (tx, rx) = channel();
//         self.rx = Some(rx);
//         self.txs = Some(vec![tx]);
//     }
// }

// fn main() {
//     // Example usage
//     let neuron = Neuron::new(1, 0.5, vec![
//         Input {
//             firing_times: vec![1.0, 2.0],
//             source_id: 1,
//             delay: 0.1,
//             weight: 0.5,
//         },
//     ]);

//     // Serialize the neuron to a JSON string
//     let serialized = serde_json::to_string(&neuron).unwrap();
//     println!("Serialized: {}", serialized);

//     // Deserialize the JSON string back to a Neuron
//     let mut deserialized: Neuron = serde_json::from_str(&serialized).unwrap();
//     deserialized.reinitialize_channels();
//     println!("Deserialized: {:?}", deserialized);
// }

// use std::thread;
// use std::sync::mpsc::channel;
// use std::sync::mpsc::{Receiver, Sender};

// use rusty_snn::spike_train::error::SpikeTrainError;

// #[derive(Debug)]
// struct Input {
//     firing_times: Vec<f64>,
//     source_id: usize,
//     delay: f64,
//     weight: f64
// }

// struct Spike {
//     time: f64,
//     source_id: usize,
// }

// struct Neuron {
//     id: usize,
//     threshold: f64,
//     firing_times: Vec<f64>,
//     inputs: Vec<Input>,
//     rx: Receiver<Spike>,
//     txs: Vec<Sender<Spike>>,
// }

// impl Neuron {
//     fn new(id: usize, threshold: f64, inputs: Vec<Input>, rx: Receiver<Spike>, txs: Vec<Sender<Spike>>) -> Self {
//         Neuron {
//             id,
//             threshold,
//             firing_times: Vec::new(),
//             inputs,
//             rx,
//             txs,
//         }
//     }

//     /// Extend the firing times of the neuron with the provided times.
//     /// If necessary, the firing times are sorted before being added.
//     /// The function returns an error if the refractory period is violated.
//     fn extend_firing_times(&mut self, firing_times: Vec<f64>) -> Result<(), SpikeTrainError> {
//         let mut firing_times = firing_times.clone();
//         firing_times.sort_by(|a, b| a.partial_cmp(b).expect("A problem occured while sorting the provided firing times."));
//         if firing_times.windows(2).map(|w| (w[1] - w[0])).any(|dt| dt <= 1.0) {
//             return Err(SpikeTrainError::RefractoryPeriodViolation);
//         }
//         match (firing_times.first(), self.firing_times.last()) {
//             (Some(&first), Some(&last)) => {
//                 if first <= last + 1.0 {
//                     return Err(SpikeTrainError::RefractoryPeriodViolation);
//                 }
//             },
//             _ => {}
//         }
//         self.firing_times.extend(firing_times);
//         Ok(())
//     }

//     fn update_inputs(&mut self, source_id: usize, time: f64) {
//         for input in self.inputs.iter_mut() {
//             if input.source_id == source_id {
//                 input.firing_times.push(time + input.delay);
//             }
//         }
//     }
// }

// impl Neuron {
//     fn new(id: usize, threshold: f64) -> Self {
//         use std::thread;
// use std::sync::mpsc::channel;

// // Create a simple streaming channel
// let (tx, rx) = channel();
//     }
// }

// fn main() {
    // // Example usage
    // let mut inputs = vec![
    //     Input { firing_times: vec![1.0, 2.0], source_id: 1, delay: 1.0, weight: 0.5 },
    //     Input { firing_times: vec![3.0], source_id: 2, delay: 2.0, weight: 1.0 },
    //     Input { firing_times: vec![4.0, 5.0], source_id: 1, delay: 4.0, weight: 0.5 },
    // ];

    // println!("Before update: {:?}", inputs);

    // // Update inputs with source_id = 1 by adding time 6.0
    // update_inputs(&mut inputs, 1, 6.0);

    // println!("After update: {:?}", inputs);
// }

// use std::sync::{Arc, Barrier, Mutex};
// use std::thread;
// use std::time::Duration;

// // Define the shared item structure
// #[derive(Debug, Clone)]
// struct SharedItem {
//     value: i32,
// }

// // Define the computational unit structure
// struct ComputationalUnit {
//     id: usize,
//     read_indices: Vec<usize>,
//     write_indices: Vec<usize>,
//     internal_state: i32,
// }

// impl ComputationalUnit {
//     fn new(id: usize, read_indices: Vec<usize>, write_indices: Vec<usize>) -> Self {
//         ComputationalUnit {
//             id,
//             read_indices,
//             write_indices,
//             internal_state: 0,
//         }
//     }

//     fn read_and_compute(&mut self, shared_items: &[Arc<Mutex<SharedItem>>]) {
//         let read_items = shared_items[self.id..self.id+2].lock().unwrap();
//         println!("Unit {} is reading items", self.id);
//         thread::sleep(Duration::from_millis(500)); // Simulate computation time
//         self.internal_state = read_items.sum() * 2; // Example computation
//         println!("Unit {} computed new internal state: {}", self.id, self.internal_state);
//     }

//     fn update_items(&self, shared_items: &[Arc<Mutex<SharedItem>>]) {
//         for &write_index in &self.write_indices {
//             let mut write_item = shared_items[write_index].lock().unwrap();
//             println!("Unit {} is writing to item {}: {:?}", self.id, write_index, *write_item);
//             thread::sleep(Duration::from_millis(500)); // Simulate computation time
//             write_item.value = self.internal_state;
//             println!("Unit {} updated item {} to: {:?}", self.id, write_index, *write_item);
//         }
//     }
// }

// fn main() {
//     let num_units = 3;
//     let num_items = 6;
//     let rounds = 3; // Number of rounds to alternate between read and write

//     // Initialize shared items
//     let shared_items: Vec<_> = (0..num_items)
//         .map(|_| Arc::new(Mutex::new(SharedItem { value: 0 })))
//         .collect();

//     // Initialize computational units
//     let units = vec![
//         ComputationalUnit::new(0, vec![0, 1], vec![2, 4]), // U1: read I1, I2; write I3, I5
//         ComputationalUnit::new(1, vec![2, 3], vec![0, 5]), // U2: read I3, I4; write I1, I6
//         ComputationalUnit::new(2, vec![4, 5], vec![1, 3]), // U3: read I5, I6; write I2, I4
//     ];

//     let barrier = Arc::new(Barrier::new(num_units));

//     // Spawn threads, each representing a computational unit
//     let handles: Vec<_> = units
//         .into_iter()
//         .map(|mut unit| {
//             let barrier = Arc::clone(&barrier);
//             let shared_items = shared_items.clone();
//             thread::spawn(move || {
//                 for round in 0..rounds {
//                     println!("Unit {} starting round {}", unit.id, round + 1);

//                     // Step 1: Read and compute new internal state
//                     unit.read_and_compute(&shared_items);

//                     // Wait for all units to finish reading and computing
//                     barrier.wait();

//                     // Step 2: Update the items with the new internal state
//                     unit.update_items(&shared_items);

//                     // Wait for all units to finish updating
//                     barrier.wait();
//                 }
//             })
//         })
//         .collect();

//     // Wait for all threads to finish
//     for handle in handles {
//         handle.join().unwrap();
//     }
// }

fn main() {}