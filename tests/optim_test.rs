// use core::f64;

// use rusty_snn::alpha::neuron::{AlphaInputSpikeTrain, AlphaNeuron};
// use rusty_snn::core::neuron::{Input, InputSpikeTrain, Neuron};
// use rusty_snn::core::optim::{Objective, TimeTemplate};
// use rusty_snn::core::spikes::MultiChannelCyclicSpikeTrain;

// #[test]
// fn test_memorize_empty_cyclic_spike_train() {
//     let mut neuron = AlphaNeuron::new_empty(0, 0);
//     neuron.add_input(1, f64::NAN, 1.0);
//     neuron.add_input(2, f64::NAN, 2.0);
//     neuron.add_input(3, f64::NAN, 5.5);

//     let cyclic_spike_train = MultiChannelCyclicSpikeTrain {
//         spike_train: vec![
//             vec![2.0, 4.0, 5.0],
//             vec![6.0, 25.0, 28.0],
//             vec![0.0, 72.0, 84.0],
//         ],
//         period: 100.0,
//     };
//     let time_templates = TimeTemplate::new_cyclic_from(&vec![], 0.5, 100.0);
//     let input_spike_trains = AlphaInputSpikeTrain::new_cyclic_from(
//         neuron.inputs(),
//         &cyclic_spike_train,
//         &time_templates.interval,
//     );

//     neuron
//         .memorize(
//             vec![time_templates],
//             vec![input_spike_trains],
//             (-1.0, 1.0),
//             0.0,
//             0.0,
//             Objective::L2,
//         )
//         .expect("Memorization error");

//     assert_eq!(
//         neuron.inputs(),
//         &vec![
//             Input {
//                 source_id: 1,
//                 weight: 0.0,
//                 delay: 1.0
//             },
//             Input {
//                 source_id: 2,
//                 weight: 0.0,
//                 delay: 2.0
//             },
//             Input {
//                 source_id: 3,
//                 weight: 0.0,
//                 delay: 5.5
//             }
//         ]
//     );
// }
