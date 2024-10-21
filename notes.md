# Package Structure

```
lib
├── snn 
│   └── snn // (build to create network : recurrent or feedforward)
│
├── spikes
│   ├── spikes // (with struct/enum? for spike train, duration or period, firing times)
│   ├── rand // (generate random spike trains)
│   ├── measure // similarity measure for spike trains
│   └── viz // (plotting spike trains)ﬁﬁ
├── learning
│   ├── offline
│   ├── online // not implemented yet
│   └── eigs // (spectral analysis)
└── sim
    ├── ??? (struct/enum Sim)
    └── ???
```

Or workspace is the following crates: network (simulation) and learning.
Needs to check the rust standard library...

# Parallelism

Which parallelism model to use? 
Threads? 
Futures and tasks?
A combination of both?

## Simulating Spiking Neural Networks

*CPU-bound!*

The simulation of spiking neural networks (SNN) is a computationally intensive task.
However, one can highly benefit from the event-driven nature of SNNs to parallelize the simulation.

To update its state, a neuron needs:
- information about the firing times of presynaptic neurons
- and some additional **personal** information (own firing times for refractory period, synaptic kernel, etc.)

In particular, concerning firing times permission, we have the following rules:
- A neuron owns its firing times, it has a unique permission to mutate and read them.
- A neuron can only read the firing times of other neurons.

## Training Spiking Neural Networks

Not necessarily needed for offline learning as solving the optimization problem for each neuron is the mane bottleneck.
For online learning, it might be more relevant.

:soon: Backward filtering forward deciding (BFFD) algorithm for (offline) learning.

## Questions

```rust
// Trigger error[E0507]: cannot move out of index of `Vec<String>`
fn main() {
    let v = vec![String::from("Hello ")];
    let mut s = v[0];
    s.push_str("world");
    println!("{s}");
  }
```
&rarr; Non-copyable types cannot be moved out of a vector by indexing



