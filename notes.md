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

:soon: Backward filtering forward deciding (BFFD) algorithm for (offline) learning


## Simulation
- continuous-time simulation, based on the paper of Comsa

## Learning
- continuous-time, using the most restrictive constraints by iterative refinement

## Jitter
- power iteration, unstable, works well when eigenvalue < 1
- arnoldi iteration instead??? 

## Firing Rates

default parameters: 
- 200 neurons
- 500 inputs per neuron
- period = 50 taumin
- min slope = 2.0
- max level = 0.0
- half-width = 0.2

low firing rate = 0.2 (around 7.2 firing times) -> phi = 0.00016 | ~45 constraints on average
--- firing rate = 0.25 -> phi = 0.01648 | ~55 constraints on average -> unstable eigenvalues!!!
medium firing rate = 0.5 (around 13 firing times) -> phi = 0.00066 | ~95 constraints on average
--- firing rate = 0.75 -> phi = 0.00148 | ~115 constraints on average
high firing rate = 1.0 -> phi = 0.00256 | ~130 constraints on average
extreme firing rate = 2.0 -> phi = 0.00511 | ~190 constraints on average

saturation effect for too large or too low firing rates
linear in between???

## Better Stability than Expected

**Default parameters:**
- weight in (-0.2, 0.2)
- max_level is 0.0
- min_slope is  2.0
- half_width is 0.2
- no regularization

### Half-Width

- Too short: steep slope (few hard to satisfy constraints)
- Too long: flat slope (many easy to satisfy constraints), required to be <= tau_min/2
- For stability, too short is better
- The right half-width is essential to make sure that the neuron fires even if the firing threshold is higher than the nominal value
- e.g., all with min_slope = 0.0, max_level = 0.0, spectral radii are 
    - infeasible for 0.1
    - 0.24699145550294957 (✅) for 0.2
    - 137023.20508808902 (❌) for 0.3 -> validated by network simulation with threshold noise 0.05
    - 1113690890769.3613 (❌) for 0.4
    - 2296555406949657700000000000000000 (❌) for 0.5
- e.g., all with min_slope = 1.0, max_level = 0.0, spectral radii are 
    - infeasible for 0.1
    - 0.011578460980049011 (✅) for 0.2
    - 0.24659501241182624 (✅) for 0.3
    - infeasible for 0.4
    - infeasible for 0.5

### Minimum Slope

- Not crucial for stability, but help to improve it.
- Should be positive, otherwise, the neuron might fire too early.
- e.g., spectral radii are 0.24699145550294957 (✅), 0.011578460980049011 (✅), and 0.006434780168794575 (✅) for min_slope = 0, 1, and 2, respectively.
- but too restrictive min_slope might lead to infeasible constraints, e.g., min_slope = 3.0 was infeasible

### Maximum Level

- Necessary for robustness against threshold noise.

**Remark:** all three parameters are strongly intricated.
- e.g., min_slope and half-width set a maximum level right before the firing zone
- e.g., max_level and half-width set the minimum average slope before the firing time 

### Weights

- Large magnitude weigths can break the stability of the network. But too low weights might lead to infeasible constraints.
- e.g., for weights in (-1,1) and (-10,10), the spectral radii are 94.2 (❌) and 347479.6850729508 (❌), respectively.
- Two options:
    - Bounded weights, e.g., weights in (-0.2, 0.2), spectral radius is 0.006434780168794575 (✅)
    - Regularization (better?), e.g., weights in (-1,1) and l2-regularization, spectral radius is 0.00015830453689061905 (✅)





### Experiments

For 100 runs, compute recall, precision and spectral radius for different parameters.
Reference is: minimum slope=0, maximum level=0, half-width=0.2, weight in (-0.2, 0.2), no regularization.
Threshold noise are 2%, 5%, 10%, 20%

- Minimum slope effect: min_slope = 1.0 and 2.0
- Half-width effect: half_width = 0.1, 0.2, 0.3, 0.4, 0.5
- Maximum level effect: max_level = 0.2, 0.5
- Regularization effect: l1 and l2