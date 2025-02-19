# ⚡️piking Neural Network - A Rust Implementation 🦀

We consider (continuous-time) spiking neural networks and their use as robust memorizers of multichannel spike trains.
We refer to [Aguettaz and Loeliger, 2024](https://arxiv.org/abs/2408.01166) for details.

The implementation is based on the [Rust programming language](https://www.rust-lang.org/).
This language was designed with several key goals, which revolve around improving safety, performance, and concurrency.[^1]

## Installation

To install Rust, follow the instructions on the [official website](https://www.rust-lang.org/tools/install).
The installation includes the Rust compiler, Cargo (the Rust package manager), and the Rust standard library.

### Building From Source

To build the project from source, clone the repository and execute the following command in the root directory:

```bash
cargo build --release
```

You can generate and view the rusty-snn documentation locally by running:

```bash
cargo doc --open
```

### Installation From Crates.io

This feature will be available soon.

## Usage

An example of how to use the library is provided in the `examples` directory.
- `example1.rs` demonstrates how to randomly create a network, train it to memorize a random spike train, and simulate the network.

## References

* [Aguettaz and Loeliger, "Continuous-time neural networks can stably memorize random spike trains", *arXiv*, 2024.](https://arxiv.org/abs/2408.01166)
* [Klabnik and Nichols, *The Rust Programming Language*, 2021.](https://doc.rust-lang.org/book/)

[^1]: Read as concurrency or parallelism.