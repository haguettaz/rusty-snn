# ⚡️piking Neural Network - A Rust Implementation 🦀 

We consider (continuous-time) spiking neural networks and their use as robust memorizers of arbitrary spike trains. 
For details, we refer to [Aguettaz and Loeliger, 2024](https://arxiv.org/abs/2408.01166). 

The implementation is based on the [Rust programming language](https://www.rust-lang.org/).
This language was designed with several key goals in mind, which revolve around improving safety, performance, and concurrency[^1], while making systems programming more accessible and ergonomic.

## Installation

To install Rust, follow the instructions on the [official website](https://www.rust-lang.org/tools/install).
The installation includes the Rust compiler, Cargo, which is the Rust package manager, and the Rust standard library.

### From Source

To build the project from source, clone the repository and run the following command in the root directory:
```bash
cargo build --release
```
Then, you can start using the crate by importing it into your project.
It is also possible to generate the documentation by running:
```bash
cargo doc --open
```

### From Crates.io

Coming soon...

## References

* [Aguettaz and Loeliger, "Continuous-time neural networks can stably memorize random spike trains", *arXiv*, 2024.](https://arxiv.org/abs/2408.01166)
* [Klabnik and Nichols, *The Rust Programming Language*, 2021.](https://doc.rust-lang.org/book/)

[^1]: Read as concurrency or parallelism.