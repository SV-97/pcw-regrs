# pcw-regrs
Fast, optimal, extensible and cross-validated heterogeneous piecewise polynomial regression for timeseries.

The algorithm is implemented in Rust in the `pcw_regrs` subdirectory. A Python API can be found in `pcw_regrs_py` and installed via `maturin`.

# General structure

![A basic pipeline diagram for the algorithm. The input consists of a timeseries sample as well as the optional parameters. These are sent into a preprocessing stage for validation and to optionally calculate the training errors. Once that's finished we move on to the main algorithm which solves the main dynamic program, determines the CV and model functions and then determines the CV- or OSE-optimal piecewise functions. Finally a postprocessing stage is used to select cut representatives either by calculating segment middlepoints or running a continuity optimization](./arch.svg)

# Building

> [!NOTE]
> Note that building currently requires a the nightly rust compiler which may be easily installed using `rustup toolchain install nightly`. We're only using it for [`let-chains`](https://github.com/rust-lang/rust/issues/53667); these could be removed relatively easily if a stable variant is needed, at the expense of making the code a bit more verbose. See also the [`https://crates.io/crates/if_chain`](if_chain) crate in this regard.

## Rust

The pure Rust part is a regular `cargo` project and can be managed accordingly. Note also that `pcw-regrs` has been published to [crates.io](https://crates.io/).

## Python

> [!IMPORTANT]
> Usually building from source is not necessary and you can simply install `pcw-regrs-py` via `pip`.

To build the python package install [maturin](https://www.maturin.rs/) for example via `pip` by running

```bash
pip install maturin
````

Running

```bash
maturin develop --release
```

from the python subfolder (`pcw_regrs_py`) will compile an release (=optimized) build, build a python wheel and install it locally. For more details like how to cross compile please consult the maturin documentation.
