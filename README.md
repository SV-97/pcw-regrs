# pcw-regrs
Fast, optimal, extensible and cross-validated heterogeneous piecewise polynomial regression for timeseries

The algorithm is implemented in Rust in the `pcw_regrs` subdirectory. A Python API can be found in `pcw_regrs_py` and installed via `maturin`.

# Building

Note that building currently requires a the nightly rust compiler which may be easily installed using `rustup toolchain install nightly`. We're only using it for [`let-chains`](https://github.com/rust-lang/rust/issues/53667); these could be removed relatively easily if a stable variant is needed, at the expense of making the code a bit more verbose. See also the [`https://crates.io/crates/if_chain`](if_chain) crate in this regard.