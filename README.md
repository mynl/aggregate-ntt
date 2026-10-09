# aggregate-ntt

A compiled engine for **exact** bounded discrete compound distributions: the
number-theoretic transform over prime fields, plus Chinese remainder
reconstruction, in Rust.

This package is an **optional accelerator** for the
[`aggregate`](https://github.com/mynl/aggregate) actuarial library. It computes
nothing that `aggregate` cannot compute on its own; it computes the same
integers faster. `aggregate` discovers it at import time and uses it when
present, falling back to its pure-Python path when absent, so you never need
this package to get a correct answer.

## Status

Alpha, and a deliberate side project. `aggregate`'s own pure-Python engine is
fast enough for any ordinary problem: it does a 500 x 500 uniform dice of dice
exactly in about 17 seconds and a 1000 x 1000 in about 212. This package exists
because those numbers can be an order of magnitude better, not because they
need to be.

## Install

```
pip install aggregate-ntt
```

`aggregate` then uses it automatically. To check:

```python
from aggregate import build
a = build('nttagg Dice dfreq [1:100] dsev [1:100]')
a.ntt_audit          # names the engine that produced the answer
```

To choose explicitly, or to rule out the accelerator when chasing a
discrepancy:

```python
build('nttagg ...', ntt_engine='python')   # or 'rust', or 'auto' (default)
```

## Build from source

Needs a Rust toolchain (see `rust-toolchain.toml`) and
[maturin](https://www.maturin.rs/).

```
uv sync --extra dev
uv run maturin develop --release
uv run pytest
```

The release profile here targets the build machine. A redistributable wheel
drops the machine-specific flags; see `[Rust-Build-Local]` in
`dev/plan-rust-engine.md`.

## Why it is a separate package

Three reasons, all of which keep `aggregate` better off.

`aggregate` stays pure Python: no wheels per platform and interpreter, no Rust
toolchain for contributors, no compiled artifact in the release process of a
library whose value is its mathematics.

The license stays clean. If this engine ever links GMP for its bigint
arithmetic, the LGPL obligation lives here, in a separate distribution, and
never reaches `aggregate`, which is BSD.

And this package can be built for one machine. A published wheel must target a
baseline instruction set; a local build does not, which is most of the point
of compiling at all.

## Documentation

The design, the milestones and the measurements are in
`dev/plan-rust-engine.md`. The consuming side is specified in
`dev/plan-ntt-rust-integration.md` in the `aggregate` repository. `CHANGELOG.md`
is the release record.

## License

BSD 3-Clause. Author: Stephen J. Mildenhall.
