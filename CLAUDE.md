# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

`aggregate-ntt` is a Rust compiled engine for **exact** bounded discrete
compound distributions: the number-theoretic transform over prime fields, plus
Chinese remainder reconstruction. It is an **optional accelerator** for the
Python [`aggregate`](https://github.com/mynl/aggregate) actuarial library,
discovered at import time and used when present.

It computes nothing `aggregate` cannot compute alone. It computes the same
integers faster. That framing is load bearing: correctness lives in
`aggregate`, which keeps the three end-to-end checks and an independent
polynomial oracle, and this engine is therefore **verified rather than
trusted**.

This is a side project, taken on because the pure-Python engine's numbers can
be an order of magnitude better, not because they need to be. `aggregate` at
1.2.0a5 does the 1000 x 1000 uniform dice of dice exactly in 212 s, which is
fine for any rational problem.

Author: Stephen J. Mildenhall.

## Relationship to `aggregate`

**The runtime dependency goes one way, and must stay that way.** `aggregate`
may import this package; this package must never import `aggregate` at
runtime. Only the `dev` extra depends on it, for the conformance suite. If
that edge ever reverses, the two packages can no longer be installed or
released independently, which is the whole point of the split.

The division of labor, specified in `dev/plan-rust-engine.md`:

| Stays in `aggregate` | Lives here |
|---|---|
| Input validation, clearing denominators | Per-prime transform |
| Prime selection (`ntt_primes`) | Spectral evaluation over frequency runs |
| Size guardrails, `NTT_MAX_*` | Chinese remainder reconstruction |
| The three end-to-end checks | |
| `ntt_audit`, `accuracy_df`, `stats_df` | |
| The polynomial oracle `exact_compound_poly` | |

The consuming side is specified in `dev/plan-ntt-rust-integration.md` in the
`aggregate` repository. That file and `dev/plan-rust-engine.md` here are a
**reflected pair**: each states its own side as requirements, and each is
retired on its own side when that side lands. Read both before changing the
seam.

## The seam is versioned, and refusal beats a wrong answer

`PROTOCOL_VERSION` is bumped on **any** incompatible change to the signatures
or semantics the compiled module exposes. `aggregate` reads it and refuses a
version it does not recognize. Two separately versioned packages drift; a loud
refusal is acceptable and a silently wrong answer is not. When you change a
signature, bump it, and say so in `CHANGELOG.md` and in the integration plan.

## Commands

```
uv venv
uv pip install maturin numpy pytest pytest-xdist ruff
uv pip install -e <path to an aggregate checkout>   # for the conformance suite
uv run maturin develop --release     # build and install into .venv
uv run pytest                        # conformance suite
cargo check                          # fast type check, no Python needed
cargo clippy --all-targets           # lints
cargo fmt                            # format
```

**`aggregate` is installed by hand, not by an extra.** The surface this engine
accelerates, `aggregate.ntt`, is newer than anything on PyPI, so a version
constraint in `pyproject.toml` would be unresolvable. The conformance tests
`importorskip` it: without it they skip and the seam tests still run, which is
how `cargo`-only work stays possible. Set `UV_LINK_MODE=copy` when invoking
`uv` on this drive.

`maturin develop` without `--release` builds a debug binary that is 10x to 50x
slower. **Never quote a measurement from a debug build.** Benchmarks are
release builds only.

## Testing

The suite is a **conformance suite**, not a unit-test suite, and that is the
central discipline of this repository. Every kernel here has an independent,
slower, already-trusted reference on the Python side:

- `aggregate.ntt.exact_compound` with `ntt_engine='python'` is the reference
  for the whole pipeline.
- `aggregate.ntt.exact_compound_poly` is a direct polynomial oracle,
  independent of the transform entirely.
- `aggregate.ntt._residues_mod_prime` is the reference for one prime's slice.

So every test asserts **bit-for-bit equality of integers** against one of
those, never a tolerance. Exact arithmetic has no tolerance, and a test that
allows one has stopped testing the thing that matters. Add a case to
`tests/test_conformance.py` for every kernel, and keep the random corpora
seeded so a failure reproduces.

## Code style

House rules apply (`~\.claude\CLAUDE.md`) plus the dev tree's
(`V:\dev\CLAUDE.md`). Specific to this repository:

- **NumPy-style docstrings** on every Python function, and `///` doc comments
  on every public Rust item, with the mathematics in a `# Notes` section.
  Comment the *why*: the arithmetic here is not self-explanatory and the
  overflow arguments in particular must be written down beside the code.
- **Every modular arithmetic kernel states its overflow bound in a comment.**
  This is not optional. A Montgomery reduction written as `t + m * p`
  overflows at these prime sizes, which is what killed an earlier numba
  attempt in `aggregate` (`dev/done/plan-ntt-speed.md`). If a line can reach
  `2**64` or `2**128`, the comment says why it cannot.
- **No dashes as punctuation. Ever.** Not the em dash, not `--`, not a spaced
  hyphen. Use a comma, a colon, parentheses, or a new sentence. Applies to
  docstrings, Rust doc comments, `CHANGELOG.md`, `dev/` plans and replies.
- US spelling, ISO dates, descriptive `[Bracket-Label]` names for every task,
  plan and phase, never terse codes.
- `unsafe` needs a comment naming the invariant it relies on. Prefer none.

## Versioning

SemVer. Alpha from `1.0.0a1` to `1.0.0a5` while the kernels landed, and
**1.0.0 since 2026-10-14**. The compiled seam is the public surface, so a
`PROTOCOL_VERSION` bump is an incompatible change and takes the major
version.

**The version is written in one place, `version` in `Cargo.toml`** (SemVer,
such as `1.1.0-beta.1`). `pyproject.toml` declares it `dynamic` and maturin
converts it to PEP 440 (`1.1.0b1`) for the wheel metadata. The engine
compiles it in:
`engine_version()` returns the Cargo spelling and `__version__` the PEP 440
one, both read from the binary, so a stale build shows in the audit record of
any `aggregate` run and fails `test_engine_version_matches_cargo`.
`Cargo.lock` repeats the version but is derived, and rides in the bump
commit. A bump edits that one line and rebuilds with `maturin develop
--release`; it touches no environment. A `uv sync` is for a dependency change
only.

Every capability change bumps. One bump, one commit, never batched, with the
`CHANGELOG.md` section in the same commit. Claude commits version bumps;
anything that does not bump is left uncommitted for the author. Never push.

## Deltas from house rules

- **`maturin` is the build backend, not `hatchling`.** The dev tree default
  (`V:\dev\CLAUDE.md`, "Stack defaults": *`hatchling` build backend;
  `src/PACKAGE` layout*) cannot build a compiled extension. The layout follows
  maturin's mixed convention instead: Rust in `src/`, Python in
  `python/aggregate_ntt/`. The `src/PACKAGE` rule is not violated in spirit;
  `src/` still holds the source, which here is Rust.
- **`requires-python >= 3.13`**, where the dev tree says `>= 3.13` and
  `aggregate` itself says `>= 3.12`. The `abi3-py313` feature sets the floor,
  and one `abi3` wheel then covers every later 3.x, which is worth more than
  reaching 3.12.
- **`Cargo.lock` is committed**, against the usual convention for a library
  crate. This ships as a wheel, so reproducible builds matter more than
  downstream version flexibility.
