# TODO

Live list for `aggregate-ntt`. What has landed is in `CHANGELOG.md` and the git
log. The design and the measurements are in `dev/plan-rust-engine.md`; the
consuming side is `dev/plan-ntt-rust-integration.md` in the `aggregate`
repository, which is this plan's reflected half.

## In flight

- **[Rust-Reconstruction]** (`dev/plan-rust-engine.md`) — the Chinese
  remainder phase, the first kernel and the larger half of the cost (86
  percent of `aggregate`'s threaded wall clock). Balanced product tree built
  online as a merge stack, the same structure the Python side uses so the two
  compare level by level, with `rayon` parallelizing *inside* each wide merge
  over coefficients and the GIL released for the duration. Narrow merges stay
  serial, since at one or two primes the arithmetic is smaller than the task
  overhead. Target: the headline 1000 x 1000 case's reconstruction at or below
  25 s against 187 s today. Watch the boundary: `finish` returns fixed-width
  limbs, and the bulk conversion to Python integers must come in under 3 s or
  `[Rust-Query-Object]` gets promoted ahead of the transform.

## Next

- **[Rust-Transform]** — the per-prime transform. Compiled butterflies for the
  register win, `rayon` across primes for the real parallel speedup Python
  cannot have, and 64-bit primes via `u128`, which halves the prime count from
  321 to about 160. Two consequences to handle deliberately: prime selection
  must learn a width, and the two engines will then use different prime sets,
  so conformance tests must compare *answers* and never audit records. Carry
  the overflow finding: `t + m * p` in a Montgomery reduction overflows at
  these prime sizes, and the high-word formulation is the fix.

- **[Rust-Query-Object]** — stop materializing 1.25 GB of Python integers
  nobody reads. A handle holding the bigints on the Rust side, answering for
  the exact sum, exact moments, a float density vector and exact coefficients
  over a slice. Promote ahead of `[Rust-Transform]` if the limb conversion
  misses its budget. Needs coordinating with the integration plan, since it
  reshapes how `aggregate` builds `p_exact`.

## Recently landed

- **[Rust-Scaffold]** DONE at **1.0.0a1** (2026-10-09): repository, Cargo and
  maturin project, PyO3 0.29.3 with `abi3-py313`, mixed Rust and Python
  layout, the versioned seam (`protocol_version`, `engine_version`,
  `Reconstructor`, `residues_mod_prime`) with both kernels raising
  `NotImplementedError`, and a conformance suite whose seam tests run now and
  whose kernel tests are `xfail(strict=True)` so they become failures the
  moment a kernel works and is wrong. Built and ran: 5 passed, 9 xfailed.
  `[Rust-Dependency-Pins]` was folded in here, since the first pins attempted
  (PyO3 0.23) had no `abi3-py313` feature at all and had to be resolved to get
  a build.
