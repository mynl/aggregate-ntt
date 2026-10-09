//! Per-prime number-theoretic transform, the engine's second milestone.
//!
//! Scaffold: the seam is declared, the kernel is not written yet.
//! `[Rust-Transform]` in `dev/plan-rust-engine.md` specifies it.
//!
//! # Why this is second
//!
//! On the Python side this phase is already vectorized with `numpy` `uint64`
//! ufuncs and runs on a thread pool, so it is the half that has already had
//! its easy wins. It is still worth compiling, for two reasons that only
//! appear here. A compiled butterfly keeps its operands in registers where
//! `numpy` pays a full memory pass per ufunc call, which is worth 5x to 20x.
//! And the thread pool on the Python side is throttled by the GIL it has to
//! reacquire between ufunc calls, which is why eight workers there inflate
//! transform thread-seconds by 13x; `rayon` behind a released GIL does not.
//!
//! # The prime size, which is the real prize
//!
//! Python is pinned to primes below `2**32` because a product of two residues
//! must fit in `u64` for the vectorized multiply. Rust has `u128`, so 64-bit
//! primes cost nothing structurally, and that **halves the prime count**: 321
//! primes become about 160 for the same coefficient bound. The transform does
//! half as many passes, and the reconstruction's bottom levels, which are
//! pure per-coefficient overhead, halve with it.
//!
//! Beware the sequence that defeated the abandoned numba attempt, recorded in
//! `aggregate`'s `dev/done/plan-ntt-speed.md`: a Montgomery reduction written
//! as `t + m * p` overflows at these prime sizes, because every prime sits
//! just under its word boundary and the sum reaches almost `2**65`. The
//! high-word formulation is the fix.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

/// One prime's residues of the cleared-denominator compound polynomial.
///
/// Forward transform of the severity polynomial, spectral evaluation over the
/// frequency outcome runs, inverse transform. Pure function of its arguments,
/// so it releases the GIL and is safe to call from several threads.
///
/// Parameters mirror `aggregate.ntt._residues_mod_prime` so the two can be
/// compared directly: `prime` and `root` are the field and a primitive
/// `length`-th root of unity in it, `length` is the padded transform length,
/// `support` the highest outcome, `sev_index` and `sev_weights` the severity
/// polynomial's sparse support and its cleared integer weights, `freq_runs`
/// the frequency outcomes decomposed into maximal equal-weight runs, `ds` the
/// severity denominator and `kmax` the largest claim count.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn residues_mod_prime(
    _prime: u64,
    _root: u64,
    _length: usize,
    _support: usize,
    _sev_index: PyReadonlyArray1<i64>,
    _sev_weights: PyReadonlyArray1<u64>,
    _freq_runs: Vec<(u64, u64, u64)>,
    _ds: PyReadonlyArray1<u64>,
    _kmax: u64,
) -> PyResult<()> {
    Err(PyNotImplementedError::new_err(
        "[Rust-Transform] transform kernel not implemented yet; see \
         dev/plan-rust-engine.md",
    ))
}
