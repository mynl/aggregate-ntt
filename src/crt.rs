//! Chinese remainder reconstruction, the engine's first milestone.
//!
//! Scaffold: the seam is real and exercised, the kernel is not written yet.
//! `[Rust-Reconstruction]` in `dev/plan-rust-engine.md` specifies it.
//!
//! # The shape, and why it is a stateful accumulator
//!
//! The obvious signature takes every prime's residue vector at once and
//! returns the answer. It is also wrong here: 321 vectors of 10^6 `u32` is
//! 1.3 GB handed across the boundary on top of the 1.25 GB the answer itself
//! occupies, and the Python caller currently consumes residue vectors one at
//! a time as its thread pool produces them. So the accumulator preserves that
//! online property: push each prime's residues as it arrives, and the merge
//! stack lives on the Rust side.
//!
//! The merge stack is the same structure `aggregate.ntt` uses in pure Python
//! (a balanced product tree built online, each partial tagged with the number
//! of primes behind it, equal-weight neighbors merged on arrival), so the two
//! implementations can be compared level by level while this is developed.
//!
//! # The boundary, which is where a naive port loses its winnings
//!
//! `finish` must not build a million Python integers one at a time. Every
//! coefficient is smaller than the final modulus, so the width is uniform:
//! the answer comes back as one contiguous `(n_coeffs, n_limbs)` array of
//! little-endian 64-bit limbs, which is a memory copy rather than a million
//! object allocations, and the caller converts in bulk. Milestone 3 adds a
//! query object that answers for moments and tail probabilities without
//! materializing the coefficients at all.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// One partial reconstruction: coefficients exact modulo `modulus`, with
/// `weight` primes behind them.
///
/// `weight` is what makes the merge stack a *balanced* tree: a partial is only
/// ever merged with one of equal weight, so operand sizes double level by
/// level and no merge is lopsided.
#[allow(dead_code)]
struct Partial {
    coefficients: Vec<num_bigint::BigUint>,
    modulus: num_bigint::BigUint,
    weight: usize,
}

/// Online Chinese remainder reconstruction over NTT primes.
///
/// Construct with the primes in the order residues will be pushed, push one
/// residue vector per prime, then call `finish` once.
#[pyclass]
pub struct Reconstructor {
    primes: Vec<u32>,
    n_coefficients: usize,
    pushed: usize,
    #[allow(dead_code)]
    stack: Vec<Partial>,
}

#[pymethods]
impl Reconstructor {
    /// Create an accumulator for `primes` over `n_coefficients` coefficients.
    ///
    /// The primes must be distinct and below `2**32`; they are the moduli the
    /// caller's transform phase worked in, in the order its residues arrive.
    #[new]
    fn new(primes: Vec<u32>, n_coefficients: usize) -> PyResult<Self> {
        if primes.is_empty() {
            return Err(PyValueError::new_err("primes must not be empty"));
        }
        let mut sorted = primes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != primes.len() {
            return Err(PyValueError::new_err("primes must be distinct"));
        }
        Ok(Self {
            primes,
            n_coefficients,
            pushed: 0,
            stack: Vec::new(),
        })
    }

    /// Number of primes this accumulator expects.
    #[getter]
    fn n_primes(&self) -> usize {
        self.primes.len()
    }

    /// Number of residue vectors pushed so far.
    #[getter]
    fn pushed(&self) -> usize {
        self.pushed
    }

    /// Push one prime's residue vector, in the order the primes were given.
    ///
    /// The array must hold `n_coefficients` values, each already reduced
    /// modulo the corresponding prime.
    fn push(&mut self, residues: PyReadonlyArray1<u64>) -> PyResult<()> {
        if self.pushed >= self.primes.len() {
            return Err(PyValueError::new_err(
                "all residue vectors have already been pushed",
            ));
        }
        let view = residues.as_array();
        if view.len() != self.n_coefficients {
            return Err(PyValueError::new_err(format!(
                "expected {} coefficients, got {}",
                self.n_coefficients,
                view.len()
            )));
        }
        Err(PyNotImplementedError::new_err(
            "[Rust-Reconstruction] merge stack not implemented yet; see \
             dev/plan-rust-engine.md",
        ))
    }

    /// Finish and return `(limbs, modulus_bytes)`.
    ///
    /// `limbs` is a `(n_coefficients, n_limbs)` array of little-endian 64-bit
    /// limbs, one row per coefficient, zero padded to a uniform width.
    /// `modulus_bytes` is the product of the primes, little-endian, which the
    /// caller turns into an integer with `int.from_bytes(..., 'little')`.
    fn finish(&mut self, _py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyBytes>)> {
        Err(PyNotImplementedError::new_err(
            "[Rust-Reconstruction] merge stack not implemented yet; see \
             dev/plan-rust-engine.md",
        ))
    }
}
