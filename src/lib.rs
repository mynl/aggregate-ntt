//! Compiled engine for exact bounded discrete compound distributions.
//!
//! This crate is an optional accelerator for the Python `aggregate` library.
//! It computes nothing the library cannot compute on its own; it computes the
//! same integers faster, and `aggregate` falls back to pure Python when this
//! package is absent. The division of labor is deliberate and is specified in
//! `dev/plan-rust-engine.md`:
//!
//! - Python keeps input validation, denominator clearing, prime selection,
//!   the size guardrails, the audit record, and **all three end-to-end
//!   correctness checks**. Those checks therefore police this engine's answer
//!   exactly as they police the pure-Python one.
//! - Rust gets the two hot phases: the per-prime number-theoretic transform
//!   and the Chinese remainder reconstruction.
//!
//! The reconstruction is the first target because it is the larger half (86
//! percent of the threaded wall clock at `aggregate` 1.2.0a5) and because it
//! has no vectorized implementation on the Python side at all, being a
//! per-coefficient loop over Python integers.

use pyo3::prelude::*;

mod crt;
mod ntt;

/// Seam contract version, bumped on any incompatible change to the signatures
/// or semantics this module exposes.
///
/// `aggregate` reads this before using the engine and refuses a version it
/// does not know, rather than computing against a surface that moved. Two
/// separately versioned packages will drift; the one unacceptable outcome is a
/// silently wrong answer instead of a loud refusal.
const PROTOCOL_VERSION: u32 = 1;

/// The seam contract version this build implements.
#[pyfunction]
fn protocol_version() -> u32 {
    PROTOCOL_VERSION
}

/// This crate's own version, for the audit record.
#[pyfunction]
fn engine_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(protocol_version, m)?)?;
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    m.add_class::<crt::Reconstructor>()?;
    m.add_function(wrap_pyfunction!(ntt::residues_mod_prime, m)?)?;
    Ok(())
}
