"""
Conformance of the compiled engine against the pure-Python reference.

Every kernel here has an independent, slower, already-trusted reference in
:mod:`aggregate.ntt`, so every test in this file asserts **bit-for-bit
equality of integers**, never a tolerance. Exact arithmetic has no tolerance,
and a test that allows one has stopped testing the property that matters.

Three references, in increasing independence:

- :func:`aggregate.ntt.exact_compound` on its pure-Python path, the same
  algorithm this engine reimplements.
- :func:`aggregate.ntt.exact_compound_poly`, a direct polynomial computation
  that does not use a transform at all, so it shares no machinery with either
  implementation.
- The three end-to-end checks inside ``exact_compound``, which run on this
  engine's answer too: coefficients sum to exactly the denominator, none is
  negative, and none beyond the support is nonzero.

The scaffold's kernels raise ``NotImplementedError``, so the kernel tests are
expected failures until ``[Rust-Reconstruction]`` lands. The seam tests run
now and pin the contract.
"""

import random
from fractions import Fraction

import numpy as np
import pytest

import aggregate_ntt

aggregate_ntt_module = aggregate_ntt


# --------------------------------------------------------------- [the seam]

def test_protocol_version_is_an_integer():
    """The seam contract version is readable and sane."""
    assert isinstance(aggregate_ntt.PROTOCOL_VERSION, int)
    assert aggregate_ntt.PROTOCOL_VERSION >= 1


def test_engine_version_matches_cargo():
    """``engine_version`` reports the Cargo version, not the PEP 440 one.

    The two spellings differ deliberately (``1.0.0-alpha.1`` against
    ``1.0.0a1``); what matters is that the compiled module reports its own
    build rather than the Python package metadata, so a stale ``.pyd`` is
    visible in an ``aggregate`` audit record.
    """
    assert aggregate_ntt.__version__
    assert aggregate_ntt.engine_version() == aggregate_ntt.__version__


def test_reconstructor_rejects_bad_primes():
    """Construction validates before any work is queued."""
    with pytest.raises(ValueError, match='must not be empty'):
        aggregate_ntt.Reconstructor([], 4)
    with pytest.raises(ValueError, match='distinct'):
        aggregate_ntt.Reconstructor([17, 17], 4)


def test_reconstructor_reports_its_shape():
    r = aggregate_ntt.Reconstructor([17, 97], 4)
    assert r.n_primes == 2
    assert r.pushed == 0


def test_reconstructor_rejects_a_wrong_length_push():
    """A residue vector of the wrong length is caught, not reinterpreted."""
    r = aggregate_ntt.Reconstructor([17, 97], 4)
    with pytest.raises(ValueError, match='expected 4 coefficients'):
        r.push(np.zeros(5, dtype=np.uint64))


# ------------------------------------------------- [reconstruction kernel]

def _reconstruct(primes, residues):
    """Drive the engine over ``residues``, returning ``(numerators, M)``."""
    n = len(residues[0])
    r = aggregate_ntt.Reconstructor([int(p) for p in primes], n)
    for head in residues:
        r.push(np.asarray(head, dtype=np.uint64))
    limbs, modulus_bytes = r.finish()
    modulus = int.from_bytes(modulus_bytes, 'little')
    numerators = [int.from_bytes(row.tobytes(), 'little') for row in limbs]
    return numerators, modulus


@pytest.mark.xfail(reason='[Rust-Reconstruction] kernel not implemented yet',
                   raises=NotImplementedError, strict=True)
@pytest.mark.parametrize('seed', range(4))
def test_reconstruction_matches_python_crt(seed):
    """The engine reconstructs the integers the Python merge stack does.

    Residues are drawn at random per prime, which is a harder test than a
    real case: a real residue vector is the image of a single integer below
    the modulus, where random residues exercise the full range of the
    reconstruction including coefficients near the modulus.
    """
    ntt = pytest.importorskip('aggregate.ntt')
    ntt_primes = ntt.ntt_primes

    pairs = ntt_primes(2 ** 10, 10 ** 60)
    primes = [p for p, _ in pairs]
    rng = np.random.default_rng(seed)
    residues = [rng.integers(0, p, size=64, dtype=np.uint64) for p in primes]

    numerators, modulus = _reconstruct(primes, residues)

    assert modulus == np.prod([int(p) for p in primes], dtype=object)
    for j, value in enumerate(numerators):
        assert 0 <= value < modulus
        for prime, head in zip(primes, residues):
            assert value % prime == int(head[j])


@pytest.mark.xfail(reason='needs [Rust-Reconstruction] here and the '
                          'ntt_engine keyword from the integration plan in '
                          'aggregate; not strict, since it fails two ways',
                   strict=False)
@pytest.mark.parametrize('seed', range(4))
def test_exact_compound_matches_the_polynomial_oracle(seed):
    """End to end against the oracle that shares no machinery with either.

    ``exact_compound_poly`` multiplies polynomials directly in Python
    integers, with no transform and no CRT, so agreement here is evidence
    about the whole pipeline rather than about one kernel.
    """
    ntt = pytest.importorskip('aggregate.ntt')
    exact_compound = ntt.exact_compound
    exact_compound_poly = ntt.exact_compound_poly

    rng = random.Random(seed)
    m = rng.randint(1, 4)
    outcomes = [rng.randint(0, 8) for _ in range(m)]
    weights = [rng.randint(1, 9) for _ in range(m)]
    total = sum(weights)
    probs = [Fraction(w, total) for w in weights]
    sev_outcomes = [rng.randint(0, 6) for _ in range(rng.randint(1, 4))]
    sev_weights = [rng.randint(1, 9) for _ in sev_outcomes]
    sev_total = sum(sev_weights)
    sev_probs = [Fraction(w, sev_total) for w in sev_weights]

    fast = exact_compound(outcomes, probs, sev_outcomes, sev_probs,
                          ntt_engine='rust')
    slow = exact_compound_poly(outcomes, probs, sev_outcomes, sev_probs)
    assert fast[1] == slow[1]
    assert fast[0] == slow[0]


# ----------------------------------------------------- [transform kernel]

@pytest.mark.xfail(reason='[Rust-Transform] kernel not implemented yet',
                   raises=NotImplementedError, strict=True)
def test_residues_mod_prime_matches_python():
    """One prime's residues, against ``aggregate.ntt._residues_mod_prime``.

    Filled in with the real argument shapes when ``[Rust-Transform]``
    starts; the reference call is private on purpose, since this is the one
    place a private name is legitimately the comparison target.
    """
    raise NotImplementedError(
        '[Rust-Transform] kernel not implemented yet')
