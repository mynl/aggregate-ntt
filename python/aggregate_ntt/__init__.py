"""
Compiled engine for exact bounded discrete compound distributions.

An optional accelerator for the :mod:`aggregate` actuarial library. It
computes nothing the library cannot compute alone: it computes the same
integers faster. ``aggregate`` discovers this package at import time and falls
back to its pure-Python path when it is absent, so nothing here is required
for a correct answer.

The engine owns the two hot phases of an exact compound computation, the
per-prime number-theoretic transform and the Chinese remainder
reconstruction. Everything else stays in Python: input validation, clearing
denominators, prime selection, the size guardrails, the audit record, and the
three end-to-end correctness checks, which therefore police this engine's
answer exactly as they police the pure-Python one.

Notes
-----
The seam between the two packages is versioned. :data:`PROTOCOL_VERSION` is
bumped on any incompatible change to the signatures or semantics the compiled
module exposes, and ``aggregate`` refuses a version it does not recognize
rather than computing against a surface that has moved. Two separately
versioned packages drift; the one unacceptable outcome is a silently wrong
answer rather than a loud refusal.

The version is written once, in ``Cargo.toml``, and compiled into the binary.
:func:`engine_version` returns it as Cargo spells it (``1.0.0-alpha.5``) and
:data:`__version__` is the PEP 440 spelling (``1.0.0a5``) that ``pip``, the
wheel metadata and ``aggregate``'s audit record all show. Both come from the
binary rather than from package metadata, so a stale build stays visible.

Design, milestones and measurements: ``dev/plan-rust-engine.md`` in this
repository, and ``dev/plan-ntt-rust-integration.md`` in ``aggregate``, which
states the consuming side.
"""

from ._engine import (
    Reconstructor,
    engine_version,
    protocol_version,
    residues_mod_prime,
)

__all__ = [
    'PROTOCOL_VERSION',
    'Reconstructor',
    '__version__',
    'engine_version',
    'protocol_version',
    'residues_mod_prime',
]


def _pep440(cargo_version):
    """
    Spell a Cargo version the way PEP 440, ``pip`` and the wheel do.

    Parameters
    ----------
    cargo_version : str
        A version as ``Cargo.toml`` writes it, such as ``'1.0.0-alpha.5'``.

    Returns
    -------
    str
        The PEP 440 spelling: ``-alpha.N`` becomes ``aN``, ``-beta.N``
        becomes ``bN`` and ``-rc.N`` becomes ``rcN``, so ``'1.0.0a5'``. A
        release version such as ``'1.0.0'`` passes through unchanged.

    Notes
    -----
    These are the conversions maturin applies when it writes the wheel
    metadata from ``Cargo.toml``, so for any build maturin produced the
    result equals ``importlib.metadata.version('aggregate-ntt')``. Any other
    pre-release form passes through unconverted, and the conformance suite's
    metadata test then fails loudly rather than letting the two drift.
    """
    return (cargo_version.replace('-alpha.', 'a')
            .replace('-beta.', 'b')
            .replace('-rc.', 'rc'))


#: Seam contract version implemented by the compiled module.
PROTOCOL_VERSION = protocol_version()

#: This build's version in PEP 440 spelling, read from the binary itself.
__version__ = _pep440(engine_version())
