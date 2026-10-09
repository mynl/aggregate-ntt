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

#: Seam contract version implemented by the compiled module.
PROTOCOL_VERSION = protocol_version()

__version__ = engine_version()
