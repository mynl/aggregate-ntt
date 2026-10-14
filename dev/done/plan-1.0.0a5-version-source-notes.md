# Execution notes: [version-source], the [ntt-engine] step

Companion to `plan-1.0.0a5-version-source.md`, kept as a separate file so the
plan itself stays byte identical with its three sibling copies (canonical in
`aggregate-api`). Executed 2026-10-14 from `1.0.0a4`.

## Divergences

- **[order-of-work]** The plan says the steps run in the order given, and
  neither `[library]` (aggregate 1.2.1) nor `[relativity-plugin]`
  (relativity 0.10.0) had landed. The `[ntt-engine]` step depends on neither,
  and the author asked for it explicitly, so it ran third in the list but
  first in time. Only `[api]` reads this step's result.
- **[uv-no-sync]** Every `uv run` here was `uv run --no-sync`, so the
  author's untracked `uv.lock` was never rewritten and the editable
  `aggregate` worktree install was never disturbed.
- **[cargo-lock]** `Cargo.lock` records the crate's own version, so it
  changes with every bump. It is derived, not a second written source, and
  rides in the bump commit.
- **[stale-build-test]** `test_engine_version_matches_cargo` claimed to
  check Cargo but compared `engine_version()` with `__version__`, which this
  step makes deliberately different. It now reads `Cargo.toml` and compares,
  which is the stale-binary tripwire the plan's reasoning relies on. Two new
  tests cover the plan's asks: `__version__` against the PEP 440 spelling of
  `engine_version()`, and against `importlib.metadata`. A parametrized test
  pins the spelling conversion itself.
- **[fresh-build-check]** The plan's "in a fresh build" check ran as a
  release wheel (`maturin build --release`, no `target-cpu`) installed into a
  clean scratch venv, with the seam tests run there against the wheel.
- **[then-1.0.0]** The author asked, in the same request, to mint 1.0.0
  once this step landed. That is a separate bump and commit after
  `1.0.0a5`, not part of this plan, so the end-to-end acceptance line
  "`aggregate-ntt` shows `1.0.0a5` when installed" will read `1.0.0` in
  practice.
