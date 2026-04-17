# Refactor Migration Memo

This refactor intentionally prioritized a cleaner internal structure over backward compatibility. Old notebooks can still be updated with a small set of mechanical changes.

## API and Behavior Changes

1. `setup.py` is gone.
   Use `pip install .` or `pip install -e ".[dev]"` instead of `python setup.py build_ext --inplace`.

2. The C++ BHT backend was removed.
   `bht_method="cpp"` is no longer a real execution path. The basin hierarchy tree is now always represented in Python, while the heavy link-reduce kernel stays in C++.

3. New preferred class names are `ImageFilter` and `GraphFilter`.
   The old names `TopologicalFilterImage` and `TopologicalFilterGraph` are still exported as aliases.

4. 1D array inputs are handled directly by `ImageFilter`.
   Old notebooks that reshaped 1D signals to `(1, n)` can keep doing that, but they no longer have to.

5. Backend discovery is explicit.
   `topapprox.available_backends()` reports which of `python`, `numba`, and `cpp` are usable in the current environment.

6. `PersistenceFilter` caching is stricter.
   Cached runs now distinguish different thresholds, sizes, and backend settings, so repeated calls with different parameters no longer reuse stale intermediate results.

7. `GraphWithFaces` now uses canonical attributes internally:
   - `edges` instead of relying on `E`
   - `dual_edges` instead of relying on `dualE`
   Compatibility properties `E` and `dualE` are still present.

8. Test and release workflows are now driven by `pyproject.toml`, CMake, and GitHub Actions.
   The old local-only extension build flow is no longer the source of truth.

## Notebook Update Checklist

1. Replace any `setup.py` build cells with editable installs:
   `/opt/homebrew/Caskroom/miniconda/base/bin/python -m pip install -e ".[dev]"`

2. If a notebook used `bht_method="cpp"`, remove that argument.

3. Prefer:
   - `ta.ImageFilter(...)`
   - `ta.GraphFilter(...)`
   The old class names still resolve, but the new names match the refactored API.

4. If a notebook depends on specific backend availability, guard it with:
   `ta.available_backends()`

5. If a notebook relied on undocumented internals under `topapprox.mixins` or the old extension build scripts, update it to the public API. Those internal paths are no longer part of the intended structure.
