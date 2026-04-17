# topapprox

`topapprox` is a Python package for persistent-homology-based filtering of scalar signals on:

- 1D, 2D, and 3D arrays
- graphs with faces

Low-persistence filtering removes topological features that are small relative to a chosen threshold `epsilon`.

In practice:

1. Build the persistence diagram of the input signal.
2. Identify features whose persistence is smaller than `epsilon`.
3. Modify the signal so those low-persistence features disappear, while the more significant ones remain.

This makes the filter useful for denoising and simplification: small oscillations, shallow basins, and weak cycles are removed, while large-scale structure is preserved. For array signals, the filtered output remains within `epsilon` in the `L^inf` norm of the original input.

## Installation

From PyPI:

```bash
pip install topapprox
```

From source:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import topapprox as ta
from topapprox.persistence import get_PD_gwf

img = np.array([[0, 5, 3], [5, 6, 4], [2, 5, 1]], dtype=float)

image_filter = ta.ImageFilter(img, method="python")
filtered_h0 = image_filter.low_pers_filter(1.5)

dual_filter = ta.ImageFilter(img, dual=True, method="python")
filtered_h1 = dual_filter.low_pers_filter(1.5)

faces = [[0, 1, 2, 3]]
holes = [[0, 1, 2, 3]]
signal = np.array([0.0, 1.0, 0.5, 0.2])

graph_filter = ta.GraphFilter(method="python")
graph_filter.compute_gwf(F=faces, H=holes, signal=signal)
pd0, pd1 = graph_filter.get_diagram()

persistence_filter = ta.PersistenceFilter().load_signal(img)
filtered_01 = persistence_filter.low_pers_filter(1.5, iteration_order="01", method="python")

pd0_fn, pd1_fn = get_PD_gwf(faces, holes, signal, method="python")
```

## Documentation & Examples

- [Interactive Tutorial](notebook/Interactive_Tutorial_topapprox.ipynb)
- [Topological Filtering Walkthrough](notebook/Topological_Filtering.ipynb)
- [BHT Basin Visualization](notebook/BHT_Basin_Visualization.ipynb)
- [Reproducing Paper Examples](notebook/Reproducing_paper_examples.ipynb)
- [Notebook/API migration memo](docs/migration.md)
- [Original paper on arXiv](https://arxiv.org/abs/2408.14109)

## Development

Run the full test suite:

```bash
python -m pytest -q
```

Build distributions locally:

```bash
python -m build --wheel --sdist
```

Binary wheels are built with `cibuildwheel`, and the native extension is compiled through `scikit-build-core` + CMake.

## Citation

If you use this package in your work, please cite:

> Matias de Jong van Lier, Sebastían Elías Graiff Zurita, Shizuo Kaji.
> Topological filtering of a signal over a network (2024).
> [arXiv:2408.14109](https://arxiv.org/abs/2408.14109)

```bibtex
@article{vanlier2024topological,
  title={Topological filtering of a signal over a network},
  author={de Jong van Lier, Matias and Graiff Zurita, Sebastían Elías and Kaji, Shizuo},
  journal={arXiv preprint arXiv:2408.14109},
  year={2024}
}
```

## Migration Notes

The refactor intentionally simplified the public surface and build system. If you are adapting older notebooks, see [docs/migration.md](docs/migration.md).
