# topapprox

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Authors**:
[Matias de Jong van Lier](https://sites.google.com/view/matiasvanlier), [Junyan Chu](https://sites.google.com/view/junyan-chu/), Sebastían Elías Graiff Zurita, [Shizuo Kaji](https://www.skaji.org/)

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

## Quick start

```python
import numpy as np
import topapprox as ta

image = np.random.rand(256, 256)

# One call: remove H0 features, then H1 features.
filtered = ta.low_pers_filter(image, 0.1, order="01")

# Persistence diagrams as (birth, death) arrays.
pd0, pd1 = ta.persistence_diagram(image)
```

Keep the filter object around when you want several thresholds: the merge tree is
built once and every query after that is a handful of array operations.

```python
image_filter = ta.ImageFilter(image)           # H0 (connected components)
mild   = image_filter.low_pers_filter(0.05)
strong = image_filter.low_pers_filter(0.30)
basins = image_filter.basin_map(0.10)          # which basin each pixel belongs to

cycle_filter = ta.ImageFilter(image, dual=True)  # H1 (cycles); H2 for 3D arrays
```

Signals on an embedded graph work the same way. A `GraphWithFaces` carries the
vertex signal together with the 2-cells that are filled (`faces`) and those that
are not (`holes`, including the outer boundary):

```python
faces  = [[0, 1, 2, 3]]
holes  = [[0, 1, 2, 3]]
signal = np.array([0.0, 1.0, 0.5, 0.2])

graph = ta.GraphWithFaces(faces, holes, signal)
pd0, pd1 = ta.persistence_diagram(graph)

graph_filter = ta.GraphFilter(graph)
filtered = graph_filter.low_pers_filter(0.5)
filtered_cycles = graph_filter.with_dual(True).low_pers_filter(0.5)
```

## How it works

| Module | Role |
| --- | --- |
| `topapprox.grids` | grid neighbourhoods and sweep edges |
| `topapprox.graph` | `GraphWithFaces`: the primal and dual filtrations of a graph signal |
| `topapprox.backends` | backend discovery; each kernel builds a merge tree from edges |
| `topapprox.merge_tree` | `MergeTree`: filtering, basins and diagrams, vectorised |
| `topapprox.filters` | `ImageFilter`, `GraphFilter` and the one-shot helpers |
| `topapprox.plotting` | optional matplotlib/networkx helpers |

The kernels return only `parent`, `linking_vertex` and `root`; children, subtree
sizes, depths and persistence pairs are recovered with NumPy. Path queries on the
tree use pointer doubling, so filtering is `O(n log h)` vectorised operations
rather than a Python-level tree walk.

Three kernels are interchangeable and produce identical merge trees, plateaus
included:

```python
ta.available_backends()          # ('python', 'numba', 'cpp'), depending on the install
ta.ImageFilter(image, method="python")   # NumPy reference implementation
```

`method="auto"` (the default) picks the fastest one available.

## Documentation & examples

- [Interactive Tutorial](notebook/Interactive_Tutorial_topapprox.ipynb)
- [BHT Basin Visualization](notebook/BHT_Basin_Visualization.ipynb)
- [Reproducing Paper Examples](notebook/ExamplesInPaper.ipynb)
- [API migration notes](docs/migration.md)
- [Original paper on arXiv](https://arxiv.org/abs/2408.14109)

## Development

```bash
python -m pytest -q
```

Tests that compare against [CubicalRipser](https://github.com/shizuo-kaji/CubicalRipser_3dim)
are skipped when `cripser` is not installed.

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
