# Migration notes

The package was restructured for readability and speed, and the API was allowed
to change. This page maps the old names onto the new ones.

## Module layout

| Old | New |
| --- | --- |
| `topapprox/bht.py` (`BasinHierarchyTree`) | `topapprox/merge_tree.py` (`MergeTree`) |
| `topapprox/gwf.py` (`GraphWithFaces`) | `topapprox/graph.py` (`GraphWithFaces`) |
| `topapprox/filter_image.py`, `filter_graph.py`, `_filter_base.py`, `persistence_filter.py`, `persistence.py` | `topapprox/filters.py` |
| `topapprox/_backends.py` | `topapprox/backends.py` |
| `topapprox/_grid.py` | `topapprox/grids.py` |
| `topapprox/link_reduce.py` | `topapprox/_reduce_python.py` |
| `topapprox/link_reduce_numba.py` | `topapprox/_reduce_numba.py` |
| `topapprox/link_reduce_cpp.cpp` | `topapprox/_reduce_cpp.cpp` |
| `topapprox/visualize.py` | `topapprox/plotting.py` |
| `topapprox/tools.py` (`Tools.gwf_generator`) | `topapprox.random_graph_with_faces` |

## API changes

1. **`TopologicalFilterImage` / `TopologicalFilterGraph` are gone.**
   Use `ImageFilter` and `GraphFilter`.

2. **`GraphFilter` takes a `GraphWithFaces`.**

   ```python
   # before
   tfg = ta.GraphFilter(method="cpp")
   tfg.compute_gwf(F=faces, H=holes, signal=signal)

   # after
   tfg = ta.GraphFilter.from_faces(faces, holes, signal)
   # or, when you already have the complex:
   tfg = ta.GraphFilter(ta.GraphWithFaces(faces, holes, signal))
   ```

   `GraphFilter.from_array(array)` replaces `TopologicalFilterGraph(array)`.

3. **`GraphWithFaces` uses positional `faces, holes, signal`** and keyword
   `edges=` / `is_triangulated=`. The `compute="normal"|"dual"|"both"` switch is
   gone: `edges` and `dual_edges` are both computed lazily, on demand.
   `E` / `dualE` / `E_signal` / `dualE_signal` are now `edges`, `dual_edges` and
   `dual_values`. `with_signal(new_signal)` reuses the complex for a new signal.

4. **`PersistenceFilter` was replaced by a function.**

   ```python
   # before
   lpf = ta.PersistenceFilter().load_signal(img).low_pers_filter(eps, iteration_order="01")

   # after
   lpf = ta.low_pers_filter(img, eps, order="01")
   ```

   `epsilon` may be a scalar, a `(eps_H0, eps_H1)` pair, or a mapping keyed by
   `0`/`1`. The per-run caching went away; hold on to an `ImageFilter` or
   `GraphFilter` instead — its merge tree is reused across thresholds.

5. **`get_PD_gwf(F, H, signal)` became `persistence_diagram(graph_or_array)`**,
   returning the same `(pd0, pd1)` pair of `(n, 2)` arrays.

6. **`filter.bht` became `filter.tree`**, a `MergeTree`. Renamed methods:

   | Old | New |
   | --- | --- |
   | `get_persistence(reduced=...)` | `diagram(reduced=...)` |
   | `get_diagram()` | `diagrams()` (on `GraphFilter`) |
   | `_update_BHT()` / `get_BHT()` | `filter.tree` |
   | `get_descendants(v)` | `tree.descendants(v)` |
   | `basin_representative_map(eps)` | `tree.basin_representatives(eps)` |
   | `describe_pers()` | `topapprox.DIAGRAM_COLUMNS` |

   The diagram still has the seven columns named by `DIAGRAM_COLUMNS`.

7. **Backend selection is `method="auto"` by default**, and falls back to
   `"python"` with a warning when a named backend is unavailable.
   `available_backends()` no longer takes a `dimensions` argument: all three
   backends now handle 1D, 2D and 3D grids. The dropped arguments `bht_method`,
   `recursive` and `iter_vertex` no longer have any effect and were removed.

8. **Weight names for `depth_distribution` / `height_distribution`** are now
   `"basin_size"`, `"persistence"` and `"basin_size_x_persistence"`, or an array
   of per-pair weights. The parametrised `basin_size_+_persistence(a,b)` string
   form is gone — pass `a * pd[:, 4] + b * (pd[:, 1] - pd[:, 0])` as an array.

## Behaviour changes

- **Merge trees are canonical.** All backends and both routes (grid sweep and
  edge list) now follow one total order — by value, then by vertex index — and
  merge every component meeting at a vertex into the oldest of them. Results no
  longer depend on which kernel ran, and a basin's subtree is exactly the
  component it had just before merging, so `basin_size` is well defined on
  plateaus. Filtered signals are unchanged; individual persistence *pairings*
  may differ from earlier releases where several minima shared a value.
- **`grid_boundary` (was `boundary_from_array_shape`) is fixed.** It used to
  repeat one vertex and emit a wrap-around edge joining the two ends of a row,
  which added a spurious adjacency to `GraphFilter.from_array`.
- **3D works on every backend**, not just C++.
