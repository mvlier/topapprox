/*
  Native "link and reduce" kernels for topapprox.

  Both kernels build the merge tree (basin hierarchy tree) of a scalar signal by
  sweeping the sub-level-set filtration with a union-find structure.  They return
  the minimal description of that tree -- `parent`, `linking_vertex`, `root` --
  and nothing else: every derived quantity (children, subtree sizes, depths,
  persistence pairs) is recovered with vectorised NumPy code on the Python side.
  Returning plain arrays instead of lists-of-lists is what makes the boundary
  cheap; materialising one Python list per vertex used to dominate the runtime.

  Input arrays are normalised to float64 / int64 and C-contiguous layout by
  topapprox/backends.py, so nanobind binds them zero-copy (unlike pybind11's
  forcecast, nanobind does not silently convert dtypes).
*/

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace {

using Index = std::int64_t;

using BirthArray = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OrderArray = nb::ndarray<const Index, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using EdgeArray = nb::ndarray<const Index, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// Union-find over vertices, carrying the merge tree that the unions induce.
///
/// Every merge happens at one *witness* vertex, the one whose arrival in the
/// sub-level set connects the components.  All components meeting there are
/// attached to the oldest of them, and the witness joins that same component,
/// so a basin's subtree is exactly the component it had just before it merged --
/// independent of the order in which the witness's edges are visited.
class MergeForest {
public:
    explicit MergeForest(std::size_t size, const double* birth)
        : parent(size), linking_vertex(size, -1), root(0), birth_(birth), ancestor_(size) {
        std::iota(parent.begin(), parent.end(), Index{0});
        std::iota(ancestor_.begin(), ancestor_.end(), Index{0});
    }

    /// Iterative find with path halving (no recursion: merge trees can be deep).
    Index find(Index vertex) {
        while (ancestor_[vertex] != vertex) {
            ancestor_[vertex] = ancestor_[ancestor_[vertex]];
            vertex = ancestor_[vertex];
        }
        return vertex;
    }

    /// Order in which the sweep visits vertices: by birth value, then by index.
    bool is_older(Index left, Index right) const {
        return birth_[left] < birth_[right] || (birth_[left] == birth_[right] && left < right);
    }

    /// Attach the component of `dying` to the component rooted at `survivor`.
    void attach(Index dying, Index survivor, Index witness) {
        const Index dying_root = find(dying);
        if (dying_root == survivor) {
            return;
        }
        parent[dying_root] = survivor;
        ancestor_[dying_root] = survivor;
        linking_vertex[dying_root] = witness;
        root = survivor;
    }

    /// Merge `witness` and the components of `neighbours` into their oldest one.
    void merge_at(Index witness, const Index* neighbours, std::size_t count) {
        Index survivor = find(witness);
        for (std::size_t i = 0; i < count; ++i) {
            const Index candidate = find(neighbours[i]);
            if (is_older(candidate, survivor)) {
                survivor = candidate;
            }
        }
        for (std::size_t i = 0; i < count; ++i) {
            attach(neighbours[i], survivor, witness);
        }
        attach(witness, survivor, witness);
    }

    std::vector<Index> parent;
    std::vector<Index> linking_vertex;
    Index root;

private:
    const double* birth_;
    std::vector<Index> ancestor_;
};

/// Move a vector onto the heap and expose its storage as a 1-D NumPy array.
/// The capsule keeps the vector alive until the array is collected.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> as_array(std::vector<T> values) {
    auto* heap = new std::vector<T>(std::move(values));
    nb::capsule owner(heap, [](void* payload) noexcept {
        delete static_cast<std::vector<T>*>(payload);
    });
    const std::size_t size = heap->size();
    // data() may be null for an empty vector; hand nanobind a valid address it
    // will never dereference so that it wraps our storage instead of allocating.
    void* data = size != 0 ? static_cast<void*>(heap->data()) : static_cast<void*>(heap);
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(data, {size}, owner);
}

nb::tuple as_tuple(MergeForest forest) {
    return nb::make_tuple(as_array(std::move(forest.parent)),
                          as_array(std::move(forest.linking_vertex)),
                          forest.root);
}

/// Neighbour offsets of a `Dim`-dimensional grid, as signed steps per axis.
///
/// Primal: the 2 * Dim axis-aligned steps.  Dual: all 3^Dim - 1 non-zero steps.
/// The enumeration matches `topapprox.grids.neighbour_steps`.
template <int Dim>
std::vector<std::array<int, Dim>> neighbour_steps(bool dual) {
    std::vector<std::array<int, Dim>> steps;
    if (!dual) {
        for (int axis = 0; axis < Dim; ++axis) {
            for (int direction : {-1, 1}) {
                std::array<int, Dim> step{};
                step[axis] = direction;
                steps.push_back(step);
            }
        }
        return steps;
    }

    int count = 1;
    for (int axis = 0; axis < Dim; ++axis) {
        count *= 3;
    }
    for (int code = 0; code < count; ++code) {
        std::array<int, Dim> step{};
        int rest = code;
        bool is_zero = true;
        for (int axis = Dim - 1; axis >= 0; --axis) {
            step[axis] = rest % 3 - 1;
            rest /= 3;
            is_zero = is_zero && step[axis] == 0;
        }
        if (!is_zero) {
            steps.push_back(step);
        }
    }
    return steps;
}

/// Sweep a regular grid vertex by vertex in `order` (increasing birth value).
///
/// When `dual` is true the array carries one extra vertex at index `n_cells`,
/// the point at infinity, which every boundary vertex is connected to.  Its
/// birth value is -inf, so it always ends up as the root.
template <int Dim>
MergeForest reduce_grid_impl(const double* birth,
                             const Index* order,
                             Index n_cells,
                             const std::vector<Index>& shape,
                             bool dual) {
    std::array<Index, Dim> extent{};
    std::array<Index, Dim> stride{};
    for (int axis = 0; axis < Dim; ++axis) {
        extent[axis] = shape[static_cast<std::size_t>(axis)];
    }
    stride[Dim - 1] = 1;
    for (int axis = Dim - 2; axis >= 0; --axis) {
        stride[axis] = stride[axis + 1] * extent[axis + 1];
    }

    const auto steps = neighbour_steps<Dim>(dual);
    std::vector<Index> offsets(steps.size());
    for (std::size_t k = 0; k < steps.size(); ++k) {
        Index offset = 0;
        for (int axis = 0; axis < Dim; ++axis) {
            offset += static_cast<Index>(steps[k][axis]) * stride[axis];
        }
        offsets[k] = offset;
    }

    const Index infinity = n_cells;  // index of the point at infinity
    MergeForest forest(static_cast<std::size_t>(dual ? n_cells + 1 : n_cells), birth);

    std::array<Index, Dim> coordinate{};
    std::array<Index, 27> arrived{};  // at most 3^3 - 1 neighbours, plus infinity
    for (Index i = 0; i < n_cells; ++i) {
        const Index vertex = order[i];
        Index rest = vertex;
        bool on_boundary = false;
        for (int axis = 0; axis < Dim; ++axis) {
            coordinate[axis] = rest / stride[axis];
            rest %= stride[axis];
            on_boundary = on_boundary || coordinate[axis] == 0 ||
                          coordinate[axis] == extent[axis] - 1;
        }

        std::size_t count = 0;
        for (std::size_t k = 0; k < steps.size(); ++k) {
            bool inside = true;
            for (int axis = 0; axis < Dim && inside; ++axis) {
                const Index shifted = coordinate[axis] + steps[k][axis];
                inside = shifted >= 0 && shifted < extent[axis];
            }
            if (!inside) {
                continue;
            }
            const Index neighbour = vertex + offsets[k];
            if (birth[neighbour] > birth[vertex]) {
                continue;  // the neighbour joins the sub-level set later
            }
            arrived[count++] = neighbour;
        }
        if (dual && on_boundary) {
            arrived[count++] = infinity;
        }
        if (count != 0) {
            forest.merge_at(vertex, arrived.data(), count);
        }
    }
    return forest;
}

}  // namespace

/// Build the merge tree of a signal given a pre-sorted edge list.
///
/// Row `i` of `edges` is `(passive, active)`: `active` is the endpoint whose
/// arrival creates the edge.  Rows sharing an `active` endpoint must be
/// consecutive -- they are merged as one event -- which is what
/// `topapprox.graph.sweep_order` and `topapprox.grids.sweep_edges` produce.
nb::tuple reduce_edges(const BirthArray& birth, const EdgeArray& edges) {
    if (edges.shape(1) != 2) {
        throw std::invalid_argument("edges must have shape (m, 2).");
    }
    const double* birth_data = birth.data();
    const Index* edge_data = edges.data();
    const std::size_t n_edges = edges.shape(0);

    MergeForest forest(static_cast<std::size_t>(birth.shape(0)), birth_data);
    std::vector<Index> arrived;
    for (std::size_t first = 0; first < n_edges;) {
        const Index witness = edge_data[2 * first + 1];
        std::size_t last = first;
        arrived.clear();
        while (last < n_edges && edge_data[2 * last + 1] == witness) {
            arrived.push_back(edge_data[2 * last]);
            ++last;
        }
        forest.merge_at(witness, arrived.data(), arrived.size());
        first = last;
    }
    return as_tuple(std::move(forest));
}

/// Build the merge tree of a signal sampled on a 1-D, 2-D or 3-D grid.
nb::tuple reduce_grid(const BirthArray& birth,
                      const OrderArray& order,
                      const std::vector<Index>& shape,
                      bool dual) {
    Index n_cells = 1;
    for (const Index extent : shape) {
        n_cells *= extent;
    }
    const Index expected = dual ? n_cells + 1 : n_cells;
    if (static_cast<Index>(birth.shape(0)) != expected) {
        throw std::invalid_argument("birth has " + std::to_string(birth.shape(0)) +
                                    " entries, expected " + std::to_string(expected) + ".");
    }
    if (static_cast<Index>(order.shape(0)) != n_cells) {
        throw std::invalid_argument("order must list every grid cell exactly once.");
    }

    switch (shape.size()) {
        case 1:
            return as_tuple(reduce_grid_impl<1>(birth.data(), order.data(), n_cells, shape, dual));
        case 2:
            return as_tuple(reduce_grid_impl<2>(birth.data(), order.data(), n_cells, shape, dual));
        case 3:
            return as_tuple(reduce_grid_impl<3>(birth.data(), order.data(), n_cells, shape, dual));
        default:
            throw std::invalid_argument("Only 1-D, 2-D and 3-D grids are supported.");
    }
}

NB_MODULE(_reduce_cpp, m) {
    m.attr("__doc__") = "Native link-and-reduce kernels for topapprox.";
    m.def("reduce_edges", &reduce_edges, nb::arg("birth"), nb::arg("edges"));
    m.def("reduce_grid", &reduce_grid, nb::arg("birth"), nb::arg("order"), nb::arg("shape"),
          nb::arg("dual"));
}
