#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace {

using Index = std::int64_t;
using Birth = double;
using Children = std::vector<std::vector<Index>>;

// Input arrays are normalized to the exact dtype / C-contiguous layout on the
// Python side (see topapprox/_backends.py::_cpp_backend), so nanobind can bind
// them zero-copy. nanobind's typed ndarray does not silently upcast dtypes the
// way pybind11's py::array::forcecast did, hence the boundary normalization.
using BirthArray = nb::ndarray<const Birth, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using EdgeArray = nb::ndarray<const Index, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

struct LinkReduceResult {
    std::vector<Index> parent;
    Children children;
    Index root;
    std::vector<Index> linking_vertex;
    Children persistent_children;
    std::vector<Index> positive_pers;
};

Index compute_root(Index vertex, std::vector<Index>& ancestor) {
    if (ancestor[vertex] == vertex) {
        return vertex;
    }
    ancestor[vertex] = compute_root(ancestor[vertex], ancestor);
    return ancestor[vertex];
}

// Move a std::vector onto the heap and expose its storage as a 1D NumPy array.
// A capsule keeps the vector alive until the array is garbage-collected.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> vector_to_array(std::vector<T> values) {
    auto* heap = new std::vector<T>(std::move(values));
    nb::capsule owner(heap, [](void* payload) noexcept {
        delete static_cast<std::vector<T>*>(payload);
    });
    const std::size_t size = heap->size();
    // For an empty vector, data() may be nullptr on some standard libraries.
    // Pass a valid, never-dereferenced address so nanobind always wraps the
    // provided storage instead of self-allocating.
    void* data = size != 0 ? static_cast<void*>(heap->data()) : static_cast<void*>(heap);
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(data, {size}, owner);
}

std::vector<std::array<Index, 2>> copy_edges(const EdgeArray& edges) {
    if (edges.shape(1) != 2) {
        throw std::runtime_error("Edges must have shape (n, 2).");
    }

    const Index* edge_data = edges.data();
    const std::size_t rows = edges.shape(0);
    std::vector<std::array<Index, 2>> copied;
    copied.reserve(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        copied.push_back({edge_data[row * 2 + 0], edge_data[row * 2 + 1]});
    }
    return copied;
}

LinkReduceResult link_reduce_edges(const BirthArray& birth, const EdgeArray& edges) {
    const Birth* birth_data = birth.data();
    const auto copied_edges = copy_edges(edges);
    const auto size = static_cast<std::size_t>(birth.shape(0));

    LinkReduceResult result;
    result.parent = std::vector<Index>(size);
    result.children = Children(size);
    result.root = 0;
    result.linking_vertex = std::vector<Index>(size, -1);
    result.persistent_children = Children(size);
    std::iota(result.parent.begin(), result.parent.end(), 0);
    auto ancestor = result.parent;

    for (const auto& edge : copied_edges) {
        const auto first = edge[0];
        const auto second = edge[1];
        auto first_root = compute_root(first, ancestor);
        auto second_root = compute_root(second, ancestor);
        const auto death = std::max(birth_data[first], birth_data[second]);

        if (first_root == second_root) {
            continue;
        }

        if (birth_data[first_root] < birth_data[second_root]) {
            std::swap(first_root, second_root);
        }

        result.children[second_root].push_back(first_root);
        result.parent[first_root] = second_root;
        ancestor[first_root] = second_root;
        result.root = second_root;
        result.linking_vertex[first_root] = birth_data[first] > birth_data[second] ? first : second;

        if (birth_data[first_root] < death) {
            result.persistent_children[second_root].push_back(first_root);
            result.positive_pers.push_back(first_root);
        }
    }

    return result;
}

std::vector<Index> neighbors_2d(Index vertex, Index n_rows, Index n_cols, bool dual, Index extra_vertex) {
    const auto row = vertex / n_cols;
    const auto col = vertex % n_cols;
    const bool top = row == 0;
    const bool bottom = row == n_rows - 1;
    const bool left = col == 0;
    const bool right = col == n_cols - 1;
    const Index invalid = n_rows * n_cols + (dual ? 1 : 0);

    std::vector<Index> neighbors = {
        left ? invalid : vertex - 1,
        right ? invalid : vertex + 1,
        top ? invalid : vertex - n_cols,
        bottom ? invalid : vertex + n_cols,
    };
    if (!dual) {
        return neighbors;
    }

    neighbors.push_back((top || left) ? extra_vertex : vertex - n_cols - 1);
    neighbors.push_back((bottom || right) ? extra_vertex : vertex + n_cols + 1);
    neighbors.push_back((top || right) ? invalid : vertex - n_cols + 1);
    neighbors.push_back((bottom || left) ? invalid : vertex + n_cols - 1);
    return neighbors;
}

std::vector<Index> neighbors_3d(
    Index vertex,
    Index n_x,
    Index n_y,
    Index n_z,
    bool dual,
    Index extra_vertex
) {
    const Index invalid = n_x * n_y * n_z + (dual ? 1 : 0);
    const auto x = vertex / (n_y * n_z);
    const auto y = (vertex / n_z) % n_y;
    const auto z = vertex % n_z;

    const std::array<bool, 6> on_boundary = {
        x == 0,
        x == n_x - 1,
        y == 0,
        y == n_y - 1,
        z == 0,
        z == n_z - 1,
    };
    const std::array<Index, 6> delta = {
        -n_y * n_z,
        n_y * n_z,
        -n_z,
        n_z,
        -1,
        1,
    };

    std::vector<Index> neighbors;
    neighbors.reserve(dual ? 27 : 6);
    for (std::size_t axis = 0; axis < on_boundary.size(); ++axis) {
        neighbors.push_back(on_boundary[axis] ? invalid : vertex + delta[axis]);
    }

    if (!dual) {
        return neighbors;
    }

    for (std::size_t first = 0; first < on_boundary.size(); ++first) {
        for (std::size_t second = first + 1 + ((first + 1) % 2); second < on_boundary.size(); ++second) {
            neighbors.push_back((on_boundary[first] || on_boundary[second]) ? invalid : vertex + delta[first] + delta[second]);
            for (std::size_t third = second + 1 + ((second + 1) % 2); third < on_boundary.size(); ++third) {
                neighbors.push_back(
                    (on_boundary[first] || on_boundary[second] || on_boundary[third])
                        ? invalid
                        : vertex + delta[first] + delta[second] + delta[third]
                );
            }
        }
    }

    const bool touches_boundary = std::any_of(on_boundary.begin(), on_boundary.end(), [](bool value) { return value; });
    neighbors.push_back(touches_boundary ? extra_vertex : invalid);
    return neighbors;
}

template <typename NeighborFn>
LinkReduceResult link_reduce_grid(const BirthArray& birth, Index size, NeighborFn&& compute_neighbors) {
    const Birth* birth_data = birth.data();

    std::vector<Index> ordered_vertices(static_cast<std::size_t>(size));
    std::iota(ordered_vertices.begin(), ordered_vertices.end(), 0);
    std::sort(ordered_vertices.begin(), ordered_vertices.end(), [&](Index left, Index right) {
        return birth_data[left] < birth_data[right];
    });

    LinkReduceResult result;
    result.parent = std::vector<Index>(static_cast<std::size_t>(size));
    result.children = Children(static_cast<std::size_t>(size));
    result.root = 0;
    result.linking_vertex = std::vector<Index>(static_cast<std::size_t>(size), -1);
    result.persistent_children = Children(static_cast<std::size_t>(size));
    std::iota(result.parent.begin(), result.parent.end(), 0);
    auto ancestor = result.parent;

    for (const auto vertex : ordered_vertices) {
        for (const auto neighbor : compute_neighbors(vertex)) {
            if (neighbor >= size) {
                continue;
            }
            if (birth_data[neighbor] > birth_data[vertex]) {
                continue;
            }

            auto neighbor_root = compute_root(neighbor, ancestor);
            auto vertex_root = compute_root(vertex, ancestor);
            if (neighbor_root == vertex_root) {
                continue;
            }

            if (birth_data[neighbor_root] < birth_data[vertex_root]) {
                std::swap(neighbor_root, vertex_root);
            }

            result.children[vertex_root].push_back(neighbor_root);
            result.parent[neighbor_root] = vertex_root;
            ancestor[neighbor_root] = vertex_root;
            result.root = vertex_root;
            result.linking_vertex[neighbor_root] = vertex;

            if (birth_data[neighbor_root] < birth_data[vertex]) {
                result.persistent_children[vertex_root].push_back(neighbor_root);
                result.positive_pers.push_back(neighbor_root);
            }
        }
    }

    return result;
}

nb::tuple to_python_tuple(LinkReduceResult result) {
    return nb::make_tuple(
        vector_to_array(std::move(result.parent)),
        std::move(result.children),
        result.root,
        vector_to_array(std::move(result.linking_vertex)),
        std::move(result.persistent_children),
        vector_to_array(std::move(result.positive_pers))
    );
}

}  // namespace

nb::tuple link_reduce_edges_py(
    const BirthArray& birth,
    const EdgeArray& edges,
    Birth epsilon,
    bool keep_basin
) {
    (void)epsilon;
    (void)keep_basin;
    return to_python_tuple(link_reduce_edges(birth, edges));
}

nb::tuple link_reduce_vertices_2d_py(const BirthArray& birth, std::pair<Index, Index> shape, bool dual) {
    const auto size = static_cast<Index>(birth.size());
    const auto extra_vertex = size - 1;
    auto result = link_reduce_grid(
        birth,
        size,
        [&](Index vertex) { return neighbors_2d(vertex, shape.first, shape.second, dual, extra_vertex); }
    );
    return to_python_tuple(std::move(result));
}

nb::tuple link_reduce_vertices_3d_py(const BirthArray& birth, std::tuple<Index, Index, Index> shape, bool dual) {
    const auto size = static_cast<Index>(birth.size());
    const auto extra_vertex = size - 1;
    auto result = link_reduce_grid(
        birth,
        size,
        [&](Index vertex) {
            return neighbors_3d(
                vertex,
                std::get<0>(shape),
                std::get<1>(shape),
                std::get<2>(shape),
                dual,
                extra_vertex
            );
        }
    );
    return to_python_tuple(std::move(result));
}

NB_MODULE(link_reduce_cpp, m) {
    m.attr("__doc__") = "Native link-reduce kernels for topapprox.";

    m.def(
        "_link_reduce_cpp",
        &link_reduce_edges_py,
        nb::arg("birth"),
        nb::arg("edges"),
        nb::arg("epsilon") = 0.0,
        nb::arg("keep_basin") = false
    );
    m.def("_link_reduce_vertices_cpp", &link_reduce_vertices_2d_py, nb::arg("birth"), nb::arg("shape"), nb::arg("dual"));
    m.def("_link_reduce_vertices_cpp_3D", &link_reduce_vertices_3d_py, nb::arg("birth"), nb::arg("shape"), nb::arg("dual"));
}
