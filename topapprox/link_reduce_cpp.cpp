#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

using Index = std::int64_t;
using Birth = double;
using Children = std::vector<std::vector<Index>>;

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

template <typename T>
py::array_t<T> vector_to_array(const std::vector<T>& values) {
    py::array_t<T> array(values.size());
    auto mutable_view = array.template mutable_unchecked<1>();
    for (py::ssize_t index = 0; index < mutable_view.shape(0); ++index) {
        mutable_view(index) = values[static_cast<std::size_t>(index)];
    }
    return array;
}

std::vector<std::array<Index, 2>> copy_edges(
    const py::array_t<Index, py::array::c_style | py::array::forcecast>& edges
) {
    auto view = edges.unchecked<2>();
    if (view.shape(1) != 2) {
        throw std::runtime_error("Edges must have shape (n, 2).");
    }

    std::vector<std::array<Index, 2>> copied;
    copied.reserve(static_cast<std::size_t>(view.shape(0)));
    for (py::ssize_t row = 0; row < view.shape(0); ++row) {
        copied.push_back({view(row, 0), view(row, 1)});
    }
    return copied;
}

LinkReduceResult link_reduce_edges(
    const py::array_t<Birth, py::array::c_style | py::array::forcecast>& birth,
    const py::array_t<Index, py::array::c_style | py::array::forcecast>& edges
) {
    auto birth_view = birth.unchecked<1>();
    const auto copied_edges = copy_edges(edges);
    const auto size = static_cast<std::size_t>(birth_view.shape(0));

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
        const auto death = std::max(birth_view(first), birth_view(second));

        if (first_root == second_root) {
            continue;
        }

        if (birth_view(first_root) < birth_view(second_root)) {
            std::swap(first_root, second_root);
        }

        result.children[second_root].push_back(first_root);
        result.parent[first_root] = second_root;
        ancestor[first_root] = second_root;
        result.root = second_root;
        result.linking_vertex[first_root] = birth_view(first) > birth_view(second) ? first : second;

        if (birth_view(first_root) < death) {
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
LinkReduceResult link_reduce_grid(
    const py::array_t<Birth, py::array::c_style | py::array::forcecast>& birth,
    Index size,
    NeighborFn&& compute_neighbors
) {
    auto birth_view = birth.unchecked<1>();

    std::vector<Index> ordered_vertices(size);
    std::iota(ordered_vertices.begin(), ordered_vertices.end(), 0);
    std::sort(ordered_vertices.begin(), ordered_vertices.end(), [&](Index left, Index right) {
        return birth_view(left) < birth_view(right);
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
            if (birth_view(neighbor) > birth_view(vertex)) {
                continue;
            }

            auto neighbor_root = compute_root(neighbor, ancestor);
            auto vertex_root = compute_root(vertex, ancestor);
            if (neighbor_root == vertex_root) {
                continue;
            }

            if (birth_view(neighbor_root) < birth_view(vertex_root)) {
                std::swap(neighbor_root, vertex_root);
            }

            result.children[vertex_root].push_back(neighbor_root);
            result.parent[neighbor_root] = vertex_root;
            ancestor[neighbor_root] = vertex_root;
            result.root = vertex_root;
            result.linking_vertex[neighbor_root] = vertex;

            if (birth_view(neighbor_root) < birth_view(vertex)) {
                result.persistent_children[vertex_root].push_back(neighbor_root);
                result.positive_pers.push_back(neighbor_root);
            }
        }
    }

    return result;
}

py::tuple to_python_tuple(const LinkReduceResult& result) {
    return py::make_tuple(
        vector_to_array(result.parent),
        py::cast(result.children),
        result.root,
        vector_to_array(result.linking_vertex),
        py::cast(result.persistent_children),
        vector_to_array(result.positive_pers)
    );
}

}  // namespace

py::tuple link_reduce_edges_py(
    const py::array_t<Birth, py::array::c_style | py::array::forcecast>& birth,
    const py::array_t<Index, py::array::c_style | py::array::forcecast>& edges,
    Birth epsilon,
    bool keep_basin
) {
    (void)epsilon;
    (void)keep_basin;
    return to_python_tuple(link_reduce_edges(birth, edges));
}

py::tuple link_reduce_vertices_2d_py(
    const py::array_t<Birth, py::array::c_style | py::array::forcecast>& birth,
    std::pair<Index, Index> shape,
    bool dual
) {
    const auto size = static_cast<Index>(birth.size());
    const auto extra_vertex = size - 1;
    auto result = link_reduce_grid(
        birth,
        size,
        [&](Index vertex) { return neighbors_2d(vertex, shape.first, shape.second, dual, extra_vertex); }
    );
    return to_python_tuple(result);
}

py::tuple link_reduce_vertices_3d_py(
    const py::array_t<Birth, py::array::c_style | py::array::forcecast>& birth,
    std::tuple<Index, Index, Index> shape,
    bool dual
) {
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
    return to_python_tuple(result);
}

PYBIND11_MODULE(link_reduce_cpp, module) {
    module.doc() = "Native link-reduce kernels for topapprox.";

    module.def(
        "_link_reduce_cpp",
        &link_reduce_edges_py,
        py::arg("birth"),
        py::arg("edges"),
        py::arg("epsilon") = 0.0,
        py::arg("keep_basin") = false
    );
    module.def("_link_reduce_vertices_cpp", &link_reduce_vertices_2d_py, py::arg("birth"), py::arg("shape"), py::arg("dual"));
    module.def("_link_reduce_vertices_cpp_3D", &link_reduce_vertices_3d_py, py::arg("birth"), py::arg("shape"), py::arg("dual"));
}
