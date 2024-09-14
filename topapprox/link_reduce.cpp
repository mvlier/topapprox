#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <tuple>

namespace py = pybind11;

inline int rootfind(int x, std::vector<int>& ancestor) {
    if (ancestor[x] != x) {
        ancestor[x] = rootfind(ancestor[x], ancestor);
    }
    return ancestor[x];
}

std::tuple<std::vector<double>, std::vector<std::tuple<double, double, int>>, std::unordered_map<int, std::vector<int>>, std::vector<int>>
_link_reduce(const std::vector<double>& birth, const std::vector<std::vector<int>>& edges, double epsilon, bool keep_basin = false) {
    std::vector<std::tuple<double, double, int>> persistence;
    // permanent cycle
    persistence.emplace_back(*std::min_element(birth.begin(), birth.end()), std::numeric_limits<double>::infinity(), std::min_element(birth.begin(), birth.end()) - birth.begin());

    std::vector<int> parent(birth.size());
    std::iota(parent.begin(), parent.end(), 0);

    std::vector<int> ancestor(birth.size());
    std::iota(ancestor.begin(), ancestor.end(), 0);

    std::unordered_map<int, std::vector<int>> children;
    std::vector<double> modified = birth;

    for (const auto& edge : edges) {
        int ui = edge[0];
        int vi = edge[1];

        // int up = parent[ui];
        // int vp = parent[vi];

        int up = rootfind(ui,ancestor);
        int vp = rootfind(vi,ancestor);

        double death = std::max(birth[ui], birth[vi]);

        if (up != vp) {
            if (birth[up] < birth[vp]) {
                std::swap(up, vp);
            }

            auto it = children.find(up);
            std::vector<int> region;
            if (it == children.end()) {
                region = {up};
            } else {
                if (!keep_basin) {
                    region = std::move(it->second);
                    children.erase(it);
                }else{
                    region = it->second;
                }
            }

            if (birth[up] < death) {
                persistence.emplace_back(birth[up], death, up);
                if (keep_basin && children.find(up) == children.end()) {
                    children[up] = {up};
                }
                // pushing up the signal values
                if (std::abs(birth[up] - death) < epsilon) {
                    for (int r : region) {
                        modified[r] = death;
                    }
                }
            }

            // set parent
            // for (int r : region) {
            //     parent[r] = vp;
            // }
            parent[up] = vp;
            ancestor[up] = vp;

            // set children
            auto& vp_children = children[vp];
            if (vp_children.empty()) {
                vp_children = {vp};
            }
            vp_children.insert(vp_children.end(), std::make_move_iterator(region.begin()), std::make_move_iterator(region.end()));
        }
    }

    return std::make_tuple(std::move(modified), std::move(persistence), std::move(children), std::move(parent));
}


// Wrapper function
py::tuple link_reduce_wrapper(py::array_t<double> birth, py::array_t<int> edges, double epsilon, bool keep_basin=false) {
    // Convert numpy array to std::vector
    std::vector<double> birth_vec(birth.data(), birth.data() + birth.size());
    
    // Get buffer info of the edges array
    py::buffer_info edges_buf = edges.request();

    // Check if it's a 2D array
    if (edges_buf.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }

    // Get the shape of the edges array
    int num_edges = edges_buf.shape[0];
    int edge_size = edges_buf.shape[1];

    // Directly use a single std::vector<int> to represent the edges
    std::vector<int> edges_vec(static_cast<int*>(edges_buf.ptr), static_cast<int*>(edges_buf.ptr) + num_edges * edge_size);

    // Convert edges_vec to a vector of pairs (i.e., the expected input format)
    std::vector<std::vector<int>> edges_converted;
    edges_converted.reserve(num_edges);
    for (int i = 0; i < num_edges; ++i) {
        edges_converted.emplace_back(edges_vec.begin() + i * edge_size, edges_vec.begin() + (i + 1) * edge_size);
    }

    // Call the C++ function
    auto result = _link_reduce(birth_vec, edges_converted, epsilon, keep_basin);

    // Convert the result back to Python objects
    py::list result_list;
    
    // Convert std::vector<double> to numpy array
    const auto& modified = std::get<0>(result);
    py::array_t<double> modified_array(modified.size(), modified.data());
    result_list.append(modified_array);
    
    result_list.append(std::get<1>(result));
    
    py::dict result_dict;
    for (const auto& pair : std::get<2>(result)) {
        result_dict[py::cast(pair.first)] = py::cast(pair.second);
    }
    result_list.append(result_dict);

    const auto& parent = std::get<3>(result);
    py::array_t<int> parent_array(parent.size(), parent.data());
    result_list.append(parent_array);

    return py::tuple(result_list);
}

PYBIND11_MODULE(link_reduce, m) {
    m.def("link_reduce", &link_reduce_wrapper, "A function that performs link reduction",
          py::arg("birth"), py::arg("edges"), py::arg("epsilon"), py::arg("keep_basin") = false);
}

