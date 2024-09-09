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

std::tuple<std::vector<double>, std::vector<std::tuple<double, double, int>>, std::unordered_map<int, std::vector<int>>>
_link_reduce(const std::vector<double>& birth, const std::vector<std::vector<int>>& edges, double epsilon, bool keep_basin=false){
    std::vector<std::tuple<double, double, int>> persistence;
    persistence.push_back(std::make_tuple(*min_element(birth.begin(), birth.end()), std::numeric_limits<double>::infinity(), std::min_element(birth.begin(), birth.end()) - birth.begin()));

    std::vector<int> parent(birth.size());
    std::iota(parent.begin(), parent.end(), 0);

    std::unordered_map<int, std::vector<int>> children;
    std::vector<double> modified = birth;

    for (size_t i = 0; i < edges.size(); ++i) {
        int ui = edges[i][0];
        int vi = edges[i][1];

        int up = parent[ui];
        int vp = parent[vi];

        double death = std::max(birth[ui], birth[vi]);

        if (up != vp) {
            if (birth[up] < birth[vp]) {
                std::swap(up, vp);
            }

            std::vector<int> region;
            if (children.find(up) == children.end()) {
                region = {up};
            } else {
                region = children[up];
                if (!keep_basin) {
                    children.erase(up);
                }
            }

            if (birth[up] < death) {
                persistence.push_back(std::make_tuple(birth[up], death, up));
                if (keep_basin && children.find(up) == children.end()) {
                    children[up] = {up};
                }
                if (std::abs(birth[up] - death) < epsilon) {
                    for (int r : region) {
                        modified[r] = death;
                    }
                }
            }

            for (int r : region) {
                parent[r] = vp;
            }

            if (children.find(vp) != children.end()) {
                children[vp].insert(children[vp].end(), region.begin(), region.end());
            } else {
                children[vp] = {vp};
                children[vp].insert(children[vp].end(), region.begin(), region.end());
            }
        }
    }

    return std::make_tuple(modified, persistence, children);
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
    
    // Convert 2D numpy array to std::vector<std::vector<int>>
    std::vector<std::vector<int>> edges_vec(num_edges);
    int* edges_ptr = static_cast<int*>(edges_buf.ptr);
    
    for (int i = 0; i < num_edges; ++i) {
        edges_vec[i].reserve(edge_size);
        for (int j = 0; j < edge_size; ++j) {
            edges_vec[i].push_back(edges_ptr[i * edge_size + j]);
        }
    }
        
    // Call the C++ function
    auto result = _link_reduce(birth_vec, edges_vec, epsilon, keep_basin);
    
    // Convert the result back to Python objects
    py::list result_list;
    // Convert std::vector<double> to numpy array
    const auto& birth_result = std::get<0>(result);
    py::array_t<double> birth_array(birth_result.size(), birth_result.data());
    result_list.append(birth_array);
    result_list.append(std::get<1>(result));
    
    py::dict result_dict;
    for (const auto& pair : std::get<2>(result)) {
        result_dict[py::cast(pair.first)] = pair.second;
    }
    result_list.append(result_dict);
    
    return py::cast<py::tuple>(result_list);
}

PYBIND11_MODULE(link_reduce, m) {
    m.def("link_reduce", &link_reduce_wrapper, "A function that performs link reduction",
          py::arg("birth"), py::arg("edges"), py::arg("epsilon"), py::arg("keep_basin") = false);
}

