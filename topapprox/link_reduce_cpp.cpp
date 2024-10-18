#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Function to compute the root/ancestor
int compute_root(int v, std::vector<int>& ancestor) {
    if (ancestor[v] == v) {
        return v;
    }
    ancestor[v] = compute_root(ancestor[v], ancestor);
    return ancestor[v];
}

// // Function to compute all descendants of a node
// void descendants(int v, const std::vector<std::vector<int>>& children, std::vector<int>& desc) {
//     desc.push_back(v);  // Add the current node to the descendants
//     for (int child : children[v]) {  // Recursively find the descendants
//         descendants(child, children, desc);
//     }
// }

// // Function to compute the descendants of a node
// std::vector<int> compute_descendants(int v, const std::vector<std::vector<int>>& children) {
//     std::vector<int> desc;
//     descendants(v, children, desc);
//     return desc;
// }

// The main link and reduce function
std::tuple<std::vector<int>, std::vector<std::vector<int>>, int, std::vector<int>, std::vector<std::vector<int>>, std::vector<int>>
_link_reduce_cpp(const std::vector<double>& birth, const std::vector<std::vector<int>>& edges, double epsilon, bool keep_basin = false) {
    
    std::vector<int> parent(birth.size());
    std::iota(parent.begin(), parent.end(), 0);  // Initialize the parent array
    std::vector<int> ancestor(birth.size());
    std::iota(ancestor.begin(), ancestor.end(), 0);  // Initialize the ancestor array
    std::vector<int> linking_vertex(birth.size(), -1);
    int root = 0;
    int n_pers = 1; // number of persistent intervals (starting at 1 to count the permanent cycle)

    std::vector<std::vector<int>> children(birth.size());  // Initialize the children array
    std::vector<std::vector<int>> persistent_children(birth.size());  // Track children with non-zero persistence
    // std::vector<double> modified = birth;  // Copy the birth array
    std::vector<int> positive_pers; // List to store the positive persistence vertices apart from root

    for (const auto& edge : edges) {
        int ui = edge[0];
        int vi = edge[1];
        // Find parents of u and v
        int up = compute_root(ui, ancestor);
        int vp = compute_root(vi, ancestor);
        double death = std::max(birth[ui], birth[vi]);

        if (up != vp) {  // If their connected components are different
            if (birth[up] < birth[vp]) {
                std::swap(up, vp);  // up is the younger and to be killed
            }

            // Tree structure
            children[vp].push_back(up);
            parent[up] = vp;
            ancestor[up] = vp;
            root = vp;  // By the end of the loop, root will store the only vertex that is its own parent

            linking_vertex[up] = (birth[ui] > birth[vi]) ? ui : vi;

            if (birth[up] < death) {  // A cycle is produced
                persistent_children[vp].push_back(up);
                positive_pers.push_back(up);
            }
        }
    }

    return std::make_tuple(parent, children, root, linking_vertex, persistent_children, positive_pers);
}

// Wrapper function to interface with Python
py::tuple link_reduce_wrapper(py::array_t<double> birth, py::array_t<int> edges, double epsilon, bool keep_basin = false) {
    // Convert numpy array to std::vector
    std::vector<double> birth_vec(birth.data(), birth.data() + birth.size());

    // Get buffer info of the edges array
    py::buffer_info edges_buf = edges.request();

    // Check if it's a 2D array
    if (edges_buf.ndim != 2 || edges_buf.shape[1] != 2) {
        throw std::runtime_error("Edges must be a 2D array with two columns");
    }

    // Convert numpy array to std::vector<std::vector<int>>
    std::vector<std::vector<int>> edges_converted(edges_buf.shape[0], std::vector<int>(2));
    int* edges_ptr = static_cast<int*>(edges_buf.ptr);
    for (size_t i = 0; i < edges_converted.size(); ++i) {
        edges_converted[i][0] = edges_ptr[i * 2];
        edges_converted[i][1] = edges_ptr[i * 2 + 1];
    }

    // Call the C++ function
    auto result = _link_reduce_cpp(birth_vec, edges_converted, epsilon, keep_basin);

    // Convert the result back to Python objects
    py::list result_list;

    // // Convert std::vector<double> to numpy array
    // const auto& modified = std::get<0>(result);
    // py::array_t<double> modified_array(modified.size(), modified.data());
    // result_list.append(modified_array);

    // Convert std::vector<int> parent to numpy array
    const auto& parent = std::get<0>(result);
    py::array_t<int> parent_array(parent.size(), parent.data());
    result_list.append(parent_array);

    // Convert children (vector of vectors) to Python list
    const auto& children = std::get<1>(result);
    py::list children_list;
    for (const auto& child_vec : children) {
        children_list.append(py::cast(child_vec));
    }
    result_list.append(children_list);

    // Append root, linking_vertex, and persistent_children
    result_list.append(py::cast(std::get<2>(result)));  // root
    result_list.append(py::array_t<int>(std::get<3>(result).size(), std::get<3>(result).data()));  // linking_vertex
    py::list persistent_children_list;
    for (const auto& pchild_vec : std::get<4>(result)) {
        persistent_children_list.append(py::cast(pchild_vec));
    }
    result_list.append(persistent_children_list);
    result_list.append(py::array_t<int>(std::get<5>(result).size(), std::get<5>(result).data()));  // positive_pers


    return py::tuple(result_list);
}

// Define module and expose the functions
PYBIND11_MODULE(link_reduce_cpp, m) {
    m.def("_link_reduce_cpp", &link_reduce_wrapper, "A function that performs link reduction",
          py::arg("birth"), py::arg("edges"), py::arg("epsilon"), py::arg("keep_basin") = false);

    // Expose the compute_root function
    m.def("compute_root_cpp", [](int v, py::array_t<int> ancestor) {
        std::vector<int> ancestor_vec(ancestor.data(), ancestor.data() + ancestor.size());
        return compute_root(v, ancestor_vec);
    }, "Compute the root/ancestor of a node");

    // // Expose the compute_descendants function
    // m.def("compute_descendants_cpp", [](int v, py::list children) {
    //     std::vector<std::vector<int>> children_vec;
    //     for (auto item : children) {
    //         children_vec.push_back(item.cast<std::vector<int>>());
    //     }
    //     return compute_descendants(v, children_vec);
    // }, "Compute all descendants of a node");
}
