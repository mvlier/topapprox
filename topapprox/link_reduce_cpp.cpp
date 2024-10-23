#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>  // For debugging

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
    // int n_pers = 1; // number of persistent intervals (starting at 1 to count the permanent cycle)

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

// computes neighbor vertices of a vertex v in an image
// std::vector<int> neighbors(int v, std::pair<int, int> shape) {
//     int n = shape.first;
//     int m = shape.second;
//     std::vector<int> nbs(4, -1);
    
//     nbs[0] = (v % m == 0) ? -1 : v - 1;             // Left neighbor
//     nbs[1] = (v % m == m - 1) ? -1 : v + 1;         // Right neighbor
//     nbs[2] = (v < m) ? -1 : v - m;                  // Upper neighbor
//     nbs[3] = (v > (n - 1) * m - 1) ? -1 : v + m;    // Lower neighbor
    
//     return nbs;
// }

void neighbors(int v, std::pair<int, int> shape, std::vector<int>& nbs, bool dual, int& size, int& extra_vertex) {
    int n = shape.first;
    int m = shape.second;
    int row = v / m;
    int col = v % m;
    bool top = (row==0), bottom = (row==n-1); 
    bool left = (col==0), right = (col==m-1);
    
    // Update the values in the preallocated nbs vector
    nbs[0] = left ? size : v - 1;          // Left neighbor
    nbs[1] = right ? size : v + 1;         // Right neighbor
    nbs[2] = top ? size : v - m;           // Upper neighbor
    nbs[3] = bottom ? size : v + m;        // Lower neighbor

    if (dual) {
        nbs[4] = top || left ? extra_vertex : v - m - 1;
        nbs[5] = bottom || right ? extra_vertex : v + m + 1;
        nbs[6] = top || right ? size : v - m + 1;
        nbs[7] = bottom || left ? size : v + m - 1;
    }
}


void neighbors_3D(int v, std::tuple<int, int, int> shape, std::vector<int>& nbs, bool dual, int& size, int& extra_vertex) {
    int n = std::get<0>(shape);
    int m = std::get<1>(shape);
    int l = std::get<2>(shape);
    int i = v / (m * l);
    int j = (v / l) % m;
    int k = v % l;
    std::vector<bool> condition = {i==0, i==n-1, j==0, j==m-1, k==0, k==l-1};
    std::vector<int> increment = {-1, 1, -m, m, -m*l, m*l};

    int idx;
    for (idx=0; idx<6; ++idx){
        nbs[idx] = condition[idx] ? size : v + increment[idx];
    }

    if (dual) {
        for (int idx2=0; idx2<6; ++idx2){
            for (int idx3 = idx2 + 1 + (idx2+1)%2; idx3<6; ++idx3){
                nbs[idx] = (condition[idx2] || condition[idx3]) ? size : v + increment[idx2] + increment[idx3];
                ++idx;
                for (int idx4 = idx3 + 1 + (idx3+1)%2; idx4<6; ++idx4){
                    nbs[idx] = (condition[idx2] || condition[idx3] || condition[idx4]) ? size : v + increment[idx2] + increment[idx3] + increment[idx4];
                    ++idx;
                }
            }
        }
        nbs[idx] = std::any_of(condition.begin(), condition.end(), [](bool val) { return val; }) ? extra_vertex : size;
    }
}



// Alternative version of link and reduce function iterating over vertices
py::tuple _link_reduce_vertices_cpp(py::array_t<float> birth, std::pair<int, int> shape, bool dual) {
    auto birth_unchecked = birth.unchecked<1>();  // Access elements as float
    int size = birth.size();
    int extra_vertex = size - 1; // only used if dual==True

    std::vector<int> vertices_ordered(size);
    std::iota(vertices_ordered.begin(), vertices_ordered.end(), 0);
    std::sort(vertices_ordered.begin(), vertices_ordered.end(), [&](int a, int b) {
        return birth_unchecked(a) < birth_unchecked(b);
    });

    // std::vector<int> vertex_birth_index(size);
    // for (int i = 0; i < size; ++i) {
    //     vertex_birth_index[vertices_ordered[i]] = i;
    // }

    std::vector<int> parent(size);
    std::iota(parent.begin(), parent.end(), 0);  // Parent initialized to identity
    std::vector<int> ancestor = parent;
    std::vector<int> linking_vertex(size, -1);
    int root = 0;
    std::vector<int> positive_pers;
    std::vector<std::vector<int>> children(size);
    std::vector<std::vector<int>> persistent_children(size);

    int n_neighbors = 4;
    if (dual) {
        n_neighbors = 8;
    }
    
    std::vector<int> nbs(n_neighbors, -1);

    for (const auto& v : vertices_ordered) {
        neighbors(v, shape, nbs, dual, size, extra_vertex);
        for (int u : nbs) {
            if (u < size && birth_unchecked(u) <= birth_unchecked(v)) {
                int up = compute_root(u, ancestor);
                int vp = compute_root(v, ancestor);

                if (up != vp) {
                    if (birth_unchecked(up) < birth_unchecked(vp)) {
                        std::swap(up, vp);
                    }

                    children[vp].push_back(up);
                    parent[up] = vp;  // Update the parent
                    ancestor[up] = vp;  // Update the ancestor
                    root = vp;  // Set root to vp
                    linking_vertex[up] = v;  // Track linking vertex

                    
                    if (birth_unchecked(up) < birth_unchecked(v)) {
                        persistent_children[vp].push_back(up);
                        positive_pers.push_back(up);
                    }
                }
            }
        }
    }

    py::list children_list;
    for (const auto& child_vec : children) {
        py::list child_pylist;
        for (const auto& c : child_vec) {
            child_pylist.append(c);
        }
        children_list.append(child_pylist);
    }

    py::list persistent_children_list;
    for (const auto& p_child_vec : persistent_children) {
        py::list p_child_pylist;
        for (const auto& pc : p_child_vec) {
            p_child_pylist.append(pc);
        }
        persistent_children_list.append(p_child_pylist);
    }

    return py::make_tuple(
        py::array_t<int>(parent.size(), parent.data()),
        children_list,
        root,
        py::array_t<int>(linking_vertex.size(), linking_vertex.data()),
        persistent_children_list,
        py::array_t<int>(positive_pers.size(), positive_pers.data())
    );
}









// 3D version of link and reduce function iterating over vertices
py::tuple _link_reduce_vertices_cpp_3D(py::array_t<float> birth, std::tuple<int, int, int> shape, bool dual) {
    auto birth_unchecked = birth.unchecked<1>();  // Access elements as float
    int size = birth.size();
    int extra_vertex = size - 1; // only used if dual==True

    std::vector<int> vertices_ordered(size);
    std::iota(vertices_ordered.begin(), vertices_ordered.end(), 0);
    std::sort(vertices_ordered.begin(), vertices_ordered.end(), [&](int a, int b) {
        return birth_unchecked(a) < birth_unchecked(b);
    });

    // std::vector<int> vertex_birth_index(size);
    // for (int i = 0; i < size; ++i) {
    //     vertex_birth_index[vertices_ordered[i]] = i;
    // }

    std::vector<int> parent(size);
    std::iota(parent.begin(), parent.end(), 0);  // Parent initialized to identity
    std::vector<int> ancestor = parent;
    std::vector<int> linking_vertex(size, -1);
    int root = 0;
    std::vector<int> positive_pers;
    std::vector<std::vector<int>> children(size);
    std::vector<std::vector<int>> persistent_children(size);

    int n_neighbors = 6;
    if (dual) {
        n_neighbors = 27;
    }
    
    std::vector<int> nbs(n_neighbors, -1);

    for (const auto& v : vertices_ordered) {
        neighbors_3D(v, shape, nbs, dual, size, extra_vertex);
        for (int u : nbs) {
            if (u < size && birth_unchecked(u) <= birth_unchecked(v)) {
                int up = compute_root(u, ancestor);
                int vp = compute_root(v, ancestor);

                if (up != vp) {
                    if (birth_unchecked(up) < birth_unchecked(vp)) {
                        std::swap(up, vp);
                    }

                    children[vp].push_back(up);
                    parent[up] = vp;  // Update the parent
                    ancestor[up] = vp;  // Update the ancestor
                    root = vp;  // Set root to vp
                    linking_vertex[up] = v;  // Track linking vertex

                    // Debugging
                    // std::cout << vp << "," << up << std::endl;
                    // std::cout << "birth up: " << birth_unchecked(up) << std::endl;
                    // std::cout << "birth v: " << birth_unchecked(v) << std::endl << std::endl;
                    if (birth_unchecked(up) < birth_unchecked(v)) {
                        persistent_children[vp].push_back(up);
                        positive_pers.push_back(up);
                    }
                }
            }
        }
    }

    py::list children_list;
    for (const auto& child_vec : children) {
        py::list child_pylist;
        for (const auto& c : child_vec) {
            child_pylist.append(c);
        }
        children_list.append(child_pylist);
    }

    py::list persistent_children_list;
    for (const auto& p_child_vec : persistent_children) {
        py::list p_child_pylist;
        for (const auto& pc : p_child_vec) {
            p_child_pylist.append(pc);
        }
        persistent_children_list.append(p_child_pylist);
    }

    return py::make_tuple(
        py::array_t<int>(parent.size(), parent.data()),
        children_list,
        root,
        py::array_t<int>(linking_vertex.size(), linking_vertex.data()),
        persistent_children_list,
        py::array_t<int>(positive_pers.size(), positive_pers.data())
    );
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

    m.def("_link_reduce_vertices_cpp", &_link_reduce_vertices_cpp, "Alternative link reduce, iterating on vertices");

    m.def("_link_reduce_vertices_cpp_3D", &_link_reduce_vertices_cpp_3D, "Link reduce for 3D images, iterating on vertices");
}
