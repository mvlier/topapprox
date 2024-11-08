#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

class BasinHierarchyTree {
public:
    // Attributes
    py::array_t<int> parent;
    std::vector<std::vector<int>> children;
    int root;
    py::array_t<int> linking_vertex;
    std::vector<std::vector<int>> persistent_children;
    py::array_t<int> positive_pers;
    py::array_t<int> persistence;
    py::array_t<int> birth;

    // Constructor
    BasinHierarchyTree(py::array_t<int> parent_, 
                       std::vector<std::vector<int>> children_, 
                       int root_,
                       py::array_t<int> linking_vertex_, 
                       std::vector<std::vector<int>> persistent_children_, 
                       py::array_t<int> positive_pers_, 
                       py::array_t<int> birth_)
        : parent(parent_), 
          children(children_), 
          root(root_),
          linking_vertex(linking_vertex_), 
          persistent_children(persistent_children_), 
          positive_pers(positive_pers_), 
          birth(birth_)
    {}

    // Methods
    py::array_t<int> _low_pers_filter(float epsilon) {
        py::array_t<int> modified = birth;  // Copy birth array
        auto m = modified.mutable_unchecked<1>();

        for (int vertex : persistent_children[root]) {
            filter_branch(vertex, epsilon, modified);
        }
        return modified;
    }

    void filter_branch(int vertex, float epsilon, py::array_t<int>& modified) {
        auto linking = linking_vertex.mutable_unchecked<1>();
        auto b = birth.mutable_unchecked<1>();
        auto m = modified.mutable_unchecked<1>();

        int link_vertex = linking[vertex];
        float persistence = b[link_vertex] - b[vertex];

        if (0 < persistence && persistence < epsilon) {
            std::vector<int> descendants = get_descendants_recursive(vertex);
            for (int desc : descendants) {
                m[desc] = b[link_vertex];
            }
        } else if (persistence >= epsilon) {
            for (int child_vertex : persistent_children[vertex]) {
                filter_branch(child_vertex, epsilon, modified);
            }
        }
    }

    py::array_t<int> _lpf_size_filter(float epsilon, std::vector<int> size_range) {
        py::array_t<int> modified = birth;  // Copy birth array
        auto m = modified.mutable_unchecked<1>();

        for (int vertex : persistent_children[root]) {
            filter_branch_with_size(vertex, epsilon, modified, size_range);
        }
        return modified;
    }

    void filter_branch_with_size(int vertex, float epsilon, py::array_t<int>& modified, std::vector<int> size_range) {
        auto linking = linking_vertex.mutable_unchecked<1>();
        auto b = birth.mutable_unchecked<1>();
        auto m = modified.mutable_unchecked<1>();

        int link_vertex = linking[vertex];
        float persistence = b[link_vertex] - b[vertex];

        if (0 < persistence && persistence < epsilon) {
            std::vector<int> descendants = get_descendants_recursive(vertex);
            int size = descendants.size();
            if (size_range[0] <= size && size <= size_range[1]) {
                for (int desc : descendants) {
                    m[desc] = b[link_vertex];
                }
            } else {
                for (int child_vertex : persistent_children[vertex]) {
                    filter_branch_with_size(child_vertex, epsilon, modified, size_range);
                }
            }
        } else if (persistence >= epsilon) {
            for (int child_vertex : persistent_children[vertex]) {
                filter_branch_with_size(child_vertex, epsilon, modified, size_range);
            }
        }
    }

    std::vector<int> get_descendants_recursive(int vertex) {
        std::vector<int> descendants;
        get_descendants(vertex, descendants);
        return descendants;
    }

    void get_descendants(int vertex, std::vector<int>& descendants) {
        descendants.push_back(vertex);
        for (int child : children[vertex]) {
            get_descendants(child, descendants);
        }
    }

    int get_depth(int vertex) {
        int depth = 0;
        auto p = parent.mutable_unchecked<1>();
        while (p[vertex] != vertex) {
            vertex = p[vertex];
            depth++;
        }
        return depth;
    }

    int get_height(int vertex) {
        if (children[vertex].empty()) {
            return 0;
        }
        int max_height = 0;
        for (int child : children[vertex]) {
            int child_height = get_height(child);
            if (child_height > max_height) {
                max_height = child_height;
            }
        }
        return 1 + max_height;
    }

    int get_positive_pers_height(int vertex) {
        if (persistent_children[vertex].empty()) {
            return 0;
        }
        int max_height = 0;
        for (int child : persistent_children[vertex]) {
            int child_height = get_positive_pers_height(child);
            if (child_height > max_height) {
                max_height = child_height;
            }
        }
        return 1 + max_height;
    }
};

PYBIND11_MODULE(bht_cpp, m) {
    py::class_<BasinHierarchyTree>(m, "BasinHierarchyTree")
        .def(py::init<py::array_t<int>, std::vector<std::vector<int>>, int, py::array_t<int>, std::vector<std::vector<int>>, py::array_t<int>, py::array_t<int>>())
        .def("_low_pers_filter", &BasinHierarchyTree::_low_pers_filter)
        .def("_lpf_size_filter", &BasinHierarchyTree::_lpf_size_filter)
        .def("filter_branch", &BasinHierarchyTree::filter_branch)
        .def("filter_branch_with_size", &BasinHierarchyTree::filter_branch_with_size)
        .def("get_descendants_recursive", &BasinHierarchyTree::get_descendants_recursive)
        .def("get_depth", &BasinHierarchyTree::get_depth)
        .def("get_height", &BasinHierarchyTree::get_height)
        .def("get_positive_pers_height", &BasinHierarchyTree::get_positive_pers_height);
}
