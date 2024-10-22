"""
Pure python link and reduce function
"""

__all__ = ["_link_reduce", "compute_root"]

import numpy as np

def _link_reduce(birth, edges, epsilon, keep_basin=False):
    """Link and reduce function without Numba."""
    
    #persistence = [(birth.min(), np.inf, birth.argmin())]
    parent = np.arange(birth.shape[0])
    ancestor = np.arange(birth.shape[0])
    linking_vertex = np.full(birth.shape[0], -1)
    root = 0  # Initialize root as index 0
    positive_pers = [] # list to store the positive persistence vertices apart from root

    # Initialize children as a list of lists
    children = [[] for _ in range(birth.shape[0])]

    # Record only the children with non-zero persistence
    persistent_children = [[] for _ in range(birth.shape[0])]

    for i in range(edges.shape[0]):
        ui, vi = edges[i, 0], edges[i, 1]
        # Find parents of u and v
        up = compute_root(ui, ancestor)
        vp = compute_root(vi, ancestor)
        death = max(birth[ui], birth[vi])

        if up != vp:  # If their connected components are different
            if birth[up] < birth[vp]:
                up, vp = vp, up  # up is the younger and to be killed

            # Tree structure
            children[vp].append(up)
            parent[up] = vp
            ancestor[up] = vp  # for quicker processing of ancestors
            root = vp  # by the end of the loop root will store the only vertex that is its own parent
            if birth[ui] > birth[vi]:
                linking_vertex[up] = ui
            else:
                linking_vertex[up] = vi

            if birth[up] < death:  # A cycle is produced
                persistent_children[vp].append(up)
                positive_pers.append(up)
                
                # if death - birth[up] < epsilon:
                #     desc = compute_descendants(up, children)

                #     # Update modified for each descendant
                #     for index in range(len(desc)): 
                #         modified[desc[index]] = death

    return parent, children, root, linking_vertex, persistent_children, np.array(positive_pers)


def compute_root(v, ancestor):
    if ancestor[v] == v:
        return v
    ancestor[v] = compute_root(ancestor[v], ancestor)
    return ancestor[v]


def _link_reduce_vertices(birth:np.ndarray, shape:tuple, dual:bool):
    n, m = shape
    size = n * m + 1 if dual else n * m
    #neighbors = np.empty(8 if dual else 4, dtype=np.uint32)
    
    # Precompute sorted order and birth index
    vertices_ordered = np.argsort(birth)
    idx = np.unravel_index(vertices_ordered[1:], (n,m))
    # vertex_birth_index = np.empty_like(vertices_ordered)
    # vertex_birth_index[vertices_ordered] = np.arange(len(vertices_ordered))
    
    parent = np.arange(size)
    ancestor = np.arange(size)
    linking_vertex = np.full(size, -1)
    root = [0]
    positive_pers = []
    
    children = [[] for _ in range(size)]
    persistent_children = [[] for _ in range(size)]
    
    for i, v in enumerate(vertices_ordered[1:]):
        #_neighbors(v, (n, m), neighbors, dual)
        check_and_link(v, idx[0][i], idx[1][i], n, m, dual, birth, 
                       children, parent, ancestor, root, 
                       linking_vertex, persistent_children, positive_pers)
        

    return parent, children, root[0], linking_vertex, persistent_children, np.array(positive_pers)


def check_and_link(v, i, j, n, m, dual, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers):
    notleft, notright = j!=0, j!=m-1
    nottop, notbottom = i!=0, i!=n-1
    if notright:
        _link_update(v, v+1, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers)
    if notleft:
        _link_update(v, v-1, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers)
    if nottop:
        _link_update(v, v-m, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers)
    if notbottom:
        _link_update(v, v+m, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers)
    if dual:
        if notleft and nottop:
            _link_update(v, v-m-1, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers)
        else:
            _link_update(v, n*m, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers)
        if notleft and notbottom:
            _link_update(v, v+m-1, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers)
        if notright and nottop:
            _link_update(v, v-m+1, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers)
        if notright and notbottom:
            _link_update(v, v+m+1, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers)
        else:
            _link_update(v, n*m, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers)



def _link_update(v, u, birth, children, parent, ancestor, root, linking_vertex, persistent_children, positive_pers):
    if birth[u] <= birth[v]:
        up = compute_root(u, ancestor)
        vp = compute_root(v, ancestor)

        if up != vp:
            # if vertex_birth_index[up] < vertex_birth_index[vp]:
            if birth[up] < birth[vp]:
                up, vp = vp, up

            children[vp].append(up)
            parent[up] = vp
            ancestor[up] = vp
            root[0] = vp
            linking_vertex[up] = v

            if birth[up] < birth[v]:
                persistent_children[vp].append(up)
                positive_pers.append(up)


# def _neighbors(v:int, shape:tuple, nbs:np.ndarray, dual:bool):
#     n, m = shape
#     left, right = v % m == 0, v % m == m - 1
#     top, bottom = v < m, v >= (n - 1) * m

#     nbs[:4] = (
#         n*m+1 if left else v-1,
#         n*m+1 if right else v+1,
#         n*m+1 if top else v-m,
#         n*m+1 if bottom else v+m
#     )
    
#     if dual:
#         nbs[4:] = (
#             n*m if top or left else v-m-1,
#             n*m if bottom or right else v+m+1,
#             n*m+1 if top or right else v-m+1,
#             n*m+1 if bottom or left else v+m-1
#         )

