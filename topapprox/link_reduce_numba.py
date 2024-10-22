"""main module for topapprox

Todo:
    * docstring
    
"""

__all__ = ["_link_reduce_numba", "compute_descendants_numba", "compute_root_numba"]

import numpy as np
from numba import njit,f8,i8,typed, int64
from numba.typed import List


@njit(parallel=True, fastmath=True)
def _link_reduce_numba(birth, edges, epsilon, keep_basin=False):
    """Link and reduce function."""
    
    #persistence = typed.List()
    #persistence.append((birth.min(), np.inf, birth.argmin()))
    parent = np.arange(birth.shape[0])
    ancestor = np.arange(birth.shape[0])
    linking_vertex = np.full(birth.shape[0], -1)
    root = 0 # itialize root as index 0
    positive_pers = List.empty_list(int64) # list to store the positive persistence vertices apart from root
    
    # Initialize children as a typed list of typed lists
    children = typed.List()
    for _ in range(birth.shape[0]):
        children.append(typed.List.empty_list(np.int64))

    # Record only the children with non zero persistence
    persistent_children = typed.List()
    for _ in range(birth.shape[0]):
        persistent_children.append(typed.List.empty_list(np.int64))

    for i in range(edges.shape[0]):
        ui, vi = edges[i, 0], edges[i, 1]
        # Find parents of u and v
        up = compute_root_numba(ui, ancestor)
        vp = compute_root_numba(vi, ancestor)
        death = max(birth[ui], birth[vi])

        if up != vp:  # If their connected components are different
            if birth[up] < birth[vp]:
                up, vp = vp, up  # up is the younger and to be killed
            
            # Tree structure
            children[vp].append(up)
            parent[up] = vp
            ancestor[up] = vp # for quicker processing of ancestors
            root = vp # by the end of the loop root will store the only vertex that is its own parent
            if birth[ui] > birth[vi]:
                linking_vertex[up] = ui
            else:
                linking_vertex[up] = vi
            
            if birth[up] < death:  # A cycle is produced
                persistent_children[vp].append(up)
                positive_pers.append(up)
                # persistence.append((birth[up], death, up))
                
                # if death - birth[up] < epsilon:
                #     desc = compute_descendants_numba(up, children)
                    
                #     # Update modified for each descendant
                #     for index in range(len(desc)): 
                #         modified[desc[index]] = death

    return parent, children, root, linking_vertex, persistent_children, positive_pers


@njit(fastmath=True)
def compute_root_numba(v, ancestor):
    if ancestor[v] == v:
        return v
    ancestor[v] = compute_root_numba(ancestor[v], ancestor)
    return ancestor[v]

def link_reduce_wrapper(birth, edges, epsilon, keep_basin=False):
    a,b,c,d,e,f = _link_reduce_numba(birth, edges, epsilon, keep_basin=keep_basin)
    return a,b,c,d,e,np.array(f)

def link_reduce_vertices_wrapper(birth, shape):
    a,b,c,d,e,f = _link_reduce_vertices_numba(birth, shape)
    return a,b,c,d,e,np.array(f)


# @njit(fastmath=True)
# def compute_descendants_numba(v, children):
#     desc = typed.List.empty_list(np.int64)  
#     descendants_numba(v, children, desc)  
#     return desc

# @njit(fastmath=True)
# def descendants_numba(v, children, desc):
#     desc.append(v)  # Append the current node
#     for i in range(len(children[v])):  # Use len() to iterate over the children
#         child_index = children[v][i]  # Access the child index
#         descendants_numba(child_index, children, desc)  # Recursively find descendants


@njit(parallel=True, fastmath=True)
def _link_reduce_vertices_numba(birth, shape):
    '''Link reduce function iterating over vertices (instead of edges)
    '''

    n,m = shape
    vertices_ordered = np.argsort(birth)
    vertex_birth_index = np.argsort(vertices_ordered) # if vertex_birth_index[v] = 7, then v is the 8th vertex in the ordering
    #persistence = [(birth.min(), np.inf, birth.argmin())]
    parent = np.arange(birth.shape[0])
    ancestor = np.arange(birth.shape[0])
    linking_vertex = np.full(birth.shape[0], -1)
    root = 0  # Initialize root as index 0
    positive_pers = List.empty_list(int64) # list to store the positive persistence vertices apart from root
    
    # Initialize children as a typed list of typed lists
    children = typed.List()
    for _ in range(birth.shape[0]):
        children.append(typed.List.empty_list(np.int64))

    # Record only the children with non zero persistence
    persistent_children = typed.List()
    for _ in range(birth.shape[0]):
        persistent_children.append(typed.List.empty_list(np.int64))

    #neighbors = np.empty(4, dtype=np.uint32)
    for v in vertices_ordered:
        #neighbors[:] = _neighbors
        neighbors = _neighbors(v, (n,m))
        for u in neighbors:
            if u > -1:
                if vertex_birth_index[u] < vertex_birth_index[v]:
                    up = compute_root_numba(u, ancestor)
                    vp = compute_root_numba(v, ancestor)

                    if up != vp:  # If their connected components are different
                        if vertex_birth_index[up] < vertex_birth_index[vp]:
                            up, vp = vp, up  # up is the younger and to be killed

                        # Tree structure
                        children[vp].append(up)
                        parent[up] = vp
                        ancestor[up] = vp  # for quicker processing of ancestors
                        root = vp  # by the end of the loop root will store the only vertex that is its own parent
                        linking_vertex[up] = v

                        if birth[up] < birth[v]:  # A cycle is produced
                            persistent_children[vp].append(up)
                            positive_pers.append(up)

    return parent, children, root, linking_vertex, persistent_children, positive_pers


@njit(parallel=True, fastmath=True)
def _neighbors(v, shape):
    n,m = shape
    nbs = [
        -1 if v%m == 0 else v-1,
        -1 if v%m == m-1 else v+1,
        -1 if v < m-1 else v-m,
        -1 if v > (n-1)*m - 1 else v+m
    ]
    return nbs