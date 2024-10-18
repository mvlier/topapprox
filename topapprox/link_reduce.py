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

    modified = birth.copy()

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


# def compute_descendants(v, children):
#     desc = []  
#     descendants(v, children, desc)  
#     return desc


# def descendants(v, children, desc):
#     desc.append(v)  # Append the current node
#     for child in children[v]:  # Use a loop to iterate over the children
#         descendants(child, children, desc)  # Recursively find descendants
