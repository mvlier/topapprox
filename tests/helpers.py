"""
helpers.py

Additional functions used in several testing cases.
"""

import pytest
from topapprox import GraphFilter, ImageFilter
import numpy as np

def check_bht(parent, linking_vertex, expected_parent, expected_linking_vertex, birth, *,
              children=None, root=None, persistent_children=None, positive_pers=None):
    """
    Checks if the bht is correct.
    The key word arguments can be passes for extra checking.

    Parameters:
    ----------
    parent: np array

    linking_vertex: np array

    birth: np array
           Birth values for each vertex

    expected_parent: list of np arrays
                     This list contains all possible parent arrays.

    expected_linking_vertex: list of np arrays
                             This list contains all possible linking_vertices, in accordance
                             with `expected_parent`, do each pair 
                             (`expected_parent`, `expected_linking_vertex`) determines a BHT.

    children: list of lists

    root: int

    persistent_children: list of lists

    positive_pers: np array

    """
    n = len(parent)
    check_bht = np.any([(np.all(parent == ep) and np.all(linking_vertex == elv)) for ep, elv in zip(expected_parent, expected_linking_vertex)])
    if not check_bht:
        return False, "Invalid BHT"
    result = True
    msg = "BHT is OK.\n"
    
    if not children is None:
        expected_children = [[] for x in parent]
        for i, x in enumerate(parent):
            expected_children[x].append(i)

        if not np.all([set(children[i]) == set(expected_children[i]) for i in range(n)]):
            result = False
            msg += "Children list is WRONG!\n"
    
    if not root is None:
        identity = np.arange(n)
        expected_root = parent[parent==identity]
        if (len(expected_root) != 1) or (expected_root[0] != root):
            result = False
            msg += "Root is WRONG!\n"

    if not persistent_children is None:
        expected_persistent_children = [[] for x in parent]
        for i, x in enumerate(parent):
            if birth[linking_vertex] != birth[i]:
                expected_persistent_children[x].append(i)
        
        if not np.all([set(persistent_children[i]) == set(expected_persistent_children[i]) for i in range(n)]):
            result = False
            msg += "Persistent Children is WRONG!\n"
        
    if not positive_pers is None:
        expected_positive_pers = set([i for i in range(n) if birth[i] != birth[linking_vertex[i]]])

        if set(positive_pers) != expected_positive_pers:
            result = False
            msg += "Positive Persistence is WRONG!\n"
        
    return result, msg
