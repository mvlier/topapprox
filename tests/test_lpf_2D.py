"""
test_lpf_2D.py

This file contains some 2 dimensional tests. This tests perform the low-persistence-filter
method with various thresholds values and compares the result with the expected values.
Furthermore, for each case it checks if the BHT is computed correctly. 
"""

import pytest
from topapprox import TopologicalFilterImage, TopologicalFilterGraph
import numpy as np

def test_2D_simple_python():
    aux_2D_simple("python")

def test_2D_simple_numba():
    aux_2D_simple("numba")

def test_2D_simple_numba2():
    aux_2D_simple("numba")

def test_2D_simple_cpp():
    aux_2D_simple("cpp")

def check_BHT(uf, bht_expected):
    parent, linking_vertex, root, children = uf.get_BHT(with_children=True)
    if (np.all(parent == bht_expected[0]) 
                    and np.all(linking_vertex == bht_expected[1]) 
                    and root == bht_expected[2]
                    and children == bht_expected[3]
                    ):
        return True, " "
    return False, f"""(2D simple case): BHTs are different. \n
                    Predicted BHT: \n
                    Parents:          {bht_expected[0]}\n
                    Linking vertices: {bht_expected[1]}\n
                    Root:             {bht_expected[2]}\n
                    Children:         {bht_expected[3]}\n\n
                    Computed BHT: \n
                    Parents:          {parent}\n
                    Linking vertices: {linking_vertex}\n
                    Root:             {root}\n
                    Children:         {children}\n\n
                    """
        

def aux_2D_simple(method):
    # tests is a list nx3, n is the number of tests, and each entry has
    # original array (if None, then the array is the same as the previous one), epsilon, expected result
    tests = [[np.array([[0, 5, 3],\
                        [5, 6, 4],\
                        [2, 5, 1]]), 
              1.5, # 0-homology-filtering
              np.array([[0, 5, 4],\
                        [5, 6, 4],\
                        [2, 5, 1]]),
              1.5, # 1-homology-filtering
              np.array([[0, 5, 4],\
                        [5, 5, 4],\
                        [2, 5, 1]])]
            ]
    tests_bhts = [
                    [
                        np.array([0, 0, 8, 0, 0, 2, 0, 0, 0]),
                        np.array([-1,  1,  5,  3,  4,  5,  3,  7,  1]),
                        0,
                        [[3, 6, 1, 7, 8, 4], [], [5], [], [], [], [], [], [2]]
                    ],
                    [
                        np.array([ 5,  9,  9,  9, 24,  1, 24, 24, 24,  4, 24, 24, 24, 24, 24, 21, 24, 24, 24, 24, 24, 24, 24, 24, 24]),
                        np.array([ 5,  2,  3,  9, 14,  1,  6,  7,  8,  4, 15, 11, 17, 13, 19, 21, 16, 17, 18, 24, 21, 22, 23, 24, -1]),
                        24,
                        [[], [5], [], [], [9], [0], [], [], [], [3, 2, 1], [], [], [], [], [], [], [], [], [], [], [], [15], [], [], [23, 22, 21, 20, 19, 14, 10, 4, 17, 11, 7, 12, 8, 13, 16, 18, 6]]
                    ]
                ]
    
    j=0
    for i in range(len(tests)):
        if isinstance(tests[i][0], np.ndarray):
            uf = TopologicalFilterImage(tests[i][0], method=method)
        result = uf.low_pers_filter(tests[i][1])
        assert np.all(result == tests[i][2])
        check = check_BHT(uf, tests_bhts[j]) 
        assert check[0], check[1]
        j += 1
        uf = TopologicalFilterImage(result, dual=True, method=method)
        result = uf.low_pers_filter(tests[i][3])
        assert np.all(result == tests[i][4])
        check = check_BHT(uf, tests_bhts[j]) 
        assert check[0], check[1]
        j += 1
