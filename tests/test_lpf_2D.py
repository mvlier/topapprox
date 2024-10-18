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
                        np.array([ 9,  4,  9,  9, 9,  9, 9, 9, 9,  9]),
                        np.array([ 0,  1,  2,  3,  7,  5,  6,  7,  8, -1]),
                        9,
                        [[], [], [], [], [1], [], [], [], [], [7, 4, 3, 5, 2, 6, 8, 0]]
                    ]
                ]
    
    j=0
    for i in range(len(tests)):
        if isinstance(tests[i][0], np.ndarray):
            uf = TopologicalFilterImage(tests[i][0], method=method)
        result = uf.low_pers_filter(tests[i][1])
        assert np.all(result == tests[i][2]), f'''
        Failed for array 0-homology filter with array :\n
        {tests[i][0]}\n
        With thresholf {tests[i][1]} expected:\n
        {tests[i][2]}\n
        but got:\n
        {result}'''
        check = check_BHT(uf, tests_bhts[j]) 
        assert check[0], check[1]
        j += 1
        uf = TopologicalFilterImage(result, dual=True, method=method)
        result = uf.low_pers_filter(tests[i][3])
        assert np.all(result == tests[i][4]), f'''
        Failed for array 1-homology filter with array :\n
        {tests[i][2]}\n
        With thresholf {tests[i][3]} expected:\n
        {tests[i][4]}\n
        but got:\n
        {result}'''
        check = check_BHT(uf, tests_bhts[j]) 
        assert check[0], check[1]
        j += 1
