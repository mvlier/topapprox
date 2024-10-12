"""
test_lpf_1D.py

This file contains some 1 dimensional tests. This tests perform the low-persistence-filter
method with various thresholds values and compares the result with the expected values.
Furthermore, for each case it checks if the BHT is computed correctly. 
"""

import pytest
from topapprox import TopologicalFilterImage, TopologicalFilterGraph
import numpy as np


def test_1D_simple_python():
    aux_1D_simple("python")

def test_1D_simple_numba():
    aux_1D_simple("numba")

def test_1D_simple_numba2():
    aux_1D_simple("numba")

def test_1D_simple_cpp():
    aux_1D_simple("cpp")


def aux_1D_simple(method):
    # tests is a list nx3, n is the number of tests, and each entry has
    # original array (if None, then the array is the same as the previous one), epsilon, expected result
    tests = [[np.array([[0,2,1]]), 0.5, np.array([[0,2,1]])],
             [None, 1.0, np.array([[0,2,1]])],
             [None, 1.5, np.array([[0,2,2]])],
             [None, 4.0, np.array([[0,2,2]])],
             [np.array([[0,1,-2,0,-1,2,-1,0]]), 0.5, np.array([[0,1,-2,0,-1,2,-1,0]])],
             [None, 1.5, np.array([[1,1,-2,0,0,2,-1,0]])],
             [None, 4.0, np.array([[1,1,-2,0,0,2,2,2]])],
             [np.array([[0,2,2,1,1,4,1,5,4,5,-1]]), 0.5, np.array([[0,2,2,1,1,4,1,5,4,5,-1]])],
             [None, 1.0, np.array([[0,2,2,1,1,4,1,5,4,5,-1]])],
             [None, 1.5, np.array([[0,2,2,2,2,4,1,5,5,5,-1]])],
             [None, 3.0, np.array([[0,2,2,2,2,4,1,5,5,5,-1]])],
             [None, 3.5, np.array([[0,2,2,2,2,4,4,5,5,5,-1]])],
             [None, 5.0, np.array([[0,2,2,2,2,4,4,5,5,5,-1]])],
             [None, 5.5, np.array([[5,5,5,5,5,5,5,5,5,5,-1]])]
            ]
    # bhts for `tests`, ordered in the following way
    # test1 -> parent, linking_vertex, root, children
    tests_bhts = [[np.array([0,0,0]), 
                   np.array([ -1, 1, 1]),
                   0,
                   [[1, 2], [], []]
                   ],
                  [np.array([2, 0, 2, 2, 2, 2, 2, 6]), 
                   np.array([ 1,  1, -1,  3,  3,  5,  5,  7]),
                   2,
                   [[1], [], [3, 4, 0, 5, 6], [], [], [], [7], []]
                   ],
                   [[10,  0,  0,  4,  0,  0,  0,  0,  0,  0, 10],
                    [ 9,  1,  2,  4,  2,  5,  5,  7,  7,  9, -1],
                    10,
                    [[1, 2, 4, 5, 6, 7, 8, 9], [], [], [], [3], [], [], [], [], [], [0]],  
                   ]
    ]
    
    j=0
    for i in range(len(tests)):
        first_run = isinstance(tests[i][0], np.ndarray)
        if first_run:
            uf = TopologicalFilterImage(tests[i][0], method=method)
        result = uf.low_pers_filter(tests[i][1])
        assert np.all(result == tests[i][2]), "The resulting function is not as expected (1D simple case)"
        if first_run:
            parent, linking_vertex, root, children = uf.get_BHT(with_children=True)
            assert (np.all(parent == tests_bhts[j][0]) 
                    and np.all(linking_vertex == tests_bhts[j][1]) 
                    and root == tests_bhts[j][2]
                    and children == tests_bhts[j][3]
                    ), f"""(1D simple case): BHTs are different. \n
                    Predicted BHT: \n
                    Parents:          {tests_bhts[j][0]}\n
                    Linking vertices: {tests_bhts[j][1]}\n
                    Root:             {tests_bhts[j][2]}\n
                    Children:         {tests_bhts[j][3]}\n\n
                    Computed BHT: \n
                    Parents:          {parent}\n
                    Linking vertices: {linking_vertex}\n
                    Root:             {root}\n
                    Children:         {children}\n\n
                    """
            j+=1


        
