"""
test_pers_eliminated_2D.py

This file tests if lpf really eliminates all persistence below a given threshold.
And if the l-infinity distance to the original function is less than the threshold.
For testing we use images and pre computed persistence diagrams
"""

import pytest
import numpy as np
from topapprox import TopologicalFilterImage, TopologicalFilterGraph
import cripser


def test_pagoda():
    ''' tests applying the low-persistence-filter on a natural image "pagoda.np" '''
    eps_list = [0.1, 0.2, 0.5]
    for epsilon in eps_list:
        check_all_methods("pagoda.npy", epsilon)

def l_infy_norm(arr):
    ''' l-infty norm '''
    return np.max(np.abs(arr))

def check_distance(name, arr, epsilon, method):
    ''' receives `name` (string), `arr` (numpy array) and `epsilon` (float)
    an checks if the l-infty norm of `arr` is lower than `epsilon`.

    Remark: `arr` is in general the difference between two arrays, to measure
    their distance.
    '''
    distance = l_infy_norm(arr)
    assert distance < epsilon, f'''H0 filtered function is far apart from original function:\n
    Method:                     {method}\n
    Function name:              {name}\n
    Threshold epsilon:          {epsilon}\n
    Distance between functions: {distance}'''

def check_persistence_diagrams(name, img_filtered_H0, img_filtered_H1, img, epsilon, method, *, dual=False):
    PD_original = cripser.computePH(img, maxdim=1)
    PD_original = [PD_original[PD_original[:, 0]==i][:, 1:3] for i in range(2)]
    PD_expected = [PD_original[i][PD_original[i][:,1]-PD_original[i][:,0]>=epsilon] for i in range(2)] # only intervals >= epsilon
    PD_computed = cripser.computePH(img_filtered_H0, maxdim=0) # only computing 0th diagram
    PD_computed = [PD_computed[:, 1:3]]
    PD_H1 = cripser.computePH(img_filtered_H1, maxdim=1)
    PD_H1 = PD_H1[PD_H1[:, 0]==1][:, 1:3]
    PD_computed.append(PD_H1)
    for k in range(2):
        assert np.all(PD_computed[k] == PD_expected[k]), f'''
        For image {name} with epsilon={epsilon} and method={method} the {k}-PD differs from expected:\n
        Computed PD0: {PD_computed[k]}\n
        Expected PD0: {PD_expected[k]}'''

def check_pers_and_norm(name, epsilon, method, *, location = "tests/data_for_testing/"):
    ''' Receives the `name` of a numpy file (e.g. "filename.np"), a threshold `epsilon` and a 
    `method` and tests if computing LPF with
    '''
    img = np.load(location + name)
    uf = TopologicalFilterImage(img, method=method)
    uf_dual = TopologicalFilterImage(img, dual=True, method=method)
    img_filtered_H0 = uf.low_pers_filter(epsilon)
    img_filtered_H1 = uf_dual.low_pers_filter(epsilon)

    # DISTANCE CHECK
    check_distance(name, img_filtered_H0 - img, epsilon, method) # H0 filtered check
    check_distance(name, img_filtered_H1 - img, epsilon, method) # H1 filtered check

    # PERSISTENCE DIAGRAM CHECK
    check_persistence_diagrams(name, img_filtered_H0, img_filtered_H1, img, epsilon, method) # checks both H0 and H1 filtered
    return img_filtered_H0, img_filtered_H1

    

def check_all_methods(name, epsilon, *, location = "tests/data_for_testing/"):
    ''' Receives the `name` of a numpy file (e.g. "filename.np") and a threshold `epsilon`,
    and performs tests for all possible methods of computing the low-persistence-filter.
    Apart from checking that all methods are giving results within epsilon from the original
    function, it also checks that each method gives the same result as.
    '''
    list_methods = ["python", "numba", "cpp"]
    filtered_dict = {}
    for method in list_methods:
        filtered_dict[method] = check_pers_and_norm(name, epsilon, method=method, location = location)
    for j in range(1, len(list_methods)):
        for k in range(2):
            assert np.all(filtered_dict[list_methods[0]][k] == filtered_dict[list_methods[j]][k]), f'''
            The methods {list_methods[0]} and {list_methods[j]} produced different results for H{k} filtered {name},
            with threshold {epsilon}.
            '''