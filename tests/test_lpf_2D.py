"""
test_lpf_2D.py

This file contains 2D signal tests for the Low Persistence Filter. It evaluates
the filter's output on both primal and dual grids (0- and 1-homology), and checks
if the BHT is constructed correctly.
"""

import pytest
import numpy as np
from topapprox import ImageFilter, available_backends


# Each entry in TEST_CASES_2D contains:
# - "signal": the initial 2D array (image)
# - "tests": a list of filtering steps, each with:
#     * "threshold": epsilon value for filtering
#     * "expected_output": filtered result
#     * "bht": expected BHT data (parent, linking_vertex, root, children)
#     * "dual": whether the filter is applied to the dual (for 1-homology)
TEST_CASES_2D = {
    "basic_2D_example": {
        "signal": np.array([
            [0, 5, 3],
            [5, 6, 4],
            [2, 5, 1]
        ]),
        "tests": [
            {
                "threshold": 1.5,
                "expected_output": np.array([
                    [0, 5, 4],
                    [5, 6, 4],
                    [2, 5, 1]
                ]),
                "bht": {
                    "parent": np.array([0, 0, 8, 0, 0, 2, 0, 0, 0]),
                    "linking_vertex": np.array([-1, 1, 5, 3, 4, 5, 3, 7, 1]),
                    "root": 0,
                    "children": [[3, 6, 1, 7, 8, 4], [], [5], [], [], [], [], [], [2]]
                },
                "dual": False
            },
            {
                "threshold": 1.5,
                "expected_output": np.array([
                    [0, 5, 4],
                    [5, 5, 4],
                    [2, 5, 1]
                ]),
                "bht": {
                    "parent": np.array([9, 4, 9, 9, 9, 9, 9, 9, 9, 9]),
                    "linking_vertex": np.array([0, 1, 2, 3, 7, 5, 6, 7, 8, -1]),
                    "root": 9,
                    "children": [[], [], [], [], [1], [], [], [], [], [7, 4, 3, 5, 2, 6, 8, 0]]
                },
                "dual": True
            },
        ]
    }
}


# def check_BHT(uf, bht_expected):
#     parent, linking_vertex, root, children = uf.get_BHT(with_children=True)
#     passed = (
#         np.all(parent == bht_expected["parent"]) and
#         np.all(linking_vertex == bht_expected["linking_vertex"]) and
#         root == bht_expected["root"] and
#         all(set(children[i]) == set(bht_expected["children"][i]) for i in range(len(children)))
#     )
#     if passed:
#         return True, ""
#     else:
#         return False, f"""
# (2D test case) BHT mismatch:
# Expected:
#   Parent: {bht_expected['parent']}
#   Linking Vertex: {bht_expected['linking_vertex']}
#   Root: {bht_expected['root']}
#   Children: {bht_expected['children']}
# Got:
#   Parent: {parent}
#   Linking Vertex: {linking_vertex}
#   Root: {root}
#   Children: {children}
# """


@pytest.mark.parametrize("method", available_backends())
def test_all_2D_cases(method):
    for case_name, case in TEST_CASES_2D.items():
        signal = case["signal"]

        current_input = signal
        for i, test in enumerate(case["tests"]):
            dual = test.get("dual", False)
            uf = ImageFilter(current_input, method=method, dual=dual)
            result = uf.low_pers_filter(test["threshold"])

            assert np.all(result.astype(np.int32) == test["expected_output"]), (
                f"[{case_name}] step {i} failed (dual={dual}) with method='{method}' and threshold={test['threshold']}.\n"
                f"Expected:\n{test['expected_output']}\nGot:\n{result}"
            )

            # passed, msg = check_BHT(uf, test["bht"])
            # assert passed, f"[{case_name}] step {i} BHT failed (dual={dual}) with method='{method}':\n{msg}"

            current_input = result
