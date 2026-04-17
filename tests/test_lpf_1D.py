import pytest
import numpy as np
from topapprox import ImageFilter, available_backends

# TEST_CASES is a dictionary that maps descriptive test names to corresponding 1D signal test data.
# Each entry consists of:
#   - "signal": a 1D numpy array representing the original input signal for that test case.
#   - "tests": a list of dictionaries, each containing:
#       * "threshold": the persistence threshold used for filtering,
#       * "expected_output": the expected result after applying the low persistence filter.
# These tests validate that the filter correctly preserves or simplifies the signal structure
# depending on the specified threshold.

TEST_CASES = {
    "single_peak_no_filtering": {
        "signal": np.array([[0, 2, 1]]),
        "tests": [
            {"threshold": 0.5, "expected_output": np.array([[0, 2, 1]])},
            {"threshold": 1.0, "expected_output": np.array([[0, 2, 1]])},
            {"threshold": 1.5, "expected_output": np.array([[0, 2, 2]])},
            {"threshold": 4.0, "expected_output": np.array([[0, 2, 2]])},
        ],
    },
    "multiple_peaks_symmetric": {
        "signal": np.array([[0, 1, -2, 0, -1, 2, -1, 0]]),
        "tests": [
            {"threshold": 0.5, "expected_output": np.array([[0, 1, -2, 0, -1, 2, -1, 0]])},
            {"threshold": 1.5, "expected_output": np.array([[1, 1, -2, 0, 0, 2, -1, 0]])},
            {"threshold": 4.0, "expected_output": np.array([[1, 1, -2, 0, 0, 2, 2, 2]])},
        ],
    },
    "complex_plateaus_and_valleys": {
        "signal": np.array([[0, 2, 2, 1, 1, 4, 1, 5, 4, 5, -1]]),
        "tests": [
            {"threshold": 0.5, "expected_output": np.array([[0, 2, 2, 1, 1, 4, 1, 5, 4, 5, -1]])},
            {"threshold": 1.0, "expected_output": np.array([[0, 2, 2, 1, 1, 4, 1, 5, 4, 5, -1]])},
            {"threshold": 1.5, "expected_output": np.array([[0, 2, 2, 2, 2, 4, 1, 5, 5, 5, -1]])},
            {"threshold": 3.0, "expected_output": np.array([[0, 2, 2, 2, 2, 4, 1, 5, 5, 5, -1]])},
            {"threshold": 3.5, "expected_output": np.array([[0, 2, 2, 2, 2, 4, 4, 5, 5, 5, -1]])},
            {"threshold": 5.0, "expected_output": np.array([[0, 2, 2, 2, 2, 4, 4, 5, 5, 5, -1]])},
            {"threshold": 5.5, "expected_output": np.array([[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1]])},
        ],
    },
}


@pytest.mark.parametrize("method", available_backends())
def test_all_1D_cases(method):
    for case_name, case in TEST_CASES.items():
        signal = case["signal"]
        tests = case["tests"]
        uf = ImageFilter(signal, method=method)

        for test in tests:
            epsilon = test["threshold"]
            expected = test["expected_output"]
            result = uf.low_pers_filter(epsilon)

            assert np.all(result == expected), (
                f"[{case_name}] failed with method='{method}' and threshold={epsilon}.\n"
                f"Expected:\n{expected}\nGot:\n{result}"
            )
