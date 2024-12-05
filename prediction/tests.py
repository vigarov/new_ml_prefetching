import pandas as pd

# Import the function to test
from leap import Leap
import numpy as np

# Test Case 1: Sequence as a list
def test_build_parrallel_batches_with_list():
    sequence = ["0x1000", "0x2000", "0x3000", "0x4000", "0x5000", "0x6000", "0x7000"]
    history_window_size = 3
    output_window_size = 2

    history, output = Leap.build_batches(sequence, history_window_size, output_window_size)

    expected_history = [
        np.array([1, 2, 3]),
        np.array([2, 3, 4]),
        np.array([3, 4, 5]),
        np.array([4, 5, 6]),
        np.array([5, 6, 7]),
    ]
    expected_output = [
        np.array([4, 5]),
        np.array([5, 6]),
        np.array([6, 7]),
        np.array([7,7]),
        np.array([7,7]),
    ]

    print("Test Case 1: Sequence as a list")
    for h, eh in zip(history, expected_history):
        assert np.array_equal(h, eh), f"Expected history {eh}, got {h}"
    for o, eo in zip(output, expected_output):
        assert np.array_equal(o, eo), f"Expected output {eo}, got {o}"
    print("Passed\n")

# Test Case 2: Sequence as a Pandas Series
def test_build_parrallel_batches_with_series():
    sequence = pd.Series(["0x1000", "0x2000", "0x3000", "0x4000", "0x5000", "0x6000", "0x7000"])
    history_window_size = 3
    output_window_size = 2

    history, output = Leap.build_batches(sequence, history_window_size, output_window_size)

    expected_history = pd.Series([
        np.array([1, 2, 3]),
        np.array([2, 3, 4]),
        np.array([3, 4, 5]),
        np.array([4, 5, 6]),
        np.array([5, 6, 7]),
    ])
    expected_output = pd.Series([
        np.array([4, 5]),
        np.array([5, 6]),
        np.array([6, 7]),
        np.array([7,7]),
        np.array([7,7]),
    ])

    print("Test Case 2: Sequence as a Pandas Series")
    pd.testing.assert_series_equal(history, expected_history, check_dtype=False, obj="History Windows")
    pd.testing.assert_series_equal(output, expected_output, check_dtype=False, obj="Output Windows")
    print("Passed\n")

def test_build_cut_hist():
    sequence = ["0x1000", "0x2000", "0x3000", "0x4000", "0x5000", "0x6000", "0x7000"]
    history_window_size = 3
    output_window_size = 2

    # Call the function
    history, output = Leap.build_batches(sequence, history_window_size, output_window_size,hist_remove_after=True,outputs_fill_after=False)

    expected_history = [
        np.array([1, 2, 3]),
        np.array([2, 3, 4]),
        np.array([3, 4, 5]),
    ]
    expected_output = [
        np.array([4, 5]),
        np.array([5, 6]),
        np.array([6, 7]),
    ]

    print("Test Case 3: Cutthrough")
    for h, eh in zip(history, expected_history):
        assert np.array_equal(h, eh), f"Expected history {eh}, got {h}"
    for o, eo in zip(output, expected_output):
        assert np.array_equal(o, eo), f"Expected output {eo}, got {o}"
    print("Passed\n")



if __name__ == "__main__":
    test_build_parrallel_batches_with_list()
    test_build_parrallel_batches_with_series()
    test_build_cut_hist()