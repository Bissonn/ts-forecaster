import pytest
import numpy as np
from utils.data_utils import create_sliding_window

# Test data to be used across multiple tests
SAMPLE_DATA_MULTIVARIATE = np.arange(20).reshape(10, 2)
SAMPLE_DATA_UNIVARIATE = np.arange(10)

# --- Success Cases ---

def test_standard_case_multivariate():
    """Tests the sliding window creation for a standard multivariate case."""
    X, y = create_sliding_window(SAMPLE_DATA_MULTIVARIATE, window_size=3, forecast_steps=2, step=1)

    # Expected shapes: (n_samples, window_size, features) and (n_samples, forecast_steps, features)
    # n_samples = 10 - 3 - 2 + 1 = 6
    assert X.shape == (6, 3, 2)
    assert y.shape == (6, 2, 2)

    # Check the content of the first and last windows to ensure correctness
    # First X should be [[0, 1], [2, 3], [4, 5]]
    np.testing.assert_array_equal(X[0], np.array([[0, 1], [2, 3], [4, 5]]))
    # First y should be [[6, 7], [8, 9]]
    np.testing.assert_array_equal(y[0], np.array([[6, 7], [8, 9]]))
    
    # Last X should be [[10, 11], [12, 13], [14, 15]]
    np.testing.assert_array_equal(X[-1], np.array([[10, 11], [12, 13], [14, 15]]))
    # Last y should be [[16, 17], [18, 19]]
    np.testing.assert_array_equal(y[-1], np.array([[16, 17], [18, 19]]))

def test_standard_case_univariate():
    """Tests the sliding window creation for a standard univariate case."""
    X, y = create_sliding_window(SAMPLE_DATA_UNIVARIATE, window_size=4, forecast_steps=1, step=1)

    # Expected shapes: (n_samples, window_size, 1) and (n_samples, forecast_steps, 1)
    # n_samples = 10 - 4 - 1 + 1 = 6
    assert X.shape == (6, 4, 1)
    assert y.shape == (6, 1, 1)

    # Check content of the first window
    np.testing.assert_array_equal(X[0], np.array([[0], [1], [2], [3]]))
    np.testing.assert_array_equal(y[0], np.array([[4]]))

def test_different_step_size():
    """Tests that the sliding window correctly handles a step size greater than 1."""
    X, y = create_sliding_window(SAMPLE_DATA_MULTIVARIATE, window_size=3, forecast_steps=2, step=3)
    
    # Expected number of samples: (10 - 3 - 2 + 1) // 3 = 2
    assert X.shape == (2, 3, 2)
    assert y.shape == (2, 2, 2)

    # First window starts at index 0
    np.testing.assert_array_equal(X[0], np.array([[0, 1], [2, 3], [4, 5]]))
    # Second window should start at index 3 (0 + step)
    np.testing.assert_array_equal(X[1], np.array([[6, 7], [8, 9], [10, 11]]))

# --- Failure Cases (Input Validation) ---

def test_raises_error_on_insufficient_data_length():
    """Tests that a ValueError is raised if the data is too short for the window and forecast steps."""
    with pytest.raises(ValueError, match="data length .* is insufficient"):
        create_sliding_window(SAMPLE_DATA_UNIVARIATE, window_size=8, forecast_steps=3)

def test_raises_error_on_empty_data():
    """Tests that a ValueError is raised for an empty input array."""
    with pytest.raises(ValueError, match="data cannot be empty"):
        create_sliding_window(np.array([]), window_size=1, forecast_steps=1)

def test_raises_error_on_nan_values():
    """Tests that a ValueError is raised if the input data contains NaN."""
    data_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError, match="data cannot contain NaN or infinite values"):
        create_sliding_window(data_with_nan, window_size=1, forecast_steps=1)

@pytest.mark.parametrize("window_size, forecast_steps, step, error_msg", [
    (0, 5, 1, "window_size must be a positive integer"),
    (-1, 5, 1, "window_size must be a positive integer"),
    (5, 0, 1, "forecast_steps must be a positive integer"),
    (5, -1, 1, "forecast_steps must be a positive integer"),
    (3, 2, 0, "step must be a positive integer"),
    (3, 2, -1, "step must be a positive integer"),
])
def test_raises_error_on_invalid_parameters(window_size, forecast_steps, step, error_msg):
    """Tests that ValueErrors are raised for various invalid integer parameters."""
    with pytest.raises(ValueError, match=error_msg):
        create_sliding_window(SAMPLE_DATA_UNIVARIATE, window_size, forecast_steps, step)

def test_raises_error_on_non_numpy_input():
    """Tests that a ValueError is raised if the input is not a NumPy array."""
    with pytest.raises(ValueError, match="data must be a NumPy array"):
        create_sliding_window([1, 2, 3, 4, 5], window_size=2, forecast_steps=1)
