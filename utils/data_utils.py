"""Module for preparing time series data for forecasting models.

This module provides utilities to create training data using sliding window techniques,
suitable for models like LSTM, Transformer, and Generic Transformer.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def create_sliding_window(
    data: np.ndarray, window_size: int, forecast_steps: int, step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training data (X, y) using a sliding window approach.

    Args:
        data: Time series data as a NumPy array with shape (time, features) or (time,) for univariate data.
        window_size: Number of input time steps (window size).
        forecast_steps: Number of time steps to forecast.
        step: Step size for sliding the window. Defaults to 1.

    Returns:
        Tuple of two NumPy arrays:
            - X: Input sequences with shape (n_samples, window_size, features).
            - y: Target sequences with shape (n_samples, forecast_steps, features).

    Raises:
        ValueError: If inputs are invalid (e.g., empty data, insufficient length, invalid parameters).
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a NumPy array.")
    if data.size == 0:
        raise ValueError("data cannot be empty.")
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer.")
    if not isinstance(forecast_steps, int) or forecast_steps < 1:
        raise ValueError("forecast_steps must be a positive integer.")
    if not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        logger.warning("NaN or infinite values detected in data. Raising ValueError.")
        raise ValueError("data cannot contain NaN or infinite values.")

    # Handle univariate data by reshaping to (time, 1)
    data = data.reshape(-1, 1) if data.ndim == 1 else data
    if len(data) < window_size + forecast_steps:
        raise ValueError(
            f"data length ({len(data)}) is insufficient for window_size ({window_size}) "
            f"and forecast_steps ({forecast_steps})."
        )

    try:
        n_samples = (len(data) - window_size - forecast_steps + 1) // step
        logger.debug(
            f"Creating sliding window: data_length={len(data)}, window_size={window_size}, "
            f"forecast_steps={forecast_steps}, step={step}, n_samples={n_samples}"
        )

        # Pre-allocate arrays for efficiency
        features = data.shape[1]
        X = np.zeros((n_samples, window_size, features))
        y = np.zeros((n_samples, forecast_steps, features))

        for i in range(0, len(data) - window_size - forecast_steps + 1, step):
            idx = i // step
            X[idx] = data[i : i + window_size]
            y[idx] = data[i + window_size : i + window_size + forecast_steps]

        return X, y
    except Exception as e:
        logger.error(f"Sliding window creation failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Sliding window creation failed: {str(e)}")
