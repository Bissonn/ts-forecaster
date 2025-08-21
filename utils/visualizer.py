"""Module for visualizing time series forecasting results.

This module provides the Visualizer class with methods to create and save plots for actual vs.
predicted values and cumulative prediction errors, supporting evaluation of forecasting models.
"""

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

logger = logging.getLogger(__name__)


class Visualizer:
    """Class for visualizing time series forecasting results and errors."""

    @staticmethod
    def plot_predictions(
        dataset_name: str,
        model_name: str,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        columns: List[str],
        forecast_steps: int,
    ) -> None:
        """
        Plot actual and predicted values for each feature in the time series.

        Args:
            dataset_name: Name of the dataset for organizing output files.
            model_name: Name of the forecasting model.
            test_data: DataFrame containing actual values with datetime index.
            predictions: Array of predicted values with shape (forecast_steps, num_features).
            columns: List of feature names corresponding to test_data columns.
            forecast_steps: Number of steps to forecast and plot.

        Raises:
            ValueError: If inputs are invalid (e.g., empty data, mismatched shapes, invalid forecast_steps).
            RuntimeError: If plot saving fails due to I/O errors.
        """
        if test_data.empty:
            raise ValueError("test_data cannot be empty.")
        if not isinstance(test_data.index, pd.DatetimeIndex):
            raise ValueError("test_data must have a datetime index.")
        if not isinstance(predictions, np.ndarray):
            raise ValueError("predictions must be a numpy array.")
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be positive.")
        if len(test_data) < forecast_steps:
            raise ValueError(f"test_data has {len(test_data)} rows, but forecast_steps is {forecast_steps}.")
        if predictions.shape[0] < forecast_steps:
            raise ValueError(f"predictions has {predictions.shape[0]} rows, but forecast_steps is {forecast_steps}.")
        if len(columns) != test_data.shape[1] or (predictions.ndim > 1 and len(columns) != predictions.shape[1]):
            raise ValueError("Number of columns must match test_data and predictions feature dimensions.")
        if np.any(np.isnan(test_data.values)) or np.any(np.isnan(predictions)):
            logger.warning("NaN values detected in inputs. Raising ValueError.")
            raise ValueError("test_data and predictions cannot contain NaN values.")

        num_cols = min(test_data.shape[1], predictions.shape[1] if predictions.ndim > 1 else 1)
        output_dir = f"results/plots/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        try:
            for i in range(num_cols):
                plt.figure(figsize=(10, 6))
                actual_values = test_data.iloc[:forecast_steps, i].values
                predicted_values = predictions[:forecast_steps, i] if predictions.ndim > 1 else predictions[:forecast_steps]
                plt.plot(
                    test_data.index[:forecast_steps],
                    actual_values,
                    label="Actual",
                    marker="o",
                )
                plt.plot(
                    test_data.index[:forecast_steps],
                    predicted_values,
                    label="Predicted",
                    marker="x",
                )
                plt.title(f"{dataset_name} - {columns[i]} - {model_name} (Horizon {forecast_steps})")
                plt.xlabel("Date")
                plt.ylabel(columns[i])
                plt.legend()
                plt.grid(True)
                output_path = f"{output_dir}/{columns[i]}_{model_name}_predictions.png"
                plt.savefig(output_path)
                plt.close()
                logger.info(f"Saved prediction plot for {columns[i]} ({model_name}, {dataset_name}) to {output_path}")
        except OSError as e:
            logger.error(f"Failed to save prediction plot: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to save prediction plot: {str(e)}")

    @staticmethod
    def plot_error_accumulation(
        dataset_name: str,
        model_name: str,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        columns: List[str],
        forecast_steps: int,
    ) -> None:
        """
        Plot cumulative prediction errors for each feature in the time series.

        Args:
            dataset_name: Name of the dataset for organizing output files.
            model_name: Name of the forecasting model.
            test_data: DataFrame containing actual values with datetime index.
            predictions: Array of predicted values with shape (forecast_steps, num_features).
            columns: List of feature names corresponding to test_data columns.
            forecast_steps: Number of steps to forecast and plot.

        Raises:
            ValueError: If inputs are invalid (e.g., empty data, mismatched shapes, invalid forecast_steps).
            RuntimeError: If plot saving fails due to I/O errors.
        """
        if test_data.empty:
            raise ValueError("test_data cannot be empty.")
        if not isinstance(test_data.index, pd.DatetimeIndex):
            raise ValueError("test_data must have a datetime index.")
        if not isinstance(predictions, np.ndarray):
            raise ValueError("predictions must be a numpy array.")
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be positive.")
        if len(test_data) < forecast_steps:
            raise ValueError(f"test_data has {len(test_data)} rows, but forecast_steps is {forecast_steps}.")
        if predictions.shape[0] < forecast_steps:
            raise ValueError(f"predictions has {predictions.shape[0]} rows, but forecast_steps is {forecast_steps}.")
        if len(columns) != test_data.shape[1] or (predictions.ndim > 1 and len(columns) != predictions.shape[1]):
            raise ValueError("Number of columns must match test_data and predictions feature dimensions.")
        if np.any(np.isnan(test_data.values)) or np.any(np.isnan(predictions)):
            logger.warning("NaN values detected in inputs. Raising ValueError.")
            raise ValueError("test_data and predictions cannot contain NaN values.")

        num_cols = min(test_data.shape[1], predictions.shape[1] if predictions.ndim > 1 else 1)
        output_dir = f"results/plots/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        try:
            for i in range(num_cols):
                plt.figure(figsize=(10, 6))
                actual_values = test_data.iloc[:forecast_steps, i].values
                predicted_values = predictions[:forecast_steps, i] if predictions.ndim > 1 else predictions[:forecast_steps]
                errors = np.abs(actual_values - predicted_values)
                if np.any(np.isinf(errors)):
                    logger.warning(f"Infinite errors detected for {columns[i]}. Raising ValueError.")
                    raise ValueError(f"Infinite errors detected for {columns[i]}.")
                cum_errors = np.cumsum(errors)
                plt.plot(test_data.index[:forecast_steps], cum_errors, label="Cumulative Error", marker="o")
                plt.title(f"{dataset_name} - {columns[i]} - {model_name} Error Accumulation (Horizon {forecast_steps})")
                plt.xlabel("Date")
                plt.ylabel("Cumulative Error")
                plt.legend()
                plt.grid(True)
                output_path = f"{output_dir}/{columns[i]}_{model_name}_error_accumulation.png"
                plt.savefig(output_path)
                plt.close()
                logger.info(f"Saved error accumulation plot for {columns[i]} ({model_name}, {dataset_name}) to {output_path}")
        except OSError as e:
            logger.error(f"Failed to save error accumulation plot: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to save error accumulation plot: {str(e)}")
