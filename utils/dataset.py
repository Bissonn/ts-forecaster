"""Module for managing time series datasets in the forecasting framework.

This module provides the TimeSeriesDataset class to load, preprocess, split, and manage time series data
for training and evaluating forecasting models like ARIMA, VAR, LSTM, and Transformer.
"""

import logging
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd
import os

logger = logging.getLogger(__name__)


class TimeSeriesDataset:
    """Class for managing time series datasets, including loading, preprocessing, splitting, and saving results."""

    def __init__(
        self,
        dataset_name: str,
        config: Dict,
        data: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
        freq: Optional[str] = None,
        date_column: str = "date",
    ) -> None:
        """
        Initialize the TimeSeriesDataset.

        Args:
            dataset_name: Name of the dataset, used to load from config if data is None.
            config: Configuration dictionary (e.g., from config.yaml).
            data: Optional DataFrame containing time series data. If None, loads from file specified in config.
            columns: Optional list of column names to use. If None, uses config or all numeric columns.
            freq: Optional frequency of the data (e.g., 'D', 'H'). If None, uses config or infers from data.
            date_column: Name of the column containing dates. Defaults to 'date'.

        Raises:
            ValueError: If dataset_name is invalid, config is missing required keys, or columns are invalid.
            FileNotFoundError: If data is None and the file specified in config does not exist.
        """
        if not dataset_name:
            raise ValueError("dataset_name cannot be empty.")
        if data is None and (dataset_name not in config.get("datasets", {})):
            raise ValueError(f"Dataset '{dataset_name}' not found in config['datasets'].")
        if freq is not None and not isinstance(freq, str):
            raise ValueError("freq must be a string (e.g., 'D', 'H').")

        self.name = dataset_name
        self.config = config
        self.date_column = date_column
        self.path = config["datasets"][dataset_name]["path"] if data is None else None
        self.columns = columns
        self.freq = (
            freq
            if freq is not None
            else config["datasets"][dataset_name].get("freq", None)
            if data is None
            else None
        )
        self.series = data if data is not None else self._load_data()
        self.series = self._prepare_data(self.series)
        self.development_data = None
        self.test_data = None
        logger.info(f"TimeSeriesDataset '{self.name}' initialized. Columns: {self.columns}, Freq: {self.freq}")

    def _load_data(self) -> pd.DataFrame:
        """
        Load data from a CSV file specified in the config.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the data is empty or contains no numeric columns.
            RuntimeError: If CSV loading fails due to parsing errors.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        try:
            df = pd.read_csv(self.path)
            if df.empty:
                raise ValueError(f"Dataset '{self.path}' is empty.")
            if not df.select_dtypes(include=np.number).columns.tolist():
                raise ValueError(f"Dataset '{self.path}' contains no numeric columns.")
            logger.info(f"Dataset '{self.path}' loaded with {len(df)} rows.")
            return df
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse CSV file {self.path}: {str(e)}")
            raise RuntimeError(f"Failed to parse CSV file: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load data from {self.path}: {str(e)}")
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by setting a datetime index and selecting columns.

        Args:
            df: Input DataFrame to prepare.

        Returns:
            Prepared DataFrame with datetime index and selected columns.

        Raises:
            ValueError: If required columns are missing, data is empty, or contains NaNs/infinite values.
        """
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty.")

        # First, check if the DataFrame already has a DatetimeIndex.
        if isinstance(df.index, pd.DatetimeIndex):
            logger.info("DataFrame already has a DatetimeIndex. Using it.")
            if self.freq:
                df = df.asfreq(self.freq)
        # If not, check for a date column to convert.
        elif self.date_column in df.columns:
            try:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                df = df.set_index(self.date_column)
                if self.freq:
                    df = df.asfreq(self.freq)
            except ValueError as e:
                logger.error(f"Failed to convert '{self.date_column}' to datetime: {str(e)}")
                raise ValueError(f"Invalid date column: {str(e)}")
        # If neither is true, fall back to a RangeIndex.
        else:
            logger.warning(f"No DatetimeIndex or '{self.date_column}' column found. Using default RangeIndex.")
            df.index = pd.RangeIndex(len(df))

        # Infer frequency if not provided and index is datetime
        if self.freq is None and isinstance(df.index, pd.DatetimeIndex):
            inferred_freq = pd.infer_freq(df.index)
            self.freq = inferred_freq if inferred_freq else "D" # Default to 'D' if inference fails
            logger.info(f"Inferred frequency: {self.freq}")
            df = df.asfreq(self.freq)

        # Select columns
        if self.columns:
            if not all(col in df.columns for col in self.columns):
                missing_cols = [col for col in self.columns if col not in df.columns]
                raise ValueError(f"Missing columns in dataset: {missing_cols}")
            df = df[self.columns]
        else:
            # Use config columns if provided, otherwise all numeric columns
            config_columns = self.config["datasets"].get(self.name, {}).get("columns", None)
            if config_columns:
                if not all(col in df.columns for col in config_columns):
                    missing_cols = [col for col in config_columns if col not in df.columns]
                    raise ValueError(f"Missing columns specified in config: {missing_cols}")
                self.columns = config_columns
                df = df[self.columns]
            else:
                self.columns = df.select_dtypes(include=np.number).columns.tolist()
                if not self.columns:
                    raise ValueError("No numeric columns found in dataset.")
                df = df[self.columns]
                logger.info(f"No specific columns provided. Using all numeric columns: {self.columns}")

        # Check for NaNs and infinite values
        if df.isna().any().any():
            logger.warning("NaN values detected in data. Dropping NaNs.")
            df = df.dropna()
        if np.any(np.isinf(df.values)):
            raise ValueError("Data contains infinite values.")

        return df

    def split_data(self, forecast_steps: int) -> None:
        """
        Split data into development and test sets.

        Args:
            forecast_steps: Number of steps to reserve for the test set.

        Raises:
            ValueError: If dataset is empty or too short for the specified forecast_steps.
        """
        if self.series.empty:
            raise ValueError("Dataset is empty. Cannot split data.")
        if not isinstance(forecast_steps, int) or forecast_steps < 1:
            raise ValueError("forecast_steps must be a positive integer.")
        if len(self.series) <= forecast_steps:
            raise ValueError(f"Dataset is too short ({len(self.series)} rows) for {forecast_steps} forecast steps.")

        self.development_data = self.series.iloc[:-forecast_steps]
        self.test_data = self.series.iloc[-forecast_steps:]
        logger.info(
            f"Data split into development set ({len(self.development_data)} rows) and test set ({len(self.test_data)} rows)."
        )

    def generate_walk_forward_folds(self, max_window_size: int, n_folds: int) -> List[pd.DataFrame]:
        """
        Generate folds for walk-forward validation using development data.

        Args:
            max_window_size: Maximum window size for training data.
            n_folds: Number of folds to generate.

        Returns:
            List of DataFrames, each representing a training fold.

        Raises:
            ValueError: If development_data is not set, inputs are invalid, or data is insufficient for folds.
        """
        if self.development_data is None:
            raise ValueError("Data must be split into development and test sets first. Call split_data().")
        if not isinstance(max_window_size, int) or max_window_size < 1:
            raise ValueError("max_window_size must be a positive integer.")
        if not isinstance(n_folds, int) or n_folds < 1:
            raise ValueError("n_folds must be a positive integer.")

        # Use forecast_steps from config or default to 1
        experiment_config = self.config.get("experiments", [{}])[0]
        validation_config = experiment_config.get("validation_setup", {})
        fold_forecast_steps = validation_config.get("forecast_steps", 1)
        if not isinstance(fold_forecast_steps, int) or fold_forecast_steps < 1:
            raise ValueError("forecast_steps in config['experiments'][0]['validation_setup'] must be a positive integer.")

        data_len = len(self.development_data)
        required_len = max_window_size + fold_forecast_steps * n_folds
        if data_len < required_len:
            logger.warning(
                f"Development data ({data_len} rows) is too short for max_window_size ({max_window_size}) "
                f"and {n_folds} folds with forecast_steps ({fold_forecast_steps}). Required: {required_len}."
            )

        folds = []
        initial_train_len = max(data_len - (n_folds * fold_forecast_steps), max_window_size)
        if initial_train_len <= 0:
            raise ValueError(
                f"Not enough development data for walk-forward validation with {n_folds} folds "
                f"and forecast_steps {fold_forecast_steps}. Initial train length: {initial_train_len}"
            )

        try:
            for i in range(n_folds):
                current_train_len = initial_train_len + i * fold_forecast_steps
                if current_train_len > data_len:
                    logger.warning(f"Stopped generating folds at {i} due to insufficient data.")
                    break
                fold = self.development_data.iloc[:current_train_len]
                folds.append(fold)
            logger.info(f"Generated {len(folds)} walk-forward training folds.")
            return folds
        except Exception as e:
            logger.error(f"Failed to generate walk-forward folds: {str(e)}")
            raise RuntimeError(f"Failed to generate walk-forward folds: {str(e)}")

    def _save_results(
        self, predictions: np.ndarray, model_name: str, forecast_steps: int, metrics_fn: Optional[Callable] = None
    ) -> None:
        """
        Save predictions and metrics to CSV files.

        Args:
            predictions: Array of predictions with shape (forecast_steps, num_features).
            model_name: Name of the model (e.g., 'arima', 'lstm_direct').
            forecast_steps: Number of forecast steps.
            metrics_fn: Optional function to compute metrics (e.g., from metrics.py).

        Raises:
            ValueError: If predictions are invalid or model_name is empty.
            RuntimeError: If file saving fails due to I/O errors.
        """
        if not model_name:
            raise ValueError("model_name cannot be empty.")
        if not isinstance(predictions, np.ndarray):
            raise ValueError(f"Predictions must be a numpy.ndarray, got {type(predictions)}.")
        if not isinstance(forecast_steps, int) or forecast_steps < 1:
            raise ValueError("forecast_steps must be a positive integer.")
        if self.series.empty:
            raise ValueError("Dataset is empty. Cannot save results.")

        try:
            os.makedirs("results/predictions", exist_ok=True)
            os.makedirs("results/metrics", exist_ok=True)
            logger.info(
                f"Saving results for model '{model_name}' on dataset '{self.name}'. Forecast steps: {forecast_steps}."
            )

            # Handle univariate predictions
            predictions = predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions
            pred_columns = self.series.columns[:min(predictions.shape[1], len(self.series.columns))]
            available_steps = min(forecast_steps, len(predictions))
            if available_steps < forecast_steps:
                logger.warning(f"Predictions have only {available_steps} steps, expected {forecast_steps}.")

            # Create predictions DataFrame
            predictions_df = pd.DataFrame(
                predictions[:available_steps, :len(pred_columns)], columns=pred_columns
            )

            # Set index for predictions
            if self.test_data is not None and len(self.test_data) >= available_steps:
                predictions_df.index = self.test_data.index[:available_steps]
            else:
                last_dev_date = self.development_data.index[-1] if self.development_data is not None else self.series.index[-1]
                if self.freq:
                    try:
                        new_index = pd.date_range(start=last_dev_date + pd.tseries.frequencies.to_offset(self.freq), periods=available_steps, freq=self.freq)
                    except ValueError as e:
                        logger.error(f"Failed to generate date index with freq {self.freq}: {str(e)}")
                        raise ValueError(f"Invalid frequency for index generation: {str(e)}")
                else:
                    new_index = pd.RangeIndex(start=len(self.series), stop=len(self.series) + available_steps)
                predictions_df.index = new_index

            # Save predictions
            pred_output_path = f"results/predictions/{self.name}_{model_name}_predictions.csv"
            predictions_df.to_csv(pred_output_path)
            logger.info(f"Saved predictions to {pred_output_path}")

            # Calculate and save metrics if metrics_fn is provided
            if metrics_fn:
                metrics_data = []
                for col in pred_columns:
                    actual = self.test_data[col].values[:available_steps] if self.test_data is not None else []
                    pred = predictions_df[col].values
                    if len(actual) != len(pred):
                        logger.warning(
                            f"Shape mismatch for metrics: actual {len(actual)} vs pred {len(pred)} for column {col}. Skipping."
                        )
                        continue
                    try:
                        metrics = metrics_fn(actual, pred)
                        if not isinstance(metrics, dict):
                            raise ValueError(f"metrics_fn must return a dictionary, got {type(metrics)}")
                        metrics_data.append({
                            "dataset": self.name,
                            "column": col,
                            "model": model_name,
                            "horizon": forecast_steps,
                            **metrics
                        })
                    except Exception as e:
                        logger.warning(f"Failed to compute metrics for column {col}: {str(e)}")
                        continue
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_output_path = f"results/metrics/{self.name}_{model_name}_metrics.csv"
                    metrics_df.to_csv(metrics_output_path, index=False)
                    logger.info(f"Saved metrics to {metrics_output_path}")
                else:
                    logger.warning("No metrics calculated or saved due to data issues.")

        except OSError as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise RuntimeError(f"Failed to save results: {str(e)}")

    def get_column_data(self, column: str) -> pd.Series:
        """
        Return the data for a specific column.

        Args:
            column: Name of the column to retrieve.

        Returns:
            Series containing the data for the specified column.

        Raises:
            ValueError: If column is not found or dataset is empty.
        """
        if self.series.empty:
            raise ValueError("Dataset is empty.")
        if column not in self.series.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")
        return self.series[column]

    def get_development_data(self) -> pd.DataFrame:
        """
        Return the development dataset.

        Returns:
            DataFrame containing the development data.

        Raises:
            ValueError: If development_data is not set (call split_data() first).
        """
        if self.development_data is None:
            raise ValueError("Data has not been split yet. Call split_data() first.")
        return self.development_data

    def get_test_data(self) -> pd.DataFrame:
        """
        Return the test dataset.

        Returns:
            DataFrame containing the test data.

        Raises:
            ValueError: If test_data is not set (call split_data() first).
        """
        if self.test_data is None:
            raise ValueError("Data has not been split yet. Call split_data() first.")
        return self.test_data
