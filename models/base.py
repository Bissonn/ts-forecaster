"""Base module for time series forecasting models.

This module defines abstract base classes for statistical and neural network-based time series forecasting models.
It provides a framework for model initialization, hyperparameter optimization, data preparation, fitting, and prediction.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.data_utils import create_sliding_window
from utils.dataset import TimeSeriesDataset
from utils.hyperopt.grid_params import generate_grid_params
from utils.hyperopt.optuna_params import generate_optuna_params
from utils.hyperopt.random_params import generate_random_params
from utils.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class TSForecaster(ABC):
    """Abstract base class for time series forecasting models."""

    # Class attribute indicating if the model is univariate
    is_univariate: bool = False

    def __init__(self, model_params: Dict[str, Any], num_features: int, forecast_steps: int) -> None:
        """
        Initialize the forecaster with model-specific parameters.

        Args:
            model_params: Model-specific parameters (e.g., p, d, q for ARIMA).
            num_features: Number of features in the time series data (columns).
            forecast_steps: Number of steps to forecast.

        Raises:
            ValueError: If model_params is not a dictionary or forecast_steps is not positive.
        """
        if not isinstance(model_params, dict):
            raise ValueError("model_params must be a dictionary.")
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be positive.")
        if num_features < 1:
            raise ValueError("num_features must be positive.")

        self.model_params = model_params
        self.num_features = num_features
        self.forecast_steps = forecast_steps
        self.optimize = model_params.get("optimize", False)
        self.optimization_config = model_params.get("optimization", {"method": "random", "params": {}})
        self.model = None
        self.fitted = False
        self.last_fit_timestamp = None
        self.preprocessor = Preprocessor(self.model_params.get("preprocessing", {}))
        logger.info(f"Initialized {self.__class__.__name__} with params: {model_params}")

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """
        Calculate Mean Squared Error (MSE) between true and predicted values.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Mean Squared Error value. Returns inf if alignment fails or inputs are empty.

        Raises:
            ValueError: If inputs are empty or have incompatible shapes.
        """
        if y_true.empty or y_pred.empty:
            logger.warning("Empty input DataFrames provided for evaluation.")
            return float("inf")

        y_true_aligned, y_pred_aligned = y_true.align(y_pred, join="inner", axis=1, copy=False)
        if y_true_aligned.empty:
            logger.warning("Could not align y_true and y_pred for evaluation.")
            return float("inf")

        mse = np.mean((y_true_aligned.values - y_pred_aligned.values) ** 2)
        return float(mse)

    @abstractmethod
    def _fit_and_evaluate_fold(
        self, train_fold: pd.DataFrame, validation_params: Dict[str, Any], is_final_fit: bool = False
    ) -> float:
        """
        Fit the model on a training fold and evaluate on validation data.

        Args:
            train_fold: Training data fold.
            validation_params: Validation configuration (e.g., window_size, n_folds).
            is_final_fit: Whether this is the final fit (no validation). Defaults to False.

        Returns:
            Validation loss (e.g., MSE) in the original scale, or 0.0 for final fit.

        Raises:
            ValueError: If data or parameters are invalid.
        """
        pass

    def optimize_hyperparameters(
            self, dataset: TimeSeriesDataset, model_config: Dict[str, Any], validation_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float]:
        logger.info("Starting optimize_hyperparameters")
        opt_params = model_config.get("optimization", {"method": "grid", "params": {}})
        method = opt_params.get("method", "grid")
        n_folds = validation_params.get("n_folds")
        max_window_size = validation_params.get("max_window_size")

        best_params = {
            key: value
            for key, value in model_config.items()
            if key not in ["optimization", "optimize", "method", "params", "auto_arima"]
        }
        best_loss = float("inf")

        logger.info(f"Generating folds with n_folds={n_folds}, max_window_size={max_window_size}")
        cv_folds = dataset.generate_walk_forward_folds(max_window_size, n_folds)
        logger.info(f"Generated {len(cv_folds)} folds")
        if not cv_folds:
            logger.error("No folds generated")
            raise ValueError("Could not generate any folds. Check data and validation parameters.")

        param_space = opt_params.get("params", {})
        candidates: List[Dict[str, Any]] = []
        if method == "optuna":
            candidates = generate_optuna_params(param_space=param_space, n_trials=param_space.get("n_trials", 10))
        elif method == "grid":
            candidates = generate_grid_params(param_space=param_space, n_trials=param_space.get("n_trials", 5))
        elif method == "random":
            candidates = generate_random_params(param_space=param_space, n_trials=param_space.get("n_trials", 5))
        else:
            logger.error(f"Invalid optimization method: {method}")
            raise ValueError(f"Invalid optimization method: {method}")

        logger.info(f"Generated {len(candidates)} candidates: {candidates}")
        candidates = self.filter_candidates(candidates, best_params)
        logger.info(f"Filtered to {len(candidates)} candidates")
        if not candidates:
            logger.error("No valid candidates after filtering")
            raise ValueError("No valid parameter combinations after applying constraints.")

        for params in candidates:
            current_params = best_params.copy()
            current_params.update(params)
            fold_losses = []
            logger.info(f"Evaluating params: {current_params}")

            for train_fold in cv_folds:
                try:
                    model_instance = self.__class__(current_params, self.num_features, self.forecast_steps)
                    loss = model_instance._fit_and_evaluate_fold(train_fold, validation_params)
                    fold_losses.append(loss)
                    logger.info(f"Fold loss: {loss}")
                except Exception as e:
                    logger.warning(f"Evaluation failed for params {current_params} on a fold: {e}", exc_info=True)
                    fold_losses.append(float("inf"))

            avg_loss = np.nanmean([l if np.isfinite(l) else np.nan for l in fold_losses])
            logger.info(f"Average loss for params {current_params}: {avg_loss}")

            if np.isfinite(avg_loss) and avg_loss < best_loss:
                best_loss = avg_loss
                best_params.update(params)
                logger.info(f"New best score: {best_loss:.6f} with params: {best_params}")

        if not np.isfinite(best_loss):
            logger.error("No valid parameter combinations found")
            raise ValueError("No valid parameter combinations found during optimization.")

        logger.info(f"Returning best_params: {best_params}, best_loss: {best_loss}")
        return best_params, best_loss

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """
        Fit the model to the training data.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> pd.DataFrame:
        """
        Generate predictions for the specified horizon.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Predictions in a 2D DataFrame.
        """
        pass

    def get_valid_params(self) -> set:
        """
        Get the set of valid parameter names for the model.

        Returns:
            Set of valid parameter names.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement get_valid_params.")

    def filter_candidates(self, candidates: List[Dict[str, Any]], best_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter candidate parameter combinations based on model-specific constraints.

        Args:
            candidates: List of parameter combinations.
            best_params: Fixed parameters.

        Returns:
            Filtered parameter combinations.
        """
        return candidates  # Default: no filtering

    def prepare_data(self, dataset: TimeSeriesDataset) -> List[TimeSeriesDataset]:
        """
        Prepare data for the model.

        Args:
            dataset: Input dataset.

        Returns:
            List of processed datasets.

        Raises:
            ValueError: If dataset is invalid.
        """
        return [dataset]

    def _save_model(self, model_instance: Any) -> None:
        """
        Save the model in an appropriate format.

        Args:
            model_instance: The model instance to save.

        Note:
            Placeholder method to be implemented by subclasses.
        """
        pass

    def _validate_model_specific_inputs(self, *args) -> None:
        """
        Validate model-specific inputs.

        Args:
            *args: Variable length argument list.

        Note:
            Placeholder method to be implemented by subclasses if needed.
        """
        pass


class StatTSForecaster(TSForecaster, ABC):
    """Abstract base class for statistical time series forecasting models."""

    def _fit_and_evaluate_fold(
        self, train_fold: pd.DataFrame, validation_params: Dict[str, Any], is_final_fit: bool = False
    ) -> float:
        """
        Fit and evaluate the model on a single fold for statistical models (hold-out split).
n
        Args:
            train_fold: Training data fold.
            validation_params: Validation configuration (e.g., window_size).
            is_final_fit: Whether this is the final fit (no validation). Defaults to False.

        Returns:
            Validation loss (MSE) in the original scale, or 0.0 for final fit.

        Raises:
            ValueError: If train_fold is too small for fitting and validation.
        """
        if is_final_fit:
            self.fit(train_fold)
            return 0.0

        holdout_size = self.forecast_steps
        min_fit_size = self.model_params.get("window_size", holdout_size)

        if len(train_fold) <= min_fit_size + holdout_size:
            raise ValueError(
                f"Train fold too small ({len(train_fold)}) for min fit size ({min_fit_size}) "
                f"and hold-out size ({holdout_size})."
            )

        fit_set_raw = train_fold.iloc[:-holdout_size]
        hold_out_set_raw = train_fold.iloc[-holdout_size:]

        # Pass raw data to the fit method. The fit method will handle preprocessing.
        self.fit(fit_set_raw)

        # Generate predictions and evaluate
        predictions_original = self.predict(forecast_steps=holdout_size)
        return self.evaluate(hold_out_set_raw, predictions_original)

    def prepare_data(self, dataset: TimeSeriesDataset) -> List[TimeSeriesDataset]:
        """
        Prepare data for the model, creating separate datasets for each column in univariate mode
        or returning the original dataset in multivariate mode.

        Args:
            dataset: Input dataset.

        Returns:
            List of datasets (one per column in univariate mode, or one in multivariate mode).

        Raises:
            ValueError: If development_data or test_data is not set in the dataset.
        """
        if dataset.development_data is None or dataset.test_data is None:
            raise ValueError("Dataset must have development_data and test_data. Call dataset.split_data().")

        if self.is_univariate:
            datasets = []
            for col in dataset.series.columns:
                logger.info(f"Preparing dataset for column: {col}")
                # Build a per-column DataFrame and preserve the original datetime index.
                # If TimeSeriesDataset expects a 'date' column, materialize it from the index
                # so that _prepare_data() can set the index without warnings.
                per_col_df = dataset.series[[col]].copy()
                
                # Only add 'date' column if we have a DatetimeIndex (or PeriodIndex) and
                # the source dataset uses the default date column name.
                date_col_name = getattr(dataset, "date_column", "date")
                if (
                    isinstance(per_col_df.index, (pd.DatetimeIndex, pd.PeriodIndex))
                    and date_col_name not in per_col_df.columns
                ):
                    # Materialize date column from index; keep it first for readability.
                    per_col_df = per_col_df.copy()
                    per_col_df[date_col_name] = per_col_df.index
                    # Reorder columns to have date first, then the value column.
                    per_col_df = per_col_df[[date_col_name, col]]
                    
                local_dataset = TimeSeriesDataset(
                    dataset_name=f"{dataset.name}_{col}",
                    config=dataset.config,
                    data=per_col_df,
                    columns=[col],
                    freq=dataset.freq,
                )
                local_dataset.development_data = dataset.development_data[[col]]
                local_dataset.test_data = dataset.test_data[[col]]
                datasets.append(local_dataset)
            return datasets
        return [dataset]

    @abstractmethod
    def fit(self, train_series: pd.DataFrame, val_series: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the statistical model to the training data.

        Args:
            train_series: Training data.
            val_series: Validation data (optional). Defaults to None.
        """
        pass


class NeuralTSForecaster(TSForecaster, ABC):
    """Abstract base class for neural network-based time series forecasting models."""

    def __init__(self, model_params: Dict[str, Any], num_features: int, forecast_steps: int) -> None:
        """
        Initialize the neural forecaster with model-specific parameters.

        Args:
            model_params: Model-specific parameters.
            num_features: Number of features in the time series data.
            forecast_steps: Number of steps to forecast.

        Raises:
            ValueError: If num_features or forecast_steps are invalid.
        """
        if num_features < 1:
            raise ValueError("num_features must be positive.")
        super().__init__(model_params, num_features, forecast_steps)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def _get_y_window_steps(self) -> int:
        """
        Get the number of target steps for creating sliding windows.

        Returns:
            Number of steps (defaults to forecast_steps, can be overridden by subclasses).
        """
        return self.forecast_steps

    def _fit_and_evaluate_fold(
        self, train_fold: pd.DataFrame, validation_params: Dict[str, Any], is_final_fit: bool = False
    ) -> float:
        """
        Fit and evaluate the model on a single fold for neural models.

        Args:
            train_fold: Training data fold.
            validation_params: Validation configuration.
            is_final_fit: Whether this is the final fit (no validation). Defaults to False.

        Returns:
            Validation loss, or 0.0 for final fit.

        Raises:
            ValueError: If window_size is not defined or data is insufficient.
        """
        if is_final_fit:
            self.fit(train_fold, is_final_fit=True)
            return 0.0

        early_stopping_val_percentage = validation_params.get("early_stopping_validation_percentage")
        return self.fit(train_fold, is_final_fit=False, early_stopping_validation_percentage=early_stopping_val_percentage)

    def fit(
        self,
        train_series: pd.DataFrame,
        is_final_fit: bool = False,
        early_stopping_validation_percentage: Optional[float] = None,
    ) -> float:
        """
        Fit the neural model to the training data.

        Args:
            train_series: Training data.
            is_final_fit: Whether this is the final fit (no validation). Defaults to False.
            early_stopping_validation_percentage: Percentage of data for validation. Defaults to None.

        Returns:
            Validation loss (inf if training fails).

        Raises:
            ValueError: If window_size is not defined or data is insufficient.
        """
        logger.info(f"[{self.__class__.__name__}] Starting fit process...")
        window_size = self.model_params.get("window_size")
        if not window_size:
            raise ValueError("'window_size' must be defined in model parameters.")

        # Preprocess training data
        train_proc, _, _ = self.preprocessor.apply_transforms(
            train_series, pd.DataFrame(), pd.DataFrame(), train_series
        )

        # Generate sliding windows
        steps_for_window_creation = self._get_y_window_steps()
        X_all_windows, y_all_windows = create_sliding_window(train_proc.values, window_size, steps_for_window_creation)

        # Split into training and validation sets
        X_train, y_train = np.array([]), np.array([])
        X_val, y_val = np.array([]), np.array([])

        if is_final_fit:
            X_train, y_train = X_all_windows, y_all_windows
            logger.info(f"[{self.__class__.__name__}] Final training on {X_train.shape[0]} windows.")
        else:
            total_windows = X_all_windows.shape[0]
            min_absolute_val_windows = self.forecast_steps
            num_val_windows = (
                max(
                    min_absolute_val_windows,
                    int(total_windows * (early_stopping_validation_percentage / 100.0)),
                )
                if early_stopping_validation_percentage is not None
                else min_absolute_val_windows
            )

            logger.info(
                f"[{self.__class__.__name__}] Total windows: {total_windows}, Validation windows: {num_val_windows}"
            )

            if total_windows > num_val_windows:
                X_train = X_all_windows[:-num_val_windows]
                y_train = y_all_windows[:-num_val_windows]
                X_val = X_all_windows[-num_val_windows:]
                y_val = y_all_windows[-num_val_windows:]
            else:
                logger.warning(
                    f"[{self.__class__.__name__}] Not enough windows ({total_windows}) for validation "
                    f"(needed: {num_val_windows}). Training on full fold without validation."
                )
                X_train, y_train = X_all_windows, y_all_windows

        if X_train.shape[0] == 0:
            logger.error(f"[{self.__class__.__name__}] No training windows generated.")
            self.fitted = False
            return float("inf")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = (
            torch.FloatTensor(X_val).to(self.device) if X_val.shape[0] > 0 else torch.empty(0).to(self.device)
        )
        y_val_tensor = (
            torch.FloatTensor(y_val).to(self.device) if y_val.shape[0] > 0 else torch.empty(0).to(self.device)
        )

        if self.model is None:
            raise NotImplementedError("Model (self.model) must be initialized in the subclass.")

        self.model.to(self.device)
        trained_model_instance = self._train_model(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
        self.model = trained_model_instance
        val_loss = getattr(trained_model_instance, "best_val_loss", float("inf"))

        self.fitted = True
        logger.info(f"[{self.__class__.__name__}] Fit process completed with validation loss: {val_loss:.6f}")
        return val_loss

    def _internal_predict(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Internal prediction engine for neural models.

        This default implementation performs a single forward pass of the model.
        Subclasses like iterative forecasters can override this for more complex, step-by-step prediction logic.

        Args:
            input_tensor: Input data tensor.

        Returns:
            Predictions as a NumPy array of shape (batch_size, forecast_steps, num_features).

        Raises:
            ValueError: If model is not initialized or fitted, or input shape is invalid.
        """
        if self.model is None or not self.fitted:
            raise ValueError("Model must be initialized and fitted before prediction.")

        if input_tensor.dim() != 3 or input_tensor.shape[-1] != self.num_features:
            raise ValueError(f"Expected input tensor of shape (batch_size, window_size, {self.num_features}), "
                             f"but got {input_tensor.shape}")

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            return output.cpu().numpy()

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for input data, returning results in the original scale.

        Args:
            input_data: Input data for prediction.

        Returns:
            Predicted values.

        Raises:
            ValueError: If model is not fitted or input data is invalid.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting.")
        if input_data.empty:
            raise ValueError("Input data cannot be empty.")

        logger.info(f"[{self.__class__.__name__}] Starting prediction...")
        start_after_ts = input_data.index[-1]
        input_proc = self.preprocessor.transform(input_data)
        input_tensor = torch.FloatTensor(input_proc.values).unsqueeze(0).to(self.device)

        predictions_proc_np = self._internal_predict(input_tensor)

        # Adapt the output shape. The internal predict method returns a 3D array (batch, steps, features).
        # For a single prediction from a DataFrame (batch_size=1), we squeeze it to the 2D array
        # expected by the preprocessor's inverse transform.
        if predictions_proc_np.ndim == 3 and predictions_proc_np.shape[0] == 1:
            predictions_proc_np = predictions_proc_np.squeeze(0)

        predictions_original_df = self.preprocessor.inverse_transforms(
            predictions_proc_np,
            start_after=start_after_ts
        )

        logger.info(f"[{self.__class__.__name__}] Prediction completed.")
        return predictions_original_df

    def _validate_model_specific_inputs(
        self, train_series: pd.DataFrame, val_series: Optional[pd.DataFrame] = None, forecast_steps: Optional[int] = None
    ) -> None:
        """
        Validate inputs specific to neural models.

        Args:
            train_series: Training data.
            val_series: Validation data (optional). Defaults to None.
            forecast_steps: Forecast steps (optional). Defaults to None.

        Raises:
            ValueError: If inputs are invalid.
        """
        window_size = self.model_params.get("window_size", 0)
        if len(train_series) < window_size + self.forecast_steps:
            raise ValueError("Training series too short for specified window_size and forecast_steps.")

    def prepare_data(self, dataset: TimeSeriesDataset) -> List[TimeSeriesDataset]:
        """
        Prepare data for neural models.

        Args:
            dataset: Input dataset.

        Returns:
            List containing the input dataset.
        """
        return [dataset]

    @abstractmethod
    def _train_model(self, *args, **kwargs) -> Any:
        """
        Run the training loop for the neural model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Trained model instance.
        """
        pass
