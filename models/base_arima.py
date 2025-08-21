"""Base module for ARIMA/SARIMA time series forecasting models.

This module defines the ARIMABaseForecaster class, which implements ARIMA/SARIMA models
using the statsmodels library, extending the StatTSForecaster base class.
"""

import logging
from typing import Dict, Optional, Set, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from models.base import StatTSForecaster

logger = logging.getLogger(__name__)


class ARIMABaseForecaster(StatTSForecaster):
    """Base class for ARIMA/SARIMA time series forecasting models."""

    is_univariate: bool = True

    def __init__(self, model_params: Dict[str, Any], num_features: int, forecast_steps: int, seasonal: bool = False) -> None:
        """
        Initialize the ARIMA/SARIMA forecaster.

        Args:
            model_params: Model-specific parameters (e.g., p, d, q for ARIMA; P, D, Q, seasonal_period for SARIMA).
            num_features: Number of features in the time series data (must be 1 for univariate models).
            forecast_steps: Number of steps to forecast.
            seasonal: Whether to use SARIMA (True) or ARIMA (False). Defaults to False.

        Raises:
            ValueError: If num_features is not 1 for univariate models or if model parameters are invalid.
        """
        if self.is_univariate and num_features != 1:
            raise ValueError("Univariate ARIMA/SARIMA models require num_features=1.")
        super().__init__(model_params, num_features, forecast_steps)
        self.seasonal = seasonal
        self._validate_model_params()
        logger.info(f"Initialized {self.__class__.__name__} with seasonal={seasonal}")

    def _validate_model_params(self) -> None:
        """
        Validate ARIMA/SARIMA model parameters.

        Raises:
            ValueError: If model parameters are invalid (e.g., negative orders).
        """
        required_params = ["p", "d", "q"]
        seasonal_params = ["P", "D", "Q", "seasonal_period"] if self.seasonal else []

        for param in required_params + seasonal_params:
            if param not in self.model_params:
                raise ValueError(f"Missing required parameter: {param}")
            if not isinstance(self.model_params[param], int) or self.model_params[param] < 0:
                raise ValueError(f"Parameter {param} must be a non-negative integer.")

        if self.seasonal and self.model_params["seasonal_period"] <= 0:
            raise ValueError("seasonal_period must be positive for SARIMA models.")

    def fit(self, train_series: pd.DataFrame, val_series: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the SARIMAX model to the training data.

        Args:
            train_series: Training data (single column for univariate models).
            val_series: Validation data (ignored for ARIMA/SARIMA). Defaults to None.

        Raises:
            ValueError: If train_series is invalid or model parameters are invalid.
            RuntimeError: If SARIMAX model fitting fails.
        """
        if self.is_univariate and train_series.shape[1] != 1:
            raise ValueError("Univariate ARIMA/SARIMA models require a single-column train_series.")

        self._validate_model_specific_inputs(train_series, val_series, self.forecast_steps)

        try:
            train_processed, _, _ = self.preprocessor.apply_transforms(
                train_series, pd.DataFrame(), pd.DataFrame(), train_series
            )
            train_data = train_processed.iloc[:, 0] if train_processed.shape[1] == 1 else train_processed

            order = (
                self.model_params.get("p", 1),
                self.model_params.get("d", 1),
                self.model_params.get("q", 1),
            )
            seasonal_order = (
                self.model_params.get("P", 0),
                self.model_params.get("D", 0),
                self.model_params.get("Q", 0),
                self.model_params.get("seasonal_period", 0),
            ) if self.seasonal else (0, 0, 0, 0)

            logger.info(f"Fitting SARIMAX with order={order}, seasonal_order={seasonal_order}")

            model = SARIMAX(
                endog=train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            self.model = model.fit(disp=False)
            self.last_fit_timestamp = train_data.index[-1]
            self.fitted = True
            logger.info("SARIMAX model fitted successfully")

        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to fit SARIMAX model: {str(e)}", exc_info=True)
            raise RuntimeError(f"SARIMAX fitting failed: {str(e)}")

    def predict(self, input_data: Optional[pd.DataFrame] = None, forecast_steps: Optional[int] = None) -> pd.DataFrame:
        """
        Generate predictions for the specified horizon.

        For ARIMA/SARIMA models, predictions are based on the fitted model's history, and input_data is ignored.

        Args:
            input_data: Input data (ignored for ARIMA/SARIMA). Defaults to None.
            forecast_steps: Number of steps to forecast. Defaults to self.forecast_steps.

        Returns:
            Predicted values in the original scale.

        Raises:
            ValueError: If model is not fitted or forecast_steps is invalid.
            RuntimeError: If prediction fails.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting.")

        steps = forecast_steps if forecast_steps is not None else self.forecast_steps
        if steps < 1:
            raise ValueError("forecast_steps must be positive.")

        try:
            predictions_proc = self.model.forecast(steps=steps)

            if predictions_proc is None or (hasattr(predictions_proc, "__len__") and len(predictions_proc) == 0):
                logger.warning(f"SARIMAX forecast returned empty or None for {self.__class__.__name__}. Returning NaNs.")
                return pd.DataFrame(np.full((steps, self.num_features), np.nan), columns=self.preprocessor.columns)

            predictions_proc_np = (
                predictions_proc.values if isinstance(predictions_proc, (pd.Series, pd.DataFrame))
                else predictions_proc
            )
            if predictions_proc_np.size < steps:
                logger.warning(f"SARIMAX forecast returned insufficient data for {self.__class__.__name__}. Filling with NaNs.")
                predictions_proc_np = np.full((steps, self.num_features), np.nan)
            else:
                predictions_proc_np = predictions_proc_np.reshape(-1, self.num_features)

            predictions_original_df = self.preprocessor.inverse_transforms(
                predictions_proc_np,
                start_after=self.last_fit_timestamp
            )
            logger.info(f"Generated {steps} predictions for {self.__class__.__name__}")
            return predictions_original_df

        except (ValueError, RuntimeError) as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"SARIMAX prediction failed: {str(e)}")

    def get_valid_params(self) -> Set[str]:
        """
        Get the set of valid parameter names for the ARIMA/SARIMA model.

        Returns:
            Set of valid parameter names.
        """
        valid_params = {"p", "d", "q", "window_size", "preprocessing", "n_trials"}
        if self.seasonal:
            valid_params.update({"P", "D", "Q", "seasonal_period"})
        return valid_params

    def _validate_model_specific_inputs(
        self, train_series: pd.DataFrame, val_series: Optional[pd.DataFrame] = None, forecast_steps: Optional[int] = None
    ) -> None:
        """
        Validate inputs specific to ARIMA/SARIMA models.

        Args:
            train_series: Training data.
            val_series: Validation data (optional, ignored). Defaults to None.
            forecast_steps: Forecast steps (optional). Defaults to None.

        Raises:
            ValueError: If inputs are invalid.
        """
        window_size = self.model_params.get("window_size", 0)
        steps = forecast_steps if forecast_steps is not None else self.forecast_steps
        if len(train_series) < window_size + steps:
            raise ValueError("Training series too short for specified window_size and forecast_steps.")
