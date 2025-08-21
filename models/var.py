"""Module for Vector Autoregression (VAR) time series forecasting model.

This module defines the VARForecaster class, which implements a multivariate VAR model
using the statsmodels library, extending the StatTSForecaster base class.
"""

import logging
from typing import Dict, Optional, Set, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

from models.base import StatTSForecaster
from models.model_registry import register_model

logger = logging.getLogger(__name__)


@register_model("var", is_univariate=False)
class VARForecaster(StatTSForecaster):
    """Implementation of the Vector Autoregression (VAR) forecasting model."""

    def __init__(self, model_params: Dict[str, Any], num_features: int, forecast_steps: int) -> None:
        """
        Initialize the VAR forecaster.

        Args:
            model_params: Model-specific parameters (e.g., max_lags, ic).
            num_features: Number of features in the time series data (must be > 1 for VAR).
            forecast_steps: Number of steps to forecast.

        Raises:
            ValueError: If num_features is less than 2 or model parameters are invalid.
        """
        if num_features < 2:
            raise ValueError("VAR models require at least two features (num_features >= 2).")
        super().__init__(model_params, num_features, forecast_steps)
        self.max_lags = model_params.get("max_lags", 1)
        self.ic = model_params.get("ic", None)
        self._validate_model_params()
        logger.info(f"Initialized {self.__class__.__name__} with max_lags={self.max_lags}, ic={self.ic}")

    def _validate_model_params(self) -> None:
        """
        Validate VAR model parameters.

        Raises:
            ValueError: If max_lags is invalid or ic is not a valid information criterion.
        """
        if not isinstance(self.max_lags, int) or self.max_lags < 1:
            raise ValueError("max_lags must be a positive integer.")
        if self.ic is not None and self.ic not in {"aic", "bic", "hqic", "fpe"}:
            raise ValueError("ic must be one of 'aic', 'bic', 'hqic', 'fpe', or None.")

    def _validate_model_specific_inputs(
        self, train_series: pd.DataFrame, val_series: Optional[pd.DataFrame] = None, forecast_steps: Optional[int] = None
    ) -> None:
        """
        Validate inputs specific to VAR models.

        Args:
            train_series: Training data.
            val_series: Validation data (optional, ignored). Defaults to None.
            forecast_steps: Forecast steps (optional). Defaults to None.

        Raises:
            ValueError: If train_series has insufficient data or columns.
        """
        steps = forecast_steps if forecast_steps is not None else self.forecast_steps
        if train_series.shape[1] < 2:
            raise ValueError("VAR models require at least two columns in train_series.")
        if len(train_series) < self.max_lags + steps:
            raise ValueError("Training series too short for specified max_lags and forecast_steps.")

    def fit(self, train_series: pd.DataFrame, val_series: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the VAR model to the training data.

        Args:
            train_series: Training data (must have at least two columns).
            val_series: Validation data (ignored for VAR). Defaults to None.

        Raises:
            ValueError: If train_series is invalid or model parameters are invalid.
            RuntimeError: If VAR model fitting fails.
        """
        self._validate_model_specific_inputs(train_series, val_series, self.forecast_steps)

        try:
            train_processed, _, _ = self.preprocessor.apply_transforms(
                train_series, pd.DataFrame(), pd.DataFrame(), train_series
            )
            if train_processed.shape[1] < 2:
                raise ValueError("Processed training data must have at least two columns for VAR.")

            logger.info(f"Fitting VAR with max_lags={self.max_lags}, ic={self.ic}")

            model = VAR(train_processed)
            self.model = model.fit(maxlags=self.max_lags, ic=self.ic)

            if self.model.k_ar < 1:
                raise RuntimeError("Fitted VAR model has invalid lag order (k_ar < 1).")

            self.last_series = train_processed.values[-self.model.k_ar:]
            self.last_fit_timestamp = train_processed.index[-1]
            self.fitted = True
            logger.info(f"VAR model fitted successfully with lag order: {self.model.k_ar}")

        except (ValueError, np.linalg.LinAlgError) as e:
            logger.error(f"Failed to fit VAR model: {str(e)}", exc_info=True)
            raise RuntimeError(f"VAR fitting failed: {str(e)}")

    def predict(self, input_data: Optional[pd.DataFrame] = None, forecast_steps: Optional[int] = None) -> pd.DataFrame:
        """
        Generate predictions for the specified horizon.

        For VAR models, predictions are based on the fitted model's history, and input_data is ignored.

        Args:
            input_data: Input data (ignored for VAR). Defaults to None.
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
            predictions_proc = self.model.forecast(self.last_series, steps=steps)

            if predictions_proc.size == 0 or predictions_proc.shape[0] < steps:
                logger.warning(f"VAR forecast returned empty or insufficient data for {self.__class__.__name__}. Returning NaNs.")
                return pd.DataFrame(np.full((steps, self.num_features), np.nan), columns=self.preprocessor.columns)

            predictions_original = self.preprocessor.inverse_transforms(
                predictions_proc,
                start_after=self.last_fit_timestamp
            )

            logger.info(f"Generated {steps} predictions for {self.__class__.__name__}")
            return predictions_original

        except (ValueError, np.linalg.LinAlgError) as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"VAR prediction failed: {str(e)}")

    def get_valid_params(self) -> Set[str]:
        """
        Get the set of valid parameter names for the VAR model.

        Returns:
            Set of valid parameter names.
        """
        return {"max_lags", "ic", "window_size", "preprocessing", "n_trials"}
