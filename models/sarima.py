"""Module for seasonal SARIMA time series forecasting model."""

from typing import Dict, Any

from models.base_arima import ARIMABaseForecaster
from models.model_registry import register_model


@register_model("sarima", is_univariate=True)
class SARIMAForecaster(ARIMABaseForecaster):
    """Implementation of the seasonal SARIMA forecasting model."""

    def __init__(self, model_params: Dict[str, Any], num_features: int, forecast_steps: int) -> None:
        """
        Initialize the SARIMA forecaster.

        Args:
            model_params: Model-specific parameters (e.g., p, d, q, P, D, Q, seasonal_period).
            num_features: Number of features in the time series data (must be 1).
            forecast_steps: Number of steps to forecast.
        """
        super().__init__(model_params=model_params, num_features=num_features, forecast_steps=forecast_steps, seasonal=True)