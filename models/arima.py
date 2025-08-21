"""Module for non-seasonal ARIMA time series forecasting model."""

from typing import Dict, Any

from models.base_arima import ARIMABaseForecaster
from models.model_registry import register_model


@register_model("arima", is_univariate=True)
class ARIMAForecaster(ARIMABaseForecaster):
    """Implementation of the non-seasonal ARIMA forecasting model."""

    def __init__(self, model_params: Dict[str, Any], num_features: int, forecast_steps: int) -> None:
        """
        Initialize the ARIMA forecaster.

        Args:
            model_params: Model-specific parameters (e.g., p, d, q).
            num_features: Number of features in the time series data (must be 1).
            forecast_steps: Number of steps to forecast.
        """
        super().__init__(model_params=model_params, num_features=num_features, forecast_steps=forecast_steps, seasonal=False)