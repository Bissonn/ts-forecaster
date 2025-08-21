"""Module for creating forecasting model instances using a factory pattern.

This module defines the ModelFactory class, which provides a high-level interface for
instantiating registered forecasting models (e.g., ARIMA, VAR, LSTM, Transformer) by
delegating to the model registry.
"""

import logging
from typing import Any, Dict, Optional

from models.model_registry import create_model, list_registered_models
from utils.dependencies import check_dependencies

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating instances of registered forecasting models."""

    @staticmethod
    def create(model_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Create an instance of a registered forecasting model.

        Args:
            model_name: Name of the model to instantiate (e.g., 'arima', 'lstm_direct').
            *args: Positional arguments to pass to the model class constructor (e.g., num_features, forecast_steps).
            **kwargs: Additional keyword arguments to pass to the model class constructor.

        Returns:
            Instance of the registered model class.

        Raises:
            ValueError: If model_name is empty or not registered.
            RuntimeError: If model instantiation fails due to invalid arguments or other errors.
        """
        if not model_name:
            raise ValueError("model_name cannot be empty.")
        if model_name not in list_registered_models():
            raise ValueError(
                f"Model '{model_name}' is not registered. Available models: {list_registered_models()}"
            )
        check_dependencies([model_name])
        try:
            logger.info(f"Creating model '{model_name}'")
            model = create_model(model_name, *args, **kwargs)
            logger.info(f"Successfully created model '{model_name}'")
            return model
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to create model '{model_name}': {str(e)}", exc_info=True)
            raise RuntimeError(f"Model creation failed: {str(e)}")

    @staticmethod
    def list_models() -> list[str]:
        """
        Get a list of all registered model names.

        Returns:
            List of registered model names.
        """
        return list_registered_models()
