"""Module for LSTM-based time series forecasting models.

This module defines LSTMDirectForecaster and LSTMIterativeForecaster classes, which implement
direct and iterative forecasting strategies using LSTM neural networks, extending NeuralTSForecaster.
"""

import logging
from typing import Dict, Any, Set, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.base import NeuralTSForecaster
from models.model_registry import register_model
from utils.train_loop import run_train_loop

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM neural network architecture for time series forecasting."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, output_steps: int, output_features: int, dropout: float
    ) -> None:
        """
        Initialize the LSTM model.

        Args:
            input_size: Number of input features (columns in the time series).
            hidden_size: Size of the LSTM hidden layer.
            num_layers: Number of LSTM layers.
            output_steps: Number of predicted steps (horizon for direct models, 1 for iterative).
            output_features: Number of output features (columns in the time series).
            dropout: Dropout rate for LSTM layers (0.0 for single-layer models).

        Raises:
            ValueError: If input parameters are invalid (e.g., negative sizes, invalid dropout).
        """
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1 or output_steps < 1 or output_features < 1:
            raise ValueError("input_size, hidden_size, num_layers, output_steps, and output_features must be positive.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0.")
        if num_layers == 1 and dropout != 0.0:
            raise ValueError("dropout must be 0.0 for single-layer LSTM.")

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_steps * output_features)
        self.output_steps = output_steps
        self.output_features = output_features
        self.best_val_loss = float("inf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input data through the LSTM model.

        Args:
            x: Input tensor of shape (batch_size, window_size, input_size).

        Returns:
            Output tensor of shape (batch_size, output_steps, output_features).

        Raises:
            ValueError: If input tensor shape is invalid.
        """
        if x.dim() != 3 or x.size(2) != self.lstm.input_size:
            raise ValueError(f"Expected input shape (batch_size, window_size, {self.lstm.input_size}), got {x.shape}")

        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = self.fc(output[:, -1, :])
        return output.reshape(-1, self.output_steps, self.output_features)


class LSTMBaseForecaster(NeuralTSForecaster):
    """Abstract base class for LSTM-based forecasting models."""

    def _validate_model_params(self) -> None:
        """
        Validate LSTM model parameters.

        Raises:
            ValueError: If model parameters are invalid.
        """
        required_params = {"window_size", "hidden_size", "num_layers"}
        for param in required_params:
            if param not in self.model_params:
                raise ValueError(f"Missing required parameter: {param}")
        if not isinstance(self.model_params["hidden_size"], int) or self.model_params["hidden_size"] < 1:
            raise ValueError("hidden_size must be a positive integer.")
        if not isinstance(self.model_params["num_layers"], int) or self.model_params["num_layers"] < 1:
            raise ValueError("num_layers must be a positive integer.")
        if not 0.0 <= self.model_params.get("dropout", 0.0) <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0.")
        if self.model_params["num_layers"] == 1 and self.model_params.get("dropout", 0.0) != 0.0:
            raise ValueError("dropout must be 0.0 for single-layer LSTM.")
    
    def get_valid_params(self) -> Set[str]:
        """
        Get the set of valid parameter names for the LSTM model.

        Returns:
            Set of valid parameter names.
        """
        return {
            "window_size",
            "hidden_size",
            "num_layers",
            "dropout",
            "batch_size",
            "learning_rate",
            "epochs",
            "early_stopping_patience",
            "weight_decay",
            "n_trials",
        }

    def filter_candidates(self, candidates: List[Dict[str, Any]], model_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter candidate parameter combinations based on model constraints.

        Args:
            candidates: List of parameter combinations.
            model_params: Current model parameters.

        Returns:
            Filtered parameter combinations.

        Note:
            Filters candidates where hidden_size > 2 * window_size to prevent overfitting.
        """
        return [
            c for c in candidates
            if c.get("hidden_size", model_params["hidden_size"]) <= 2 * c.get("window_size", model_params["window_size"])
        ]

    def _train_model(self, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor) -> nn.Module:
        """
        Train the model using the generalized training loop.

        Args:
            X_train: Training input tensor.
            y_train: Training target tensor.
            X_val: Validation input tensor.
            y_val: Validation target tensor.

        Returns:
            Trained model instance.

        Raises:
            RuntimeError: If training fails due to invalid inputs or model errors.
        """
        try:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.model_params.get("learning_rate", 0.001),
                weight_decay=self.model_params.get("weight_decay", 1e-5),
            )
            criterion = nn.MSELoss()

            trained_model_instance = run_train_loop(
                model=self.model,
                encoder_inputs_train=X_train,
                decoder_inputs_train=None,
                true_outputs_train=y_train,
                encoder_inputs_val=X_val,
                decoder_inputs_val=None,
                true_outputs_val=y_val,
                loss_fn=criterion,
                optimizer=optimizer,
                epochs=self.model_params.get("epochs", 100),
                early_stopping_patience=self.model_params.get("early_stopping_patience", 10),
                device=self.device,
                batch_size=self.model_params.get("batch_size", 32),
                model_name=self.__class__.__name__,
            )
            return trained_model_instance

        except (ValueError, RuntimeError) as e:
            logger.error(f"Training failed for {self.__class__.__name__}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Training failed: {str(e)}")


@register_model("lstm_direct", is_univariate=False)
class LSTMDirectForecaster(LSTMBaseForecaster):
    """Implementation of the LSTM model with direct forecasting strategy."""

    def __init__(self, model_params: Dict[str, Any], num_features: int, forecast_steps: int) -> None:
        """
        Initialize the LSTM direct forecaster.

        Args:
            model_params: Model-specific parameters (e.g., hidden_size, num_layers).
            num_features: Number of features in the time series data.
            forecast_steps: Number of steps to forecast.

        Raises:
            ValueError: If model parameters or inputs are invalid.
        """
        # Ensure that for any single-layer model, the dropout is explicitly set to 0.0.
        # This corrects invalid combinations from hyperparameter optimization (e.g., num_layers=1 with dropout > 0)
        # and handles cases where dropout is not specified, preventing it from using a faulty default.
        if model_params.get("num_layers") == 1:
            model_params["dropout"] = 0.0
        super().__init__(model_params, num_features, forecast_steps)
        self._validate_model_params()
        self.model = LSTMModel(
            input_size=num_features,
            hidden_size=self.model_params.get("hidden_size", 50),
            num_layers=self.model_params.get("num_layers", 1),
            output_steps=self.forecast_steps,
            output_features=self.num_features,
            dropout=self.model_params.get("dropout", 0.0),
        ).to(self.device)
        logger.info(f"Initialized {self.__class__.__name__} with params: {model_params}")

@register_model("lstm_iterative", is_univariate=False)
class LSTMIterativeForecaster(LSTMBaseForecaster):
    """Implementation of the LSTM model with iterative forecasting strategy."""

    def __init__(self, model_params: Dict[str, Any], num_features: int, forecast_steps: int) -> None:
        """
        Initialize the LSTM iterative forecaster.

        Args:
            model_params: Model-specific parameters (e.g., hidden_size, num_layers).
            num_features: Number of features in the time series data.
            forecast_steps: Number of steps to forecast.

        Raises:
            ValueError: If model parameters or inputs are invalid.
        """
        # Ensure that for any single-layer model, the dropout is explicitly set to 0.0.
        # This corrects invalid combinations from hyperparameter optimization (e.g., num_layers=1 with dropout > 0)
        # and handles cases where dropout is not specified, preventing it from using a faulty default.
        if model_params.get("num_layers") == 1:
            model_params["dropout"] = 0.0
        super().__init__(model_params, num_features, forecast_steps)
        self._validate_model_params()
        self.model = LSTMModel(
            input_size=num_features,
            hidden_size=self.model_params.get("hidden_size", 32),
            num_layers=self.model_params.get("num_layers", 2),
            output_steps=1,  # Iterative model predicts one step at a time
            output_features=num_features,
            dropout=self.model_params.get("dropout", 0.2),
        ).to(self.device)
        logger.info(f"Initialized {self.__class__.__name__} with params: {model_params}")

    def _internal_predict(self, input_tensor: torch.Tensor) -> np.ndarray:
        """

        Autoregressive prediction engine for iterative forecasting, supporting batch processing.

        Generates predictions step-by-step for the specified forecast horizon for a batch of time series.

        Args:
            input_tensor: Input tensor of shape (batch_size, window_size, num_features).

        Returns:
            Predictions of shape (batch_size, forecast_steps, num_features)
        
        Raises:
            ValueError: If input tensor shape is invalid or model is not fitted.
            RuntimeError: If prediction fails due to model or tensor errors.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting.")
        if input_tensor.dim() != 3 or input_tensor.size(2) != self.num_features:
            raise ValueError(f"Expected input shape (batch_size, window_size, {self.num_features}), got {input_tensor.shape}")

        batch_size = input_tensor.size(0)
        current_input = input_tensor.clone().to(self.device)
        
        # Pre-allocate a tensor to store predictions for the entire batch
        all_predictions = torch.zeros(batch_size, self.forecast_steps, self.num_features, device=self.device)
        self.model.eval()

        try:
            with torch.no_grad():
                for i in range(self.forecast_steps):
                    # Predict one step ahead for the entire batch
                    output = self.model(current_input)
                    # Store the prediction for the current step across the whole batch
                    all_predictions[:, i, :] = output.squeeze(1)
                    # Update the input window for the next step by rolling it forward
                    current_input = torch.cat((current_input[:, 1:, :], output), dim=1)

            return all_predictions.cpu().numpy()

        except (ValueError, RuntimeError) as e:
            logger.error(f"Prediction failed for {self.__class__.__name__}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Iterative prediction failed: {str(e)}")

    def _get_y_window_steps(self) -> int:
        """
        Get the number of target steps for creating sliding windows.

        Returns:
            Always 1 for iterative forecasting, as predictions are made one step at a time.
        """
        return 1
