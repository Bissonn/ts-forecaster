"""Module for Transformer-based time series forecasting models.

This module defines the TransformerForecaster class, which implements both direct and
iterative forecasting strategies using a Transformer neural network.
It supports both encoder-only and encoder-decoder architectures with various
configuration options for educational and experimental purposes.
"""

import math
import logging
from typing import Dict, List, Any, Set, Optional, Literal

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from models.base import NeuralTSForecaster
from models.model_registry import register_model
from utils.train_loop import run_train_loop

logger = logging.getLogger(__name__)


class TransformerModel(nn.Module):
    """Transformer neural network for time series, supporting encoder-only and encoder-decoder."""

    def __init__(
        self,
        # required shape-defining args
        input_size: int,      # number of input feature (including exogenic data)
        num_features: int,    # number of output features (targets)
        forecast_steps: int,  # horizon (steps to predict)
        window_size: int,     # input sequence length
        # architecture knobs
        hidden_size: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_ff_multiplier: float = 4.0,
        dropout: float = 0.1,
        architecture: Literal["encoder-only", "encoder-decoder"] = "encoder-only",
        # Mode of operation:
        # - "encoder-only": transformer with encoder only, no decoder
        # - "encoder-decoder": transformer with encoder and decoder
        positional_encoding: Literal["sinusoidal", "learned", "none"] = "sinusoidal",
        # Method for injecting positional information:
        # - "sinusoidal": deterministic, non-trainable (good default).
        # - "learned": trainable positional embeddings.
        # - "none": no positional encoding (rarely useful unless encoded elsewhere).

        readout: Literal["last", "mean", "max", "cls"] = "last",
        # Strategy to reduce the encoder sequence to a fixed-size vector:
        # - "last": last time step output
        # - "mean": average over time
        # - "max":  element-wise max over time
        # - "cls":  learned [CLS]-like token prepended to the sequence
    ):
        """
        Initialize the Transformer model.

        Args:
            input_size: Number of input features (can include exogenous variables).
            num_features: Number of target features to be forecasted.
            forecast_steps: Number of time steps to forecast (the horizon).
            window_size: Length of the input sequence (look-back window).
            hidden_size: Internal dimension of the model (embedding and attention width).
            num_heads: Number of heads in the multi-head attention mechanism. Must divide hidden_size.
            num_encoder_layers: Number of layers in the Transformer encoder.
            num_decoder_layers: Number of layers in the Transformer decoder (used if architecture is 'encoder-decoder').
            dim_ff_multiplier: Multiplier for the feed-forward network's inner dimension.
            dropout: Dropout rate applied to various layers for regularization.
            architecture: The model architecture to use.
                - "encoder-only": A stack of Transformer encoders followed by a readout layer.
                - "encoder-decoder": The classic architecture with both an encoder and a decoder.
            positional_encoding: Method for injecting sequence order information.
                - "sinusoidal": Deterministic, non-trainable encoding using sine/cosine functions.
                - "learned": Trainable positional embeddings.
                - "none": No positional encoding applied.
            readout: Strategy to aggregate the encoder's output sequence into a single vector.
                     Used only in 'encoder-only' architecture.
                - "last": Use the output of the last time step.
                - "mean": Average the outputs of all time steps.
                - "max": Element-wise max pooling over all time steps.
                - "cls": Prepend a learnable [CLS] token and use its output.
        """
        super().__init__()

        # --- Validation ---
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if architecture not in ['encoder-only', 'encoder-decoder']:
            raise ValueError("architecture must be either 'encoder-only' or 'encoder-decoder'.")
        if architecture == 'encoder-decoder' and readout != 'last':
             logger.warning(f"Readout mode '{readout}' is not applicable to 'encoder-decoder' architecture. It will be ignored.")

        self.architecture = architecture
        self.window_size = window_size
        self.forecast_steps = forecast_steps
        self.num_features = num_features
        self.readout = readout

        dim_feedforward = int(round(dim_ff_multiplier * hidden_size))

        # --- Shared Layers ---
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden_size)

        # --- Positional Encoding ---
        if positional_encoding == "learned":
            self.pos_encoder = nn.Parameter(torch.empty(1, window_size, hidden_size))
            nn.init.normal_(self.pos_encoder, mean=0.0, std=0.02)
        elif positional_encoding == "sinusoidal":
            pe = self._generate_positional_encoding(window_size, hidden_size)
            self.register_buffer("pos_encoder_buffer", pe)
            self.pos_encoder = self.pos_encoder_buffer
        else:  # 'none'
            self.pos_encoder = None

        # --- Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # --- Decoder (Conditional) ---
        if self.architecture == 'encoder-decoder':
            decoder_pe = self._generate_positional_encoding(forecast_steps, hidden_size)
            self.register_buffer("pos_decoder_buffer", decoder_pe)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
            self.fc = nn.Linear(hidden_size, num_features)
        else:  # encoder-only
            self.transformer_decoder = None
            self.fc = nn.Linear(hidden_size, forecast_steps * num_features)
            if self.readout == "cls":
                self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
                nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
            else:
                self.register_parameter("cls_token", None)

        self.best_val_loss = float("inf")
        self._init_weights()

    def _generate_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Generate sinusoidal positional encoding matrix.

        Args:
            max_len: Maximum sequence length.
            d_model: Model dimension (hidden_size).

        Returns:
            Positional encoding tensor of shape (1, max_len, d_model).
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self) -> None:
        """Initialize weights for the linear layers."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a causal mask to prevent attention to future steps in the decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Transformer model.

        Args:
            src: Source sequence tensor (batch, seq_len, input_size).
            tgt: Target sequence tensor for the decoder (batch, target_len, input_size).
                 Required only for 'encoder-decoder' architecture.

        Returns:
            Output forecast tensor.
        """
        src = self.input_projection(src)
        if self.pos_encoder is not None:
            src = src + self.pos_encoder[:, :src.size(1), :].to(src.device)
        src = self.input_norm(src)

        if self.architecture == 'encoder-decoder':
            if tgt is None:
                raise ValueError("tgt (decoder input) must be provided for encoder-decoder architecture.")

            tgt = self.input_projection(tgt)
            if hasattr(self, 'pos_decoder_buffer'):
                tgt = tgt + self.pos_decoder_buffer[:, :tgt.size(1), :].to(tgt.device)

            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(src.device)

            memory = self.transformer_encoder(src)
            output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            output = self.fc(output)
            output = self.dropout_layer(output)
            return output

        else:  # encoder-only
            if self.readout == "cls":
                B = src.shape[0]
                cls_tokens = self.cls_token.expand(B, -1, -1)
                src = torch.cat([cls_tokens, src], dim=1)

            output = self.transformer_encoder(src)

            if self.readout == "mean":
                feat = output.mean(dim=1)
            elif self.readout == "max":
                feat = output.max(dim=1).values
            elif self.readout == "cls":
                feat = output[:, 0, :]
            else:  # "last"
                feat = output[:, -1, :]

            feat = self.dropout_layer(feat)
            y = self.fc(feat)
            return y.view(-1, self.forecast_steps, self.num_features)


class TransformerBaseForecaster(NeuralTSForecaster):
    """Abstract base class for Transformer-based forecasting models."""

    def get_valid_params(self) -> Set[str]:
        """Get the set of valid parameter names for the Transformer model."""
        return {
            "window_size", "hidden_size", "num_heads", "num_encoder_layers",
            "num_decoder_layers", "architecture", "dim_ff_multiplier", "dropout",
            "batch_size", "learning_rate", "epochs", "early_stopping_patience",
            "weight_decay", "positional_encoding", "readout", "strategy", "n_trials"
        }

    def filter_candidates(self, candidates: List[Dict[str, Any]], model_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter hyperparameter candidates to ensure compatibility (e.g., hidden_size is divisible by num_heads)."""
        return [
            c for c in candidates
            if c.get("hidden_size", model_params["hidden_size"]) % c.get("num_heads", model_params["num_heads"]) == 0
        ]

    def _train_model(self, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor) -> nn.Module:
        """
        Orchestrates the training loop for the Transformer model.

        Args:
            X_train: Training input tensor.
            y_train: Training target tensor.
            X_val: Validation input tensor.
            y_val: Validation target tensor.

        Returns:
            The trained model instance.
        """
        tgt_train, tgt_val = None, None
        if self.model_params.get("architecture") == 'encoder-decoder':
            tgt_train = y_train.clone()
            if y_val is not None and y_val.numel() > 0:
                tgt_val = y_val.clone()
            else:
                tgt_val = torch.empty(0, device=self.device)

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
                decoder_inputs_train=tgt_train,
                true_outputs_train=y_train,
                encoder_inputs_val=X_val,
                decoder_inputs_val=tgt_val,
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


@register_model("transformer", is_univariate=False)
class TransformerForecaster(TransformerBaseForecaster):
    """Implementation of the Transformer model supporting multiple architectures and strategies."""

    def __init__(self, model_params: Dict[str, Any], num_features: int, forecast_steps: int) -> None:
        """
        Initialize the Transformer forecaster.

        Args:
            model_params: Model-specific parameters from the configuration file.
            num_features: Number of features in the time series data.
            forecast_steps: Number of steps to forecast.
        """
        super().__init__(model_params, num_features, forecast_steps)
        self._validate_model_params()

        # For iterative strategy, the model must be trained to predict only one step ahead.
        model_forecast_steps = 1 if self.model_params.get("strategy") == "iterative" else forecast_steps

        self.model = TransformerModel(
            input_size=self.model_params.get("input_size", num_features),
            num_features=num_features,
            forecast_steps=model_forecast_steps,
            window_size=self.model_params["window_size"],
            hidden_size=self.model_params.get("hidden_size", 128),
            num_encoder_layers=self.model_params.get("num_encoder_layers", 4),
            num_decoder_layers=self.model_params.get("num_decoder_layers", 4),
            num_heads=self.model_params.get("num_heads", 4),
            dim_ff_multiplier=self.model_params.get("dim_ff_multiplier", 4.0),
            dropout=self.model_params.get("dropout", 0.1),
            architecture=self.model_params.get("architecture", "encoder-only"),
            positional_encoding=self.model_params.get("positional_encoding", "sinusoidal"),
            readout=self.model_params.get("readout", "last")
        ).to(self.device)
        logger.info(f"Initialized {self.__class__.__name__} with params: {model_params}")

    def _get_y_window_steps(self) -> int:
        """Determines the number of steps for the target window during data preparation."""
        if self.model_params.get("strategy") == "iterative":
            return 1
        return self.forecast_steps

    def _internal_predict_iterative(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Autoregressive prediction engine for iterative forecasting."""
        if self.model is None or not self.fitted:
            raise ValueError("Model must be fitted before predicting.")

        self.model.eval()
        predictions = []
        current_input = input_tensor.clone().to(self.device)

        with torch.no_grad():
            for _ in range(self.forecast_steps):
                # Predict one step ahead
                output = self.model(current_input)
                predictions.append(output)
                
                # Append the prediction to the input sequence and drop the oldest value
                current_input = torch.cat((current_input[:, 1:, :], output), dim=1)

        return torch.cat(predictions, dim=1).cpu().numpy()

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions, dispatching to the correct strategy (direct or iterative).
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        start_after_ts = input_data.index[-1]
        input_proc = self.preprocessor.transform(input_data)
        input_tensor = torch.FloatTensor(input_proc.values).unsqueeze(0).to(self.device)
        
        if self.model_params.get("strategy") == "iterative":
            predictions_proc_np = self._internal_predict_iterative(input_tensor)
        else:
            # Use the default direct prediction method from the base class
            predictions_proc_np = super()._internal_predict(input_tensor)

        if predictions_proc_np.ndim == 3 and predictions_proc_np.shape[0] == 1:
            predictions_proc_np = predictions_proc_np.squeeze(0)

        return self.preprocessor.inverse_transforms(
            predictions_proc_np, start_after=start_after_ts
        )

    def _validate_model_params(self) -> None:
        """Validate essential model parameters from the configuration."""
        required_params = {"window_size", "hidden_size", "num_heads", "num_encoder_layers"}
        missing = [p for p in required_params if p not in self.model_params]
        if missing:
            raise ValueError(f"Missing required parameter(s): {missing}")

        if "input_size" in self.model_params:
            if int(self.model_params["input_size"]) < int(self.num_features):
                raise ValueError("input_size must be >= num_features when provided (for exogenous inputs).")
