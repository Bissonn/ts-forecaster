"""Module for training PyTorch-based forecasting models.

This module provides a universal training loop with early stopping, supporting both standard and
encoder-decoder architectures for models like LSTM, Transformer, and Generic Transformer.
"""

import logging
from typing import Optional, Union
import numpy as np

import torch
from torch import nn

from models.base import NeuralTSForecaster
from utils.logging_utils import log_training_start, log_training_success

logger = logging.getLogger(__name__)


def run_train_loop(
    model: Union[nn.Module, NeuralTSForecaster],
    encoder_inputs_train: torch.Tensor,
    decoder_inputs_train: Optional[torch.Tensor],
    true_outputs_train: torch.Tensor,
    encoder_inputs_val: Optional[torch.Tensor],
    decoder_inputs_val: Optional[torch.Tensor],
    true_outputs_val: Optional[torch.Tensor],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    early_stopping_patience: int,
    device: torch.device,
    batch_size: int = 32,
    model_name: str = "unknown",
    min_epochs: int = 5,
) -> Union[nn.Module, NeuralTSForecaster]:
    """
    Run a training loop with early stopping, supporting standard and encoder-decoder models.

    Args:
        model: PyTorch model to train (e.g., LSTM, Transformer).
        encoder_inputs_train: Training input tensor with shape (n_samples, window_size, features).
        decoder_inputs_train: Optional training decoder input tensor for encoder-decoder models.
        true_outputs_train: Training target tensor with shape (n_samples, forecast_steps, features).
        encoder_inputs_val: Optional validation input tensor.
        decoder_inputs_val: Optional validation decoder input tensor.
        true_outputs_val: Optional validation target tensor.
        loss_fn: Loss function (e.g., nn.MSELoss).
        optimizer: Optimizer (e.g., torch.optim.Adam).
        epochs: Maximum number of training epochs.
        early_stopping_patience: Number of epochs to wait for improvement before stopping.
        device: Device to train on (e.g., 'cuda', 'cpu').
        batch_size: Batch size for training. Defaults to 32.
        model_name: Name of the model for logging (e.g., 'lstm_direct'). Defaults to 'unknown'.
        min_epochs: Minimum number of epochs before early stopping. Defaults to 5.

    Returns:
        Trained model with the best state loaded based on validation or training loss.

    Raises:
        ValueError: If inputs are invalid (e.g., empty tensors, mismatched shapes, invalid parameters).
        RuntimeError: If training fails due to NaN/infinite losses or other errors.
    """
    if not isinstance(model, (nn.Module, NeuralTSForecaster)):
        raise ValueError("model must be a torch.nn.Module or NeuralTSForecaster.")
    if not isinstance(encoder_inputs_train, torch.Tensor) or encoder_inputs_train.numel() == 0:
        raise ValueError("encoder_inputs_train must be a non-empty torch.Tensor.")
    if not isinstance(true_outputs_train, torch.Tensor) or true_outputs_train.numel() == 0:
        raise ValueError("true_outputs_train must be a non-empty torch.Tensor.")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not isinstance(early_stopping_patience, int) or early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be a non-negative integer.")
    if not isinstance(min_epochs, int) or min_epochs < 1:
        raise ValueError("min_epochs must be a positive integer.")
    if not model_name:
        raise ValueError("model_name cannot be empty.")

    # Validate tensor shapes
    if encoder_inputs_train.shape[0] != true_outputs_train.shape[0]:
        raise ValueError(
            f"encoder_inputs_train ({encoder_inputs_train.shape}) and true_outputs_train "
            f"({true_outputs_train.shape}) must have the same number of samples."
        )
    if decoder_inputs_train is not None and decoder_inputs_train.shape[0] != encoder_inputs_train.shape[0]:
        raise ValueError(
            f"decoder_inputs_train ({decoder_inputs_train.shape}) must have the same number of "
            f"samples as encoder_inputs_train ({encoder_inputs_train.shape})."
        )

    # Validate validation data
    has_validation_data = (
        encoder_inputs_val is not None
        and true_outputs_val is not None
        and encoder_inputs_val.numel() > 0
        and true_outputs_val.numel() > 0
        and (decoder_inputs_val is None or decoder_inputs_val.numel() > 0)
    )
    if has_validation_data:
        if encoder_inputs_val.shape[0] != true_outputs_val.shape[0]:
            raise ValueError(
                f"encoder_inputs_val ({encoder_inputs_val.shape}) and true_outputs_val "
                f"({true_outputs_val.shape}) must have the same number of samples."
            )
        if decoder_inputs_val is not None and decoder_inputs_val.shape[0] != encoder_inputs_val.shape[0]:
            raise ValueError(
                f"decoder_inputs_val ({decoder_inputs_val.shape}) must have the same number of "
                f"samples as encoder_inputs_val ({encoder_inputs_val.shape})."
            )

    # Move model and tensors to device
    model.to(device)
    encoder_inputs_train = encoder_inputs_train.to(device)
    true_outputs_train = true_outputs_train.to(device)
    if decoder_inputs_train is not None:
        decoder_inputs_train = decoder_inputs_train.to(device)
    if has_validation_data:
        encoder_inputs_val = encoder_inputs_val.to(device)
        true_outputs_val = true_outputs_val.to(device)
        if decoder_inputs_val is not None:
            decoder_inputs_val = decoder_inputs_val.to(device)

    log_training_start(model_name, model)
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_no_improve = 0

    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for i in range(0, len(encoder_inputs_train), batch_size):
                batch_encoder_inputs = encoder_inputs_train[i : i + batch_size]
                batch_true_outputs = true_outputs_train[i : i + batch_size]
                batch_decoder_inputs = (
                    decoder_inputs_train[i : i + batch_size] if decoder_inputs_train is not None else None
                )

                optimizer.zero_grad()
                if batch_decoder_inputs is not None:
                    outputs = model(batch_encoder_inputs, batch_decoder_inputs)
                else:
                    outputs = model(batch_encoder_inputs)

                loss = loss_fn(outputs, batch_true_outputs)
                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss detected at epoch {epoch + 1}. Skipping batch.")
                    continue

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_encoder_inputs.size(0)
            train_loss /= len(encoder_inputs_train)

            log_message = f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}"
            current_loss = train_loss

            if has_validation_data:
                model.eval()
                with torch.no_grad():
                    if decoder_inputs_val is not None:
                        val_outputs = model(encoder_inputs_val, decoder_inputs_val)
                    else:
                        val_outputs = model(encoder_inputs_val)
                    val_loss = loss_fn(val_outputs, true_outputs_val).item()
                    if not np.isfinite(val_loss):
                        logger.warning(f"Non-finite validation loss at epoch {epoch + 1}. Skipping validation.")
                        continue
                    log_message += f", Val Loss: {val_loss:.6f}"
                    current_loss = val_loss

            logger.info(log_message)

            if current_loss < best_val_loss:
                best_val_loss = current_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience and epoch + 1 >= min_epochs:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)
        else:
            logger.warning("No best model state found. Returning model with final state.")

        log_training_success(model_name, best_val_loss, best_epoch)
        setattr(model, 'best_val_loss', best_val_loss)
        return model

    except Exception as e:
        logger.error(f"Training loop failed for model '{model_name}': {str(e)}", exc_info=True)
        raise RuntimeError(f"Training loop failed: {str(e)}")
