"""Unit tests for the training loop utility.

This module provides a comprehensive suite of tests for the `run_train_loop`
function located in `utils/train_loop.py`. The tests cover standard
functionality, edge cases, and error handling to ensure the training
process is robust and reliable. Dependencies like the PyTorch model,
optimizer, and logging functions are mocked to isolate the training loop's
logic and enable fast, deterministic testing.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, call
from utils.train_loop import run_train_loop
import logging


# --- Fixtures and Mocks Setup ---

@pytest.fixture
def mock_model():
    """
    Provides a mock `torch.nn.Module` for testing.

    This fixture creates a generic mock model with essential attributes and methods
    (`state_dict`, `train`, `eval`) that are called by the training loop.
    It is configured to return a dummy tensor when called.
    """
    model = MagicMock(spec=nn.Module)
    model.state_dict.return_value = {'weight': torch.randn(1)}
    model.train = MagicMock()
    model.eval = MagicMock()
    # The model must return a tensor to be compatible with the loss function.
    model.return_value = torch.randn(32, 1)
    return model


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks the logging utility functions to prevent actual logging during tests."""
    mocker.patch('utils.train_loop.log_training_start')
    mocker.patch('utils.train_loop.log_training_success')
    return mocker


@pytest.fixture
def sample_tensors():
    """
    Provides a complete dictionary of sample tensors for training and validation.
    Includes `None` for optional decoder inputs to test the default execution path.
    """
    return {
        "encoder_inputs_train": torch.randn(100, 10, 1),
        "decoder_inputs_train": None,
        "true_outputs_train": torch.randn(100, 1, 1),
        "encoder_inputs_val": torch.randn(20, 10, 1),
        "decoder_inputs_val": None,
        "true_outputs_val": torch.randn(20, 1, 1)
    }


@pytest.fixture
def mock_optimizer():
    """Provides a mock optimizer with `zero_grad` and `step` methods."""
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    return optimizer


@pytest.fixture
def mock_loss_fn():
    """
    Provides a mock loss function that returns a realistic, differentiable
    dummy loss tensor.
    """
    loss_fn = MagicMock(spec=nn.Module)
    # Using a real tensor is crucial for compatibility with torch.isfinite()
    loss_tensor = torch.tensor(0.1, requires_grad=True)
    loss_fn.return_value = loss_tensor
    return loss_fn


# --- Success Case Tests ---

def test_run_train_loop_completes_full_run(mock_model, sample_tensors, mock_optimizer, mock_loss_fn, mock_dependencies):
    """
    Scenario: A standard, successful training run.
    Assumptions: This test verifies that for a valid configuration, the training loop
    executes for the specified number of epochs. It checks that the model is
    switched between `train` and `eval` modes, the optimizer is used, and a
    final model with a `best_val_loss` attribute is returned.
    """
    trained_model = run_train_loop(
        model=mock_model,
        **sample_tensors,
        loss_fn=mock_loss_fn,
        optimizer=mock_optimizer,
        epochs=3,
        early_stopping_patience=5,
        device=torch.device('cpu'),
        model_name="test_model"
    )

    assert mock_model.train.call_count == 3
    assert mock_model.eval.call_count == 3
    assert mock_optimizer.step.call_count > 0
    assert hasattr(trained_model, 'best_val_loss')


def test_early_stopping_triggers_correctly(mock_model, sample_tensors, mock_optimizer, mock_dependencies):
    """
    Scenario: The early stopping mechanism terminates training.
    Assumptions: This test simulates a situation where the validation loss fails
    to improve. It verifies that the training loop is terminated early, after
    the defined `early_stopping_patience` has been exceeded.
    """
    mock_loss_fn = MagicMock(spec=nn.Module)
    # Simulate a scenario where validation loss increases with each epoch.
    # The side_effect list must be long enough to cover all calls to loss_fn,
    # including those for training batches and the single validation pass per epoch.
    mock_loss_fn.side_effect = [
        torch.tensor(0.1, requires_grad=True), torch.tensor(0.1, requires_grad=True),
        torch.tensor(0.1, requires_grad=True), torch.tensor(0.1, requires_grad=True),  # Train Epoch 1
        torch.tensor(0.1, requires_grad=True),  # Val Epoch 1
        torch.tensor(0.2, requires_grad=True), torch.tensor(0.2, requires_grad=True),
        torch.tensor(0.2, requires_grad=True), torch.tensor(0.2, requires_grad=True),  # Train Epoch 2
        torch.tensor(0.2, requires_grad=True),  # Val Epoch 2
        torch.tensor(0.3, requires_grad=True), torch.tensor(0.3, requires_grad=True),
        torch.tensor(0.3, requires_grad=True), torch.tensor(0.3, requires_grad=True),  # Train Epoch 3
        torch.tensor(0.3, requires_grad=True),  # Val Epoch 3
    ]

    run_train_loop(
        model=mock_model,
        **sample_tensors,
        loss_fn=mock_loss_fn,
        optimizer=mock_optimizer,
        epochs=10,
        early_stopping_patience=2,  # Stop after 2 epochs of no improvement
        min_epochs=1,
        device=torch.device('cpu'),
        model_name="early_stop_test"
    )

    # The loop should execute for 3 epochs:
    # Epoch 1: Sets the initial best loss.
    # Epochs 2 & 3: Patience window for no improvement before stopping.
    assert mock_model.train.call_count == 3
    assert mock_model.eval.call_count == 3


def test_no_validation_data_run(mock_model, sample_tensors, mock_optimizer, mock_loss_fn, mock_dependencies):
    """
    Scenario: The training loop is run without validation data.
    Assumptions: This test verifies that the loop executes for the full number of
    specified epochs and that the model's `eval()` method is never called.
    """
    trained_model = run_train_loop(
        model=mock_model,
        encoder_inputs_train=sample_tensors["encoder_inputs_train"],
        decoder_inputs_train=None,
        true_outputs_train=sample_tensors["true_outputs_train"],
        encoder_inputs_val=None,
        decoder_inputs_val=None,
        true_outputs_val=None,
        loss_fn=mock_loss_fn,
        optimizer=mock_optimizer,
        epochs=2,
        early_stopping_patience=3,
        device=torch.device('cpu')
    )

    assert mock_model.train.call_count == 2
    mock_model.eval.assert_not_called()
    assert hasattr(trained_model, 'best_val_loss')


# --- Input Validation and Error Handling Tests ---

@pytest.mark.parametrize("invalid_args, error_msg", [
    ({"encoder_inputs_train": torch.Tensor()}, "encoder_inputs_train must be a non-empty torch.Tensor"),
    ({"true_outputs_train": torch.Tensor()}, "true_outputs_train must be a non-empty torch.Tensor"),
    ({"batch_size": 0}, "batch_size must be a positive integer"),
    ({"epochs": 0}, "epochs must be a positive integer"),
    ({"early_stopping_patience": -1}, "early_stopping_patience must be a non-negative integer"),
    ({"model_name": ""}, "model_name cannot be empty"),
    ({"encoder_inputs_train": torch.randn(5, 1), "true_outputs_train": torch.randn(4, 1)},
     "must have the same number of samples"),
])
def test_run_train_loop_raises_value_error_for_invalid_inputs(
        mock_model, sample_tensors, mock_optimizer, mock_loss_fn, invalid_args, error_msg
):
    """
    Scenario: Parameterized test for input validation.
    Assumptions: This test verifies that the function's input validation is robust
    by checking that it correctly raises a `ValueError` for various invalid arguments.
    """
    args = {
        "model": mock_model,
        "encoder_inputs_train": sample_tensors["encoder_inputs_train"],
        "decoder_inputs_train": None,
        "true_outputs_train": sample_tensors["true_outputs_train"],
        "encoder_inputs_val": None, "decoder_inputs_val": None, "true_outputs_val": None,
        "loss_fn": mock_loss_fn, "optimizer": mock_optimizer, "epochs": 5,
        "early_stopping_patience": 3, "device": torch.device('cpu'), "model_name": "valid_model"
    }
    args.update(invalid_args)

    with pytest.raises(ValueError, match=error_msg):
        run_train_loop(**args)


def test_handles_non_finite_loss(mock_model, sample_tensors, mock_optimizer, caplog):
    """
    Scenario: The loop encounters a non-finite (infinite) validation loss.
    Assumptions: This test ensures the loop handles this case gracefully by
    logging a warning and continuing to the next epoch, rather than crashing.
    The test verifies the correct warning is logged and that the loop completes
    all specified epochs.
    """
    mock_loss = MagicMock(spec=nn.Module)
    # Simulate a scenario with finite training loss but an infinite validation loss.
    # The training loop has 4 batches (100/32 rounded up).
    # 4 finite values for training batches + 1 infinite value for the validation step.
    mock_loss.side_effect = [
                                torch.tensor(0.1, requires_grad=True),
                                torch.tensor(0.1, requires_grad=True),
                                torch.tensor(0.1, requires_grad=True),
                                torch.tensor(0.1, requires_grad=True),
                                torch.tensor(float('inf'), requires_grad=True)  # <-- Infinite loss on validation
                            ] * 5  # Provide enough values for all epochs

    with caplog.at_level(logging.WARNING):
        run_train_loop(
            model=mock_model,
            **sample_tensors,
            loss_fn=mock_loss,
            optimizer=mock_optimizer,
            epochs=5,
            early_stopping_patience=3,
            device=torch.device('cpu')
        )

    assert "Non-finite validation loss" in caplog.text
    # The loop should continue even with non-finite validation loss,
    # so it should run for all 5 epochs.
    assert mock_model.train.call_count == 5