import pytest
import logging
from utils.logging_utils import (
    log_training_start,
    log_training_success,
    log_trial_failure,
    log_best_hyperparams,
)
# Mock the classes that the logging functions expect as type hints
from models.base import NeuralTSForecaster, StatTSForecaster

# --- Test Setup (Mocks) ---

# Create simple mock classes that inherit from the required types.
# They don't need any logic; they exist solely for type validation in the tests.
class MockNeuralForecaster(NeuralTSForecaster):
    def __init__(self):
        super().__init__({}, num_features=1, forecast_steps=1)
    def _fit_and_evaluate_fold(self, *args, **kwargs): pass
    def fit(self, *args, **kwargs): pass
    def predict(self, *args, **kwargs): pass
    def _train_model(self, *args, **kwargs): pass

class MockStatForecaster(StatTSForecaster):
    def __init__(self):
        super().__init__({}, num_features=1, forecast_steps=1)
    def _fit_and_evaluate_fold(self, *args, **kwargs): pass
    def fit(self, *args, **kwargs): pass
    def predict(self, *args, **kwargs): pass

# --- Tests for `log_training_start` ---

def test_log_training_start_success(caplog):
    """Tests that the function correctly logs the start of a training process."""
    model_mock = MockNeuralForecaster()
    with caplog.at_level(logging.INFO):
        log_training_start("test_model", model_mock)
    assert "[test_model] Starting training with model: MockNeuralForecaster" in caplog.text

def test_log_training_start_empty_model_name_fails():
    """Tests that the function raises a ValueError for an empty model name."""
    model_mock = MockStatForecaster()
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        log_training_start("", model_mock)

# --- Tests for `log_training_success` ---

def test_log_training_success(caplog):
    """Tests that the function correctly logs a successful training completion."""
    with caplog.at_level(logging.INFO):
        log_training_success("test_model", 0.12345, 10)
    assert "[test_model] Training completed. Best val_loss: 0.123450, at epoch 10" in caplog.text

@pytest.mark.parametrize("args, error_msg", [
    (("", 0.1, 10), "model_name cannot be empty."),
    (("test", -0.1, 10), "val_loss must be a non-negative number."),
    (("test", 0.1, -1), "best_epoch must be a non-negative integer."),
    (("test", "invalid", 10), "val_loss must be a non-negative number."),
])
def test_log_training_success_invalid_inputs_fail(args, error_msg):
    """Tests that the function raises ValueErrors for various invalid inputs."""
    with pytest.raises(ValueError, match=error_msg):
        log_training_success(*args)

# --- Tests for `log_trial_failure` ---

def test_log_trial_failure(caplog):
    """Tests that the function correctly logs a failed optimization trial."""
    params = {'lr': 0.1}
    exception = ValueError("Test error")
    with caplog.at_level(logging.WARNING):
        log_trial_failure("test_model", params, exception)
    assert "[test_model] Trial failed with hyperparameters={'lr': 0.1}: Test error" in caplog.text

@pytest.mark.parametrize("args, error_msg", [
    (("", {'lr': 0.1}, ValueError()), "model_name cannot be empty."),
    (("test", "not_a_dict", ValueError()), "hyperparameters must be a dictionary."),
])
def test_log_trial_failure_invalid_inputs_fail(args, error_msg):
    """Tests that the function raises ValueErrors for various invalid inputs."""
    with pytest.raises(ValueError, match=error_msg):
        log_trial_failure(*args)

# --- Tests for `log_best_hyperparams` ---

def test_log_best_hyperparams(caplog):
    """Tests that the function correctly logs the best found hyperparameters."""
    params = {'lr': 0.01, 'layers': 2}
    with caplog.at_level(logging.INFO):
        log_best_hyperparams("test_model", "optuna", params, 0.05)
    assert "[test_model] Best hyperparameters (optuna): {'lr': 0.01, 'layers': 2}, Best loss: 0.050000" in caplog.text

@pytest.mark.parametrize("args, error_msg", [
    (("", "optuna", {}, 0.1), "model_name cannot be empty."),
    (("test", "", {}, 0.1), "method cannot be empty."),
    (("test", "optuna", "not_a_dict", 0.1), "hyperparameters must be a dictionary."),
    (("test", "optuna", {}, -0.1), "loss must be a non-negative number."),
])
def test_log_best_hyperparams_invalid_inputs_fail(args, error_msg):
    """Tests that the function raises ValueErrors for various invalid inputs."""
    with pytest.raises(ValueError, match=error_msg):
        log_best_hyperparams(*args)
