"""Unit tests for the abstract base forecaster classes.

This module provides a comprehensive suite of tests for the `TSForecaster`,
`StatTSForecaster`, and `NeuralTSForecaster` abstract base classes from
`models/base.py`. The tests verify the core, non-abstract logic of these
classes, including initialization, evaluation, data preparation, and the
hyperparameter optimization loop.

To test the abstract classes, simple concrete subclasses are defined locally.
All external dependencies, such as the dataset and hyperparameter utilities,
are mocked to ensure the tests are isolated and deterministic.
"""

import pytest
import pandas as pd
import numpy as np
from pytest import approx
import logging
from unittest.mock import MagicMock, call
from models.base import TSForecaster, StatTSForecaster, NeuralTSForecaster

# --- Mocks for Dependencies ---

@pytest.fixture
def mock_preprocessor_class(mocker):
    """Mocks the Preprocessor class and returns the patch object."""
    patcher = mocker.patch('models.base.Preprocessor')
    return patcher

@pytest.fixture
def mock_dataset(mocker):
    """Mocks the TimeSeriesDataset class."""
    mock = MagicMock()
    mock.generate_walk_forward_folds.return_value = [
        pd.DataFrame(np.random.rand(100, 2)),
        pd.DataFrame(np.random.rand(110, 2)),
    ]
    mock.series = pd.DataFrame(np.random.rand(120, 2), columns=['feature_1', 'feature_2'])
    mock.development_data = mock.series.iloc[:100]
    mock.test_data = mock.series.iloc[100:]
    mock.freq = 'D'
    mock.name = "mock_dataset"
    return mock

@pytest.fixture
def mock_hyperopt_funcs(mocker):
    """Mocks all hyperparameter generation functions."""
    mocker.patch('models.base.generate_grid_params', return_value=[{'p': 1}, {'p': 2}])
    mocker.patch('models.base.generate_optuna_params', return_value=[{'p': 3}, {'p': 4}])
    mocker.patch('models.base.generate_random_params', return_value=[{'p': 5}, {'p': 6}])

# --- Concrete Subclasses for Testing ---

class ConcreteStatForecaster(StatTSForecaster):
    """A concrete implementation of StatTSForecaster for testing."""
    def __init__(self, model_params, num_features, forecast_steps):
        super().__init__(model_params, num_features, forecast_steps)
        self._fit_and_evaluate_fold = MagicMock(return_value=np.random.rand())

    def fit(self, *args, **kwargs):
        self.fitted = True

    def predict(self, *args, **kwargs):
        return pd.DataFrame(np.random.rand(self.forecast_steps, self.num_features))

class ConcreteNeuralForecaster(NeuralTSForecaster):
    """A concrete implementation of NeuralTSForecaster for testing."""
    def __init__(self, model_params, num_features, forecast_steps):
        super().__init__(model_params, num_features, forecast_steps)
        self._fit_and_evaluate_fold = MagicMock(return_value=np.random.rand())
        self.model = MagicMock() # Mock the torch model

    def fit(self, *args, **kwargs):
        self.fitted = True
        return np.random.rand() # Fit method for neural models returns a loss

    def predict(self, *args, **kwargs):
        return pd.DataFrame(np.random.rand(self.forecast_steps, self.num_features))

    def _train_model(self, *args, **kwargs):
        return MagicMock() # Return a mock trained model instance

# --- TSForecaster Tests ---

def test_tsforecaster_initialization(mock_preprocessor_class):
    """
    Scenario: A forecaster is initialized with valid parameters.
    Assumptions: The __init__ method should correctly assign all attributes,
    including model parameters and a Preprocessor instance.
    """
    params = {'p': 1, 'preprocessing': {'scaling': {'enabled': True}}}
    forecaster = ConcreteStatForecaster(params, num_features=2, forecast_steps=10)

    assert forecaster.num_features == 2
    assert forecaster.forecast_steps == 10
    assert forecaster.model_params == params
    mock_preprocessor_class.assert_called_once_with({'scaling': {'enabled': True}})

@pytest.mark.parametrize("args, error_msg", [
    (({'p': 1}, 2, 0), "forecast_steps must be positive"),
    (({'p': 1}, 0, 10), "num_features must be positive"),
    (("not_a_dict", 2, 10), "model_params must be a dictionary"),
])
def test_tsforecaster_initialization_fails_with_invalid_params(args, error_msg):
    """
    Scenario: A forecaster is initialized with invalid parameters.
    Assumptions: The __init__ method should raise a ValueError for invalid inputs
    such as non-positive forecast steps or features.
    """
    with pytest.raises(ValueError, match=error_msg):
        ConcreteStatForecaster(*args)

def test_evaluate_method_calculates_mse():
    """
    Scenario: The evaluate method is called with valid data.
    Assumptions: The method should correctly calculate the Mean Squared Error (MSE).
    """
    forecaster = ConcreteStatForecaster({}, 1, 1)
    y_true = pd.DataFrame({'a': [1, 2, 3]})
    y_pred = pd.DataFrame({'a': [2, 3, 4]})
    mse = forecaster.evaluate(y_true, y_pred)
    assert mse == pytest.approx(1.0)

def test_evaluate_method_handles_empty_data():
    """
    Scenario: The evaluate method is called with empty DataFrames.
    Assumptions: The method should return infinity and not raise an error.
    """
    forecaster = ConcreteStatForecaster({}, 1, 1)
    y_true = pd.DataFrame()
    y_pred = pd.DataFrame({'a': [1]})
    assert forecaster.evaluate(y_true, y_pred) == float('inf')

# --- Hyperparameter Optimization Tests ---

def test_optimize_hyperparameters_grid_search(mock_dataset):
    """
    Test that optimize_hyperparameters returns the best params based on fold losses
    when using the 'grid' method.
    """

    class DummyForecaster(StatTSForecaster):
        call_counter = {}

        def __init__(self, params, num_features, forecast_steps):
            super().__init__(params, num_features, forecast_steps)
            self.params = params
            self.loss_values = {
                1: [0.2, 0.3],  # average: 0.25
                2: [0.1, 0.2]   # average: 0.15 -> should be selected
            }

        def fit(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return pd.DataFrame(np.random.rand(10, 2))  # simulate prediction

        def _fit_and_evaluate_fold(self, *args, **kwargs):
            p = self.params['p']
            if p not in DummyForecaster.call_counter:
                DummyForecaster.call_counter[p] = 0
            i = DummyForecaster.call_counter[p]
            DummyForecaster.call_counter[p] += 1
            return self.loss_values[p][i]

    model_config = {'optimization': {'method': 'grid', 'params': {'p': [1, 2]}}}
    validation_params = {'n_folds': 2, 'max_window_size': 50}

    forecaster = DummyForecaster({'p': 1}, 2, 10)  # we start with p=1, but grid will overwrite it
    best_params, best_loss = forecaster.optimize_hyperparameters(mock_dataset, model_config, validation_params)

    assert best_params['p'] == 2
    assert best_loss == approx(0.15)
    
def test_optimize_hyperparameters_raises_error_if_no_valid_params(mock_dataset, mock_hyperopt_funcs):
    """
    Scenario: The optimization loop completes, but every trial resulted in an infinite loss.
    Assumptions: A ValueError should be raised to indicate that no valid
    hyperparameter combination was found.
    """
    class DummyFailingForecaster(StatTSForecaster):
        def fit(self, *args, **kwargs): pass
        def predict(self, *args, **kwargs): return pd.DataFrame()
        def _fit_and_evaluate_fold(self, *args, **kwargs): return float('inf')

    model_config = {'optimization': {'method': 'grid', 'params': {'p': [1]}}}
    validation_params = {'n_folds': 1, 'max_window_size': 50}
    forecaster = DummyFailingForecaster(model_config, 2, 10)

    with pytest.raises(ValueError, match="No valid parameter combinations found"):
        forecaster.optimize_hyperparameters(mock_dataset, model_config, validation_params)

# --- StatTSForecaster Tests ---

def test_stat_forecaster_prepare_data_univariate(mock_dataset):
    """
    Scenario: `prepare_data` is called on a univariate statistical model.
    Assumptions: The method should split the original dataset into multiple
    univariate datasets, one for each feature.
    """
    forecaster = ConcreteStatForecaster({}, 2, 10)
    forecaster.is_univariate = True

    datasets = forecaster.prepare_data(mock_dataset)

    assert len(datasets) == 2 # One for each feature
    assert datasets[0].name == "mock_dataset_feature_1"
    assert list(datasets[0].series.columns) == ["feature_1"]

def test_stat_forecaster_prepare_data_multivariate(mock_dataset):
    """
    Scenario: `prepare_data` is called on a multivariate statistical model.
    Assumptions: The method should return the original dataset unmodified in a list.
    """
    forecaster = ConcreteStatForecaster({}, 2, 10)
    forecaster.is_univariate = False

    datasets = forecaster.prepare_data(mock_dataset)

    assert len(datasets) == 1
    assert datasets[0] == mock_dataset

# --- NeuralTSForecaster Tests ---
def test_neural_forecaster_fit_splits_data_correctly(mocker):
    """
    Scenario: The `fit` method of a neural forecaster is called for training.
    Assumption: It should correctly split the data into training and validation sets
    and call the training routine.
    """

    # Dummy subclass of NeuralTSForecaster for isolated testing
    class DummyNeuralForecaster(NeuralTSForecaster):
        def _build_model(self):
            model = MagicMock()
            model.to.return_value = model
            model.best_val_loss = 0.123
            return model

        def _train_model(self, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor):
            # Return the model after "training"
            return self.model

        def _create_data_loaders(self, X, y, validation_percentage):
            return MagicMock(), MagicMock()

    # Patch the sliding window creation to return controlled data
    mocker.patch(
        'models.base.create_sliding_window',
        return_value=(np.ones((100, 10, 1)), np.ones((100, 5, 1)))
    )

    # Instantiate the dummy forecaster
    params = {'window_size': 10}
    forecaster = DummyNeuralForecaster(params, num_features=1, forecast_steps=5)

    # Initialize the model to avoid NotImplementedError
    forecaster.model = forecaster._build_model()

    # Create input data
    train_series = pd.DataFrame(np.random.rand(1000, 1))

    # Call the fit method
    val_loss = forecaster.fit(train_series, is_final_fit=False, early_stopping_validation_percentage=20)

    # Assertions
    assert isinstance(val_loss, float)
    assert val_loss == 0.123
