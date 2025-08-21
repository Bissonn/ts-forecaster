import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock
import logging # Import logging module
from utils.dataset import TimeSeriesDataset

@pytest.fixture
def sample_config(tmp_path):
    """
    Pytest fixture to provide a sample configuration dictionary.
    It creates a dummy CSV file for testing data loading.
    """
    data_path = tmp_path / "sample_data.csv"
    data = pd.DataFrame({
        'date': pd.to_datetime(pd.date_range('2023-01-01', periods=200)),
        'value': np.arange(200, dtype=float),
        'feature': np.random.rand(200)
    })
    data.to_csv(data_path, index=False)

    return {
        'datasets': {
            'test_dataset': {
                'path': str(data_path),
                'columns': ['value', 'feature'],
                'freq': 'D'
            }
        },
        'experiments': [{'validation_setup': {'forecast_steps': 20}}]
    }

@pytest.fixture
def sample_dataframe():
    """Pytest fixture to provide a sample DataFrame for direct instantiation."""
    return pd.DataFrame({
        'date': pd.to_datetime(pd.date_range('2023-01-01', periods=100)),
        'value': np.arange(100, dtype=float)
    }).set_index('date')

# --- Initialization Tests ---

def test_dataset_initialization_from_file(sample_config):
    """Tests if the dataset is correctly initialized by loading data from a file."""
    dataset = TimeSeriesDataset('test_dataset', sample_config)
    assert not dataset.series.empty
    assert dataset.name == 'test_dataset'
    assert list(dataset.columns) == ['value', 'feature']
    assert isinstance(dataset.series.index, pd.DatetimeIndex)

def test_dataset_initialization_from_dataframe(sample_config, sample_dataframe):
    """Tests if the dataset is correctly initialized when a DataFrame is passed directly."""
    dataset = TimeSeriesDataset(
        'direct_data', sample_config, data=sample_dataframe, columns=['value']
    )
    assert not dataset.series.empty
    assert len(dataset.series) == 100
    assert list(dataset.columns) == ['value']

def test_initialization_fails_with_missing_dataset_in_config():
    """Tests that initialization raises ValueError for a non-existent dataset name."""
    with pytest.raises(ValueError, match="not found in config"):
        TimeSeriesDataset('nonexistent', {'datasets': {}})

# --- Data Splitting Tests ---

def test_data_split_correctly(sample_config):
    """Tests the split into development and test sets."""
    dataset = TimeSeriesDataset('test_dataset', sample_config)
    forecast_steps = 30
    dataset.split_data(forecast_steps=forecast_steps)

    assert dataset.development_data is not None
    assert dataset.test_data is not None
    assert len(dataset.test_data) == forecast_steps
    assert len(dataset.development_data) == len(dataset.series) - forecast_steps
    # Check for data continuity
    expected_next_date = dataset.development_data.index[-1] + pd.Timedelta(days=1)
    assert dataset.test_data.index[0] == expected_next_date

def test_split_fails_with_insufficient_data(sample_dataframe):
    """Tests that splitting raises an error if the dataset is too short."""
    dataset = TimeSeriesDataset('test', {'datasets': {}}, data=sample_dataframe.iloc[:15])
    with pytest.raises(ValueError, match="Dataset is too short"):
        dataset.split_data(forecast_steps=20)

# --- Fold Generation Tests ---

def test_generate_walk_forward_folds(sample_config):
    """Tests the generation of walk-forward validation folds."""
    dataset = TimeSeriesDataset('test_dataset', sample_config)
    dataset.split_data(forecast_steps=20)  # dev_set size = 180
    
    folds = dataset.generate_walk_forward_folds(max_window_size=100, n_folds=4)
    
    assert len(folds) == 4
    # dev_len - n_folds * forecast_steps = 180 - 4 * 20 = 100
    assert len(folds[0]) == 100
    assert len(folds[1]) == 120
    assert len(folds[2]) == 140
    assert len(folds[3]) == 160 # Last fold before consuming all data

    # Test generating all possible folds
    full_folds = dataset.generate_walk_forward_folds(max_window_size=100, n_folds=5)
    assert len(full_folds) == 5
    assert len(full_folds[-1]) == 180
    pd.testing.assert_frame_equal(full_folds[-1], dataset.development_data)


def test_generate_folds_logs_warning_when_data_is_short(sample_config, caplog):
    """Tests that fold generation handles insufficient data gracefully and logs a warning."""
    dataset = TimeSeriesDataset('test_dataset', sample_config)
    dataset.series = dataset.series.iloc[:130] # Shorten the series
    dataset.split_data(forecast_steps=10) # dev_set size = 120
    
    # Required length = 100 + 20 * 2 = 140. Data length is 120, so it's too short.
    with caplog.at_level("WARNING"):
        folds = dataset.generate_walk_forward_folds(max_window_size=100, n_folds=2)
        assert "Development data (120 rows) is too short" in caplog.text
    
    # It should still generate what it can
    assert len(folds) == 2
    assert len(folds[0]) == 100
    assert len(folds[1]) == 120

# --- Results Saving Tests ---

def test_save_results(sample_config, tmp_path, mocker):
    """Tests that predictions and metrics are saved correctly."""
    # Mock os functions to avoid actual file creation
    mock_makedirs = mocker.patch("utils.dataset.os.makedirs")
    
    # Mock the to_csv method to check its calls
    mock_to_csv = mocker.patch("pandas.DataFrame.to_csv")
    
    dataset = TimeSeriesDataset('test_dataset', sample_config)
    dataset.split_data(forecast_steps=10)
    
    predictions = np.random.rand(10, 2)
    mock_metrics_fn = MagicMock(return_value={'mae': 0.5, 'rmse': 0.7})
    
    # Redirect save path to a temporary directory
    mocker.patch('os.makedirs', return_value=None)
    pred_path = tmp_path / "results" / "predictions"
    metrics_path = tmp_path / "results" / "metrics"
    mocker.patch.object(os, 'makedirs', lambda path, exist_ok: None) # Simple mock

    # Redefine the paths inside the function for testing
    pred_file = f"{pred_path}/{dataset.name}_test_model_predictions.csv"
    metrics_file = f"{metrics_path}/{dataset.name}_test_model_metrics.csv"
    
    dataset._save_results(predictions, "test_model", 10, mock_metrics_fn)

    # Assertions
    assert mock_to_csv.call_count == 2 # Once for predictions, once for metrics
    mock_metrics_fn.assert_called()
    
# --- Getter Method Tests ---

def test_getters(sample_config):
    """Tests the getter methods for development and test data."""
    dataset = TimeSeriesDataset('test_dataset', sample_config)

    # Should raise error before splitting
    with pytest.raises(ValueError, match="Data has not been split yet"):
        dataset.get_development_data()
    with pytest.raises(ValueError, match="Data has not been split yet"):
        dataset.get_test_data()
        
    dataset.split_data(forecast_steps=20)
    
    dev_data = dataset.get_development_data()
    test_data = dataset.get_test_data()
    
    assert isinstance(dev_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)
    assert not dev_data.empty
    assert not test_data.empty

# --- TESTS FOR INDEX HANDLING ---

@pytest.fixture
def data_without_date_column():
    """Fixture to provide a simple DataFrame without any date information."""
    return pd.DataFrame({'value': np.arange(50, dtype=float)})

def test_initialization_with_missing_date_column_falls_back_to_rangeindex(sample_config, data_without_date_column, caplog):
    """
    Scenario: Initialize TimeSeriesDataset with a DataFrame that has no 'date' column
    and no DatetimeIndex.
    Assumption: The class should log a warning and correctly create a RangeIndex.
    """
    # --- Use caplog to capture logging output ---
    with caplog.at_level(logging.WARNING):
        dataset = TimeSeriesDataset(
            'sequential_data',
            sample_config,
            data=data_without_date_column
        )

    # Check if the expected warning was logged
    assert "No DatetimeIndex or 'date' column found" in caplog.text
    
    # Check if the index is correct
    assert isinstance(dataset.series.index, pd.RangeIndex), \
        f"Expected RangeIndex, but got {type(dataset.series.index)}"
    assert len(dataset.series) == 50

def test_initialization_with_custom_date_column_name(sample_config):
    """
    Scenario: Initialize TimeSeriesDataset with a DataFrame where the date column has a custom name.
    Assumption: The class should correctly identify and use this column to create the DatetimeIndex.
    """
    # Create data with a 'timestamp' column instead of 'date'
    custom_date_data = pd.DataFrame({
        'timestamp': pd.to_datetime(pd.date_range('2024-01-01', periods=50)),
        'value': np.random.rand(50)
    })
    
    dataset = TimeSeriesDataset(
        'custom_date_data',
        sample_config,
        data=custom_date_data,
        date_column='timestamp' # Explicitly provide the custom column name
    )
    
    assert isinstance(dataset.series.index, pd.DatetimeIndex), \
        f"Expected DatetimeIndex, but got {type(dataset.series.index)}"
    assert 'timestamp' not in dataset.series.columns, \
        "The custom date column should be removed after being set as index."
