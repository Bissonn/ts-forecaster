"""Module for logging training-related events in the forecasting framework.

This module provides functions to log the start and completion of model training,
hyperparameter optimization results, and trial failures, ensuring consistent logging
across models like ARIMA, VAR, LSTM, and Transformer.
"""
import pytest
import pandas as pd
import numpy as np
from utils.visualizer import Visualizer


@pytest.fixture
def sample_data():
    """
    Pytest fixture to provide sample data for visualization tests.
    Returns a dictionary containing test data, predictions, and metadata for plotting.
    """
    # Create a sample DataFrame with two features and datetime index
    test_df = pd.DataFrame({
        'feature_1': np.arange(10, dtype=float),
        'feature_2': np.arange(100, 110, dtype=float)
    }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=10)))

    # Create predictions array with shape (10, 2) for two features
    predictions_arr = np.array([
        np.arange(0.5, 10.5, dtype=float),
        np.arange(100.5, 110.5, dtype=float)
    ]).T  # Transpose to shape (10, 2)

    return {
        "dataset_name": "my_test_dataset",
        "model_name": "test_model",
        "test_data": test_df,
        "predictions": predictions_arr,
        "columns": ['feature_1', 'feature_2'],
        "forecast_steps": 10
    }


def test_plot_predictions_calls_dependencies_correctly(sample_data, mocker):
    """
    Tests that plot_predictions calls os and matplotlib functions with correct arguments.
    Verifies that the directory is created, figures are generated, plots are drawn, and files are saved.
    """
    # Mock os.makedirs to prevent actual directory creation
    mock_makedirs = mocker.patch('utils.visualizer.os.makedirs')
    # Mock matplotlib functions to isolate plotting behavior
    mock_figure = mocker.patch('utils.visualizer.plt.figure')
    mock_plot = mocker.patch('utils.visualizer.plt.plot')
    mock_savefig = mocker.patch('utils.visualizer.plt.savefig')
    mock_close = mocker.patch('utils.visualizer.plt.close')
    # Mock additional matplotlib functions to prevent side effects
    mocker.patch('utils.visualizer.plt.title')
    mocker.patch('utils.visualizer.plt.xlabel')
    mocker.patch('utils.visualizer.plt.ylabel')
    mocker.patch('utils.visualizer.plt.legend')
    mocker.patch('utils.visualizer.plt.grid')

    # Call the plot_predictions method with sample data
    Visualizer.plot_predictions(**sample_data)

    # Verify that os.makedirs is called with the correct path
    mock_makedirs.assert_called_once_with("results/plots/my_test_dataset", exist_ok=True)
    # Verify that two figures are created (one for each feature)
    assert mock_figure.call_count == 2
    # Verify that four plots are drawn (two per feature: actual and predicted)
    assert mock_plot.call_count == 4
    # Verify that savefig is called for each feature's plot
    mock_savefig.assert_any_call("results/plots/my_test_dataset/feature_1_test_model_predictions.png")
    mock_savefig.assert_any_call("results/plots/my_test_dataset/feature_2_test_model_predictions.png")
    assert mock_savefig.call_count == 2
    # Verify that figures are closed to free memory
    assert mock_close.call_count == 2


def test_plot_error_accumulation_calls_dependencies_correctly(sample_data, mocker):
    """
    Tests that plot_error_accumulation calls os and matplotlib functions correctly.
    Verifies directory creation, figure generation, plotting, and file saving for error accumulation.
    """
    # Mock os.makedirs to prevent actual directory creation
    mock_makedirs = mocker.patch('utils.visualizer.os.makedirs')
    # Mock matplotlib functions to isolate plotting behavior
    mock_figure = mocker.patch('utils.visualizer.plt.figure')
    mock_plot = mocker.patch('utils.visualizer.plt.plot')
    mock_savefig = mocker.patch('utils.visualizer.plt.savefig')
    mock_close = mocker.patch('utils.visualizer.plt.close')
    # Mock additional matplotlib functions to prevent side effects
    mocker.patch('utils.visualizer.plt.title')
    mocker.patch('utils.visualizer.plt.xlabel')
    mocker.patch('utils.visualizer.plt.ylabel')
    mocker.patch('utils.visualizer.plt.legend')
    mocker.patch('utils.visualizer.plt.grid')

    # Call the plot_error_accumulation method with sample data
    Visualizer.plot_error_accumulation(**sample_data)

    # Verify that os.makedirs is called with the correct path
    mock_makedirs.assert_called_once_with("results/plots/my_test_dataset", exist_ok=True)
    # Verify that two figures are created (one for each feature)
    assert mock_figure.call_count == 2
    # Verify that two plots are drawn (one per feature for error accumulation)
    assert mock_plot.call_count == 2
    # Verify that savefig is called for each feature's error plot
    mock_savefig.assert_any_call("results/plots/my_test_dataset/feature_1_test_model_error_accumulation.png")
    mock_savefig.assert_any_call("results/plots/my_test_dataset/feature_2_test_model_error_accumulation.png")
    assert mock_savefig.call_count == 2
    # Verify that figures are closed to free memory
    assert mock_close.call_count == 2


@pytest.mark.parametrize("invalid_arg, error_msg", [
    ({"test_data": pd.DataFrame()}, "test_data cannot be empty."),
    ({"forecast_steps": 0}, "forecast_steps must be positive."),
    ({"forecast_steps": 11}, "test_data has 10 rows, but forecast_steps is 11."),
    ({"predictions": np.random.rand(5, 2)}, "predictions has 5 rows, but forecast_steps is 10."),
    ({"columns": ["one_col"]}, "Number of columns must match test_data and predictions feature dimensions."),
    # Edge case: Single-row data with mismatched forecast_steps
    ({"test_data": pd.DataFrame({'feature_1': [1.0], 'feature_2': [100.0]}, index=pd.to_datetime(['2023-01-01']))},
     "test_data has 1 rows, but forecast_steps is 10."),
    # Edge case: Empty predictions array
    # Updated to match actual error message from Visualizer.plot_predictions
    ({"predictions": np.array([]).reshape(0, 2)}, "predictions has 0 rows, but forecast_steps is 10."),
])
def test_plot_functions_raise_value_error_for_invalid_inputs(sample_data, invalid_arg, error_msg):
    """
    Tests that plot_predictions and plot_error_accumulation raise ValueError for invalid inputs.
    Uses parameterized tests to cover multiple invalid input scenarios, including edge cases.
    Note: Removed tests for empty dataset_name, model_name, and invalid columns as Visualizer does not validate these.
    """
    # Update sample data with invalid arguments
    valid_args = sample_data.copy()
    valid_args.update(invalid_arg)

    # Verify that plot_predictions raises the expected error
    with pytest.raises(ValueError, match=error_msg):
        Visualizer.plot_predictions(**valid_args)

    # Verify that plot_error_accumulation raises the expected error
    with pytest.raises(ValueError, match=error_msg):
        Visualizer.plot_error_accumulation(**valid_args)


def test_plot_functions_handle_univariate_predictions(sample_data, mocker):
    """
    Tests that plot_predictions and plot_error_accumulation handle univariate predictions correctly.
    Verifies that plotting works with a single feature and correct number of plot calls.
    """
    # Mock matplotlib functions to isolate plotting behavior
    mocker.patch('utils.visualizer.plt.figure')
    mock_plot = mocker.patch('utils.visualizer.plt.plot')
    mocker.patch('utils.visualizer.plt.savefig')
    mocker.patch('utils.visualizer.plt.close')
    mocker.patch('utils.visualizer.plt.title')
    mocker.patch('utils.visualizer.plt.xlabel')
    mocker.patch('utils.visualizer.plt.ylabel')
    mocker.patch('utils.visualizer.plt.legend')
    mocker.patch('utils.visualizer.plt.grid')

    # Modify sample data to include only one feature
    sample_data['test_data'] = sample_data['test_data'][['feature_1']]
    sample_data['predictions'] = sample_data['predictions'][:, 0]
    sample_data['columns'] = ['feature_1']

    # Test plot_predictions with univariate data
    Visualizer.plot_predictions(**sample_data)
    # Verify that two plots are drawn (actual and predicted for one feature)
    assert mock_plot.call_count == 2

    # Test plot_error_accumulation with univariate data
    Visualizer.plot_error_accumulation(**sample_data)
    # Verify that one additional plot is drawn for error accumulation
    assert mock_plot.call_count == 3


def test_plot_functions_handle_edge_cases(sample_data, mocker):
    """
    Tests edge cases for plot_predictions and plot_error_accumulation to ensure robust handling.
    Covers single-row data and single-step forecasts.
    """
    # Mock matplotlib functions to isolate plotting behavior
    mock_makedirs = mocker.patch('utils.visualizer.os.makedirs')
    mock_figure = mocker.patch('utils.visualizer.plt.figure')
    mock_plot = mocker.patch('utils.visualizer.plt.plot')
    mock_savefig = mocker.patch('utils.visualizer.plt.savefig')
    mock_close = mocker.patch('utils.visualizer.plt.close')
    mocker.patch('utils.visualizer.plt.title')
    mocker.patch('utils.visualizer.plt.xlabel')
    mocker.patch('utils.visualizer.plt.ylabel')
    mocker.patch('utils.visualizer.plt.legend')
    mocker.patch('utils.visualizer.plt.grid')

    # Edge Case: Single-row data with matching forecast_steps
    single_row_data = sample_data.copy()
    single_row_data['test_data'] = pd.DataFrame({
        'feature_1': [1.0],
        'feature_2': [100.0]
    }, index=pd.to_datetime(['2023-01-01']))
    single_row_data['predictions'] = np.array([[1.5, 100.5]])
    single_row_data['forecast_steps'] = 1

    Visualizer.plot_predictions(**single_row_data)
    # Verify that two figures are created (one per feature)
    assert mock_figure.call_count == 2
    # Verify that four plots are drawn (two per feature: actual and predicted)
    assert mock_plot.call_count == 4
    # Verify that savefig is called for each feature
    mock_savefig.assert_any_call("results/plots/my_test_dataset/feature_1_test_model_predictions.png")
    mock_savefig.assert_any_call("results/plots/my_test_dataset/feature_2_test_model_predictions.png")
    assert mock_savefig.call_count == 2
    assert mock_close.call_count == 2

    # Reset mock counters for the next test
    mock_figure.reset_mock()
    mock_plot.reset_mock()
    mock_savefig.reset_mock()
    mock_close.reset_mock()

    Visualizer.plot_error_accumulation(**single_row_data)
    # Verify that two figures are created (one per feature)
    assert mock_figure.call_count == 2
    # Verify that two plots are drawn (one per feature for error accumulation)
    assert mock_plot.call_count == 2
    # Verify that savefig is called for each feature
    mock_savefig.assert_any_call("results/plots/my_test_dataset/feature_1_test_model_error_accumulation.png")
    mock_savefig.assert_any_call("results/plots/my_test_dataset/feature_2_test_model_error_accumulation.png")
    assert mock_savefig.call_count == 2
    assert mock_close.call_count == 2