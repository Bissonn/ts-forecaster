import pytest
import pandas as pd
import numpy as np
from typing import Optional
from utils.preprocessor import Preprocessor


# --- Test Helpers & Fixtures ---

def _create_seasonal_series(period: int, n_periods: int, name: str, trend_start: float = 0.0) -> pd.Series:
    """Helper to create a series with a clear seasonal pattern and a DatetimeIndex."""
    base = np.arange(period * n_periods, dtype=float) * 0.1 + trend_start
    index = pd.to_datetime(pd.date_range(start='2020-01-01', periods=len(base), freq='D'))
    for i in range(period, len(base)):
        base[i] += base[i - period] * 0.5  # Add some seasonality
    return pd.Series(base, name=name, index=index)


def _run_roundtrip_test(preprocessor: Preprocessor, series: pd.Series):
    """
    Helper for a full forward and inverse transformation cycle (round-trip).
    This tests the automatic detection of in-sample data when a DataFrame is passed.
    """
    col_name = series.name
    context_df = pd.DataFrame(series)

    # 1. Apply forward transform
    transformed_df, _, _ = preprocessor.apply_transforms(
        context_df, pd.DataFrame(), pd.DataFrame(), full_raw_data_context=context_df
    )

    # 2. Inverse transform by passing the DataFrame with its original index
    reconstructed_df = preprocessor.inverse_transforms(transformed_df)  # No start_after needed
    reconstructed_values = reconstructed_df[col_name].to_numpy()

    # 3. Compare against the tail of the original series
    expected_values = series.iloc[-len(reconstructed_values):].to_numpy()
    np.testing.assert_allclose(expected_values, reconstructed_values, rtol=1e-5, atol=1e-5)


# --- Basic Initialization and Validation Tests ---

def test_initialization_and_defaults():
    """Test that the preprocessor initializes correctly and sets defaults."""
    config = {'scaling': {'enabled': True}, 'differencing': {'enabled': True, 'auto': 'none'}}
    preprocessor = Preprocessor(config)
    assert preprocessor.preprocessing_steps['scaling']['method'] == 'minmax'
    assert preprocessor.preprocessing_steps['differencing']['order'] == 1


def test_validation_raises_on_invalid_data():
    """Test that input validation correctly raises errors for NaN or Inf data."""
    config = {'scaling': {'enabled': True}}
    preprocessor = Preprocessor(config)
    df_nan = pd.DataFrame({'a': [1.0, np.nan]})
    df_inf = pd.DataFrame({'a': [1.0, np.inf]})
    with pytest.raises(ValueError, match="NaN values found"):
        preprocessor.apply_transforms(df_nan, pd.DataFrame(), pd.DataFrame(), df_nan)
    with pytest.raises(ValueError, match="Infinite values found"):
        preprocessor.apply_transforms(df_inf, pd.DataFrame(), pd.DataFrame(), df_inf)


def test_invalid_config_raises_error():
    """Test that an invalid configuration raises a ValueError."""
    config = {'scaling': {'enabled': True, 'method': 'invalid_method'}}
    with pytest.raises(ValueError, match="Configuration validation failed"):
        Preprocessor(config)


# --- Round-Trip Reconstruction Tests (DataFrame In -> DataFrame Out) ---

def test_multivariate_roundtrip():
    """Tests forward and inverse transform for a multivariate DataFrame."""
    config = {'scaling': {'enabled': True, 'method': 'standard'}}
    preprocessor = Preprocessor(config)
    index = pd.date_range(start='2020-01-01', periods=100, freq='D')
    original_df = pd.DataFrame({
        'feature_A': np.linspace(10, 50, 100),
        'feature_B': np.linspace(200, 100, 100)
    }, index=index)

    transformed_df, _, _ = preprocessor.apply_transforms(
        original_df, pd.DataFrame(), pd.DataFrame(), full_raw_data_context=original_df
    )
    reconstructed_df = preprocessor.inverse_transforms(transformed_df)
    pd.testing.assert_frame_equal(original_df, reconstructed_df, rtol=1e-5)


def test_full_pipeline_roundtrip_log_scale_diff():
    """Test a round-trip with log, scaling, and seasonal/standard differencing."""
    config = {
        'log_transform': {'enabled': True},
        'scaling': {'enabled': True},
        'differencing': {'enabled': True, 'auto': 'none', 'order': 1, 'seasonal_order': 1, 'seasonal_period': 7}
    }
    preprocessor = Preprocessor(config)
    series = _create_seasonal_series(period=7, n_periods=12, name='test_series', trend_start=100)
    _run_roundtrip_test(preprocessor, series)


def test_auto_differencing_roundtrip():
    """Test round-trip reconstruction with automatic differencing (ADF)."""
    config = {'differencing': {'enabled': True, 'auto': 'adf', 'max_d': 2}}
    preprocessor = Preprocessor(config)
    series = pd.Series(
        np.arange(100, dtype=float) ** 2,
        name='non_stationary',
        index=pd.date_range(start='2020-01-01', periods=100, freq='D')
    )
    _run_roundtrip_test(preprocessor, series)


# --- Future Prediction Tests (NumPy In -> DataFrame Out) ---

def test_future_prediction_with_start_after():
    """
    Test a realistic forecasting scenario: inverse-transforming a raw NumPy array
    using an explicit start_after timestamp.
    """
    config = {
        'scaling': {'enabled': True, 'method': 'minmax'},
        'differencing': {'enabled': True, 'auto': 'none', 'order': 1}
    }
    preprocessor = Preprocessor(config)
    full_series = pd.Series(
        np.linspace(100, 150, 50),
        name='future_test',
        index=pd.date_range(start='2020-01-01', periods=50, freq='D')
    )

    # 1. Split data
    history_df = pd.DataFrame(full_series.iloc[:40])
    future_df = pd.DataFrame(full_series.iloc[40:])

    # 2. Fit preprocessor on historical data
    preprocessor.apply_transforms(history_df, pd.DataFrame(), pd.DataFrame(), full_raw_data_context=history_df)

    # 3. Simulate model prediction on future data
    transformed_future = preprocessor.transform(future_df)
    predictions_array = transformed_future.to_numpy()  # Raw model output

    # 4. Inverse-transform using the explicit start_after timestamp
    last_history_ts = history_df.index[-1]
    reconstructed_df = preprocessor.inverse_transforms(
        predictions_array,
        start_after=last_history_ts
    )

    # 5. Verify correctness of values and index
    pd.testing.assert_frame_equal(future_df, reconstructed_df, rtol=1e-5)


@pytest.mark.parametrize("seasonal_period", [3, 7, 12])
def test_future_prediction_with_seasonal_diff(seasonal_period):
    """Test future prediction with combined seasonal and standard differencing."""
    config = {
        'differencing': {
            'enabled': True, 'auto': 'none',
            'order': 1, 'seasonal_order': 1, 'seasonal_period': seasonal_period
        }
    }
    pre = Preprocessor(config)
    series = _create_seasonal_series(period=seasonal_period, n_periods=10, name='y', trend_start=100)
    history_df = series.to_frame()
    pre.apply_transforms(history_df, pd.DataFrame(), pd.DataFrame(), history_df)

    # Simulate a simple forecast of "no change" after differencing
    future_diffs = np.zeros((10, 1), dtype=float)

    reconstructed_df = pre.inverse_transforms(future_diffs, start_after=history_df.index[-1])

    # The reconstructed values should continue the trend and seasonality
    # A simple check is that the first predicted value is close to the last historical one
    last_historical_value = series.iloc[-1]
    first_reconstructed_value = reconstructed_df['y'].iloc[0]
    # This check is sensitive, so we use a larger tolerance
    assert abs(first_reconstructed_value - last_historical_value) < 1.0


# --- Edge Case and API Behavior Tests ---

def test_numpy_input_without_start_after_raises_error():
    """Verify that passing a NumPy array without `start_after` raises a ValueError."""
    config = {'scaling': {'enabled': True}}
    preprocessor = Preprocessor(config)
    history_df = pd.DataFrame(
        {'a': range(10)},
        index=pd.date_range(start='2020-01-01', periods=10, freq='D')
    )
    preprocessor.apply_transforms(history_df, pd.DataFrame(), pd.DataFrame(), history_df)

    ambiguous_input = np.ones((5, 1))

    with pytest.raises(ValueError, match="Please provide the 'start_after' timestamp"):
        preprocessor.inverse_transforms(ambiguous_input)


def test_unfitted_preprocessor_raises_error():
    """Verify that calling inverse_transforms on an unfitted preprocessor raises a RuntimeError."""
    preprocessor = Preprocessor({'scaling': {'enabled': True}})
    with pytest.raises(RuntimeError, match="Preprocessor has not been fitted"):
        preprocessor.inverse_transforms(np.ones((5, 1)), start_after=pd.Timestamp('2020-01-01'))


def test_reconstruction_of_historical_fragment():
    """
    Test the advanced use case of reconstructing a fragment from the middle of the
    historical data, relying on the DataFrame's index for context.
    NOTE: This requires the helper functions (_apply_inverse_differencing) to be implemented
    to correctly select the anchor based on the fragment's start date.
    """
    config = {
        'scaling': {'enabled': True},
        'differencing': {'enabled': True, 'order': 1, 'seasonal_order': 1, 'seasonal_period': 7}
    }
    pre = Preprocessor(config)
    series = _create_seasonal_series(period=7, n_periods=15, name='fragment_test', trend_start=50)

    # Fit on the full history
    transformed_df, _, _ = pre.apply_transforms(
        series.to_frame(), pd.DataFrame(), pd.DataFrame(), series.to_frame()
    )

    # Select a fragment from the *middle* of the transformed data
    fragment_transformed = transformed_df.iloc[20:35]

    # Reconstruct this fragment
    reconstructed_fragment_df = pre.inverse_transforms(fragment_transformed)

    # The reconstructed fragment should match the corresponding original slice
    original_fragment_to_compare = series.to_frame().loc[reconstructed_fragment_df.index]

    pd.testing.assert_frame_equal(original_fragment_to_compare, reconstructed_fragment_df, rtol=1e-5)

def test_seasonal_inverse_uses_exactly_last_s_anchors():
    cfg = {"differencing": {"enabled": True, "auto": "none", "order": 0, "seasonal_order": 1, "seasonal_period": 4}}
    pre = Preprocessor(cfg)
    idx = pd.date_range("2020-01-01", periods=12, freq="D")
    # deterministic signal
    y = pd.Series(np.arange(12, dtype=float), index=idx, name="y")
    pre.apply_transforms(y.to_frame(), pd.DataFrame(), pd.DataFrame(), y.to_frame())
    # seasonal diffs of 0 -> after reversing we should get exactly last anchor + 0
    zeros = np.zeros((3, 1))
    rec = pre.inverse_transforms(zeros, start_after=idx[-1])
    # The first 4 anchors for the forecast are y[-4:]
    assert np.isclose(rec.iloc[0, 0], y.iloc[-4])

def test_fragment_inverse_respects_index_cut():
    cfg = {"differencing": {"enabled": True, "auto": "none", "order": 1, "seasonal_order": 1, "seasonal_period": 7}}
    pre = Preprocessor(cfg)
    s = pd.Series(np.arange(100, dtype=float), index=pd.date_range("2020-01-01", periods=100, freq="D"), name="y")
    tr, _, _ = pre.apply_transforms(s.to_frame(), pd.DataFrame(), pd.DataFrame(), s.to_frame())
    frag = tr.iloc[30:50]
    rec = pre.inverse_transforms(frag)
    pd.testing.assert_frame_equal(rec, s.to_frame().loc[frag.index])
