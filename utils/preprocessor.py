"""Module for preprocessing time series data in the forecasting framework.

This module provides the Preprocessor class to apply and invert transformations like
log transform, winsorization, scaling, and differencing, ensuring consistency across
training, validation, and test datasets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats.mstats import winsorize
from schema import SchemaError
from utils.config_utils import validate_preprocessing

# Set up a logger for the module
logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Orchestrates a series of preprocessing transformations for time series data.

    This class provides a configurable pipeline to apply and invert common
    time series transformations like logging, winsorizing, scaling, and
    differencing. It is designed to be fitted on a training dataset and then
    apply the same transformations to validation, test, or new data to ensure
    consistency. It also supports inverting these transformations on model
    predictions to return them to their original scale.

    Attributes:
        config (Dict): The configuration dictionary that defines which steps are
            enabled and their parameters.
        scalers (Dict[str, Any]): Stores the fitted scaler object for each column.
        diff_orders (Dict[str, int]): Stores the determined standard differencing
            order (d) for each column.
        seasonal_diff_orders (Dict[str, int]): Stores the determined seasonal
            differencing order (D) for each column.
        active_steps (List[str]): A sorted list of preprocessing steps that are
            enabled in the configuration.
    """
    def __init__(self, config: Dict):
        """
        Initializes the Preprocessor with a given configuration.

        Args:
            config (Dict): A dictionary specifying the preprocessing steps and
                their parameters.
        """
        logger.debug(f"Received config for validation: {config}")
        try:
            validate_preprocessing(config)
        except SchemaError as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise ValueError(f"Configuration validation failed: {str(e)}")
        self.config = config
        logger.info(f"Initialized Preprocessor with config: {self.config}")

        self.scalers: Dict[str, Any] = {}
        self.diff_orders: Dict[str, int] = {}
        self.seasonal_diff_orders: Dict[str, int] = {}
        self._full_raw_data_context: pd.DataFrame = pd.DataFrame()

        step_names = ["log_transform", "winsorize", "scaling", "differencing"]

        self.preprocessing_steps = {
            name: {
                **self.config.get(name, {}),
                "enabled": self.config.get(name, {}).get("enabled", False),
            }
            for name in step_names
        }
        self._initialize_preprocessing_defaults()

        self.active_steps = [
            step for step, settings in self.preprocessing_steps.items() if settings.get("enabled", False)
        ]
        self.active_steps.sort(key=self._get_step_order)
        logger.info(
            f"Active preprocessing steps: {', '.join(self.active_steps) if self.active_steps else 'None'}"
        )

    # ---------------------------------------------------------------------------------
    # Defaults & validation
    # ---------------------------------------------------------------------------------

    def _initialize_preprocessing_defaults(self) -> None:
        """
        Fills missing fields in the preprocessing config with default values.

        This method ensures that if a preprocessing step is enabled, all its
        necessary parameters have a default value, preventing errors and
        simplifying the user configuration.
        """
        defaults = {
            "log_transform": {"method": "log1p", "epsilon": 1e-6},
            "winsorize": {"limits": [0.01, 0.01]},
            "scaling": {"method": "minmax", "range": [0, 1]},
            "differencing": {"auto": "none"},
        }
        differencing_defaults = {
            "none": {
                "order": 1,
                "seasonal_order": 0,
                "seasonal_period": 1,
                "max_d": 2,
                "max_D": 1,
                "p_value_threshold": 0.05,
            },
            "adf": {
                "order": 0,
                "seasonal_order": 0,
                "seasonal_period": 1,
                "max_d": 2,
                "max_D": 1,
                "p_value_threshold": 0.05,
            },
            "kpss": {
                "order": 0,
                "seasonal_order": 0,
                "seasonal_period": 1,
                "max_d": 2,
                "max_D": 1,
                "p_value_threshold": 0.05,
            },
        }
        for step, step_defaults in defaults.items():
            if self.preprocessing_steps.get(step, {}).get("enabled", False):
                for key, value in step_defaults.items():
                    self.preprocessing_steps[step].setdefault(key, value)
                if step == "differencing":
                    auto_method = self.preprocessing_steps[step]["auto"]
                    for key, value in differencing_defaults[auto_method].items():
                        self.preprocessing_steps[step].setdefault(key, value)

    def _get_step_order(self, step_name: str) -> int:
        """
        Defines the execution order of preprocessing steps for consistent application/inversion.

        Args:
            step_name (str): The name of the preprocessing step.

        Returns:
            int: The execution order number for the step.
        """
        order = {"log_transform": 1, "winsorize": 2, "scaling": 3, "differencing": 4}
        return order.get(step_name, 99)

    def _validate_dataframe(self, df: pd.DataFrame, context: str) -> None:
        """
        Validates the input DataFrame for NaN or Inf values.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            context (str): A string describing the source of the data (e.g.,
                "training", "test") for use in the error message.

        Raises:
            ValueError: If any NaN or infinite values are found in the DataFrame.
        """
        if df.isnull().to_numpy().any():
            raise ValueError(
                f"NaN values found in '{context}' DataFrame. Please handle them before preprocessing."
            )
        if np.isinf(df.to_numpy()).any():
            raise ValueError(
                f"Infinite values found in '{context}' DataFrame. Please handle them before preprocessing."
            )

    # ---------------------------------------------------------------------------------
    # Atomic transforms
    # ---------------------------------------------------------------------------------

    def _apply_log_transform(self, series: pd.Series) -> pd.Series:
        """Applies a natural logarithm transformation (log1p)."""
        if (series + self.preprocessing_steps["log_transform"]["epsilon"]).min() < 0:
            raise ValueError(
                "Log transform cannot be applied to series with negative values that would result in log of a non-positive number.")
        epsilon = self.preprocessing_steps["log_transform"]["epsilon"]
        return np.log1p(series + epsilon)

    def _apply_inverse_log_transform(self, series: pd.Series) -> pd.Series:
        """Applies the inverse of the logarithm transformation (expm1)."""
        if not self.preprocessing_steps["log_transform"]["enabled"]:
            return series
        epsilon = self.preprocessing_steps["log_transform"]["epsilon"]
        return np.expm1(series) - epsilon

    def _apply_winsorize(self, series: pd.Series) -> pd.Series:
        """Clips outliers in the series using winsorization."""
        if series.empty:
            return series
        limits = self.preprocessing_steps["winsorize"]["limits"]
        winsorized_array = winsorize(series.to_numpy().astype(float), limits=limits).data.astype(float)
        return pd.Series(winsorized_array, index=series.index, name=series.name)

    def _apply_scaling(
        self, series: pd.Series, column_name: str, method: str, fit_scaler: bool = True
    ) -> pd.Series:
        """
        Scales the data using either StandardScaler or MinMaxScaler.

        Args:
            series (pd.Series): The data series to scale.
            column_name (str): The name of the column, used as a key for storing the scaler.
            method (str): The scaling method, either 'standard' or 'minmax'.
            fit_scaler (bool): If True, fits a new scaler. If False, transforms using an existing one.

        Returns:
            pd.Series: The scaled data series.
        """
        if series.empty:
            return series
        data_array = series.to_numpy().reshape(-1, 1)
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            settings = self.preprocessing_steps["scaling"]
            scaler = MinMaxScaler(feature_range=tuple(settings.get("range", [0, 1])))
        else:
            return series
        if fit_scaler:
            scaled_data = scaler.fit_transform(data_array)
            self.scalers[column_name] = scaler
        else:
            if column_name not in self.scalers:
                raise ValueError(
                    f"Scaler for column {column_name} not found. Fit the scaler on training data first."
                )
            scaler = self.scalers[column_name]
            scaled_data = scaler.transform(data_array)
        return pd.Series(scaled_data.flatten(), index=series.index, name=series.name)

    def _apply_inverse_scaling(self, series: pd.Series, column_name: str) -> pd.Series:
        """
        Applies the inverse scaling transformation to a series.
        
        Args:
            series (pd.Series): The scaled data series.
            column_name (str): The name of the column to find the correct scaler.

        Returns:
            pd.Series: The data series reverted to its pre-scaled state.
        """
        if series.empty or column_name not in self.scalers:
            return series
        scaler = self.scalers[column_name]
        inverse_scaled_data = scaler.inverse_transform(series.to_numpy().reshape(-1, 1))
        return pd.Series(inverse_scaled_data.flatten(), index=series.index, name=series.name)

    # ---------------------------------------------------------------------------------
    # Stationarity & differencing
    # ---------------------------------------------------------------------------------

    def _test_stationarity(self, series: pd.Series, test_method: str = "adf") -> bool:
        """
        Tests if a time series is stationary using ADF or KPSS test.

        Args:
            series (pd.Series): The time series to test.
            test_method (str): The test to use ('adf' or 'kpss').

        Returns:
            bool: True if the series is stationary, False otherwise.
        """
        p_value_threshold = self.preprocessing_steps["differencing"]["p_value_threshold"]
        if series.nunique() <= 1 or len(series.dropna()) < 10:
            return True
        try:
            if test_method == "adf":
                return adfuller(series.dropna())[1] < p_value_threshold
            elif test_method == "kpss":
                return kpss(series.dropna(), regression="c")[1] > p_value_threshold
            return False
        except Exception:
            return False

    def _determine_differencing_order(self, series: pd.Series, is_seasonal: bool = False, period: int = 1) -> int:
        """
        Automatically determines the required order of differencing.

        Args:
            series (pd.Series): The data series.
            is_seasonal (bool): True if determining seasonal differencing order.
            period (int): The differencing period (e.g., 1 for standard, or > 1 for seasonal).

        Returns:
            int: The minimum differencing order required to make the series stationary.
        """
        settings = self.preprocessing_steps["differencing"]
        auto_mode = settings.get("auto", "none")
        if auto_mode == "none":
            return settings.get("seasonal_order" if is_seasonal else "order", 0)

        max_order = settings["max_D"] if is_seasonal else settings["max_d"]
        temp_series = series.copy()
        for i in range(1, max_order + 1):
            if len(temp_series.dropna()) <= period:
                return i - 1
            temp_series = temp_series.diff(periods=period).dropna()
            if self._test_stationarity(temp_series, test_method=auto_mode):
                return i
        return max_order

    def _apply_differencing(self, series: pd.Series, column_name: str, fit_differencing: bool = True) -> pd.Series:
        """
        Applies standard and seasonal differencing.

        Args:
            series (pd.Series): The data series.
            column_name (str): The name of the column.
            fit_differencing (bool): If True, determines and stores the orders.

        Returns:
            pd.Series: The differenced series with NaN values dropped.
        """
        settings = self.preprocessing_steps["differencing"]
        seasonal_period = settings.get("seasonal_period", 1)
        work_series = series.copy()

        if fit_differencing:
            s_d_order = (
                self._determine_differencing_order(work_series, is_seasonal=True, period=seasonal_period)
                if seasonal_period > 1
                else 0
            )
            self.seasonal_diff_orders[column_name] = s_d_order

            temp_for_d = work_series.copy()
            for _ in range(s_d_order):
                temp_for_d = temp_for_d.diff(periods=seasonal_period)

            d_order = self._determine_differencing_order(temp_for_d.dropna())
            self.diff_orders[column_name] = d_order
        else:
            s_d_order = self.seasonal_diff_orders.get(column_name, 0)
            d_order = self.diff_orders.get(column_name, 0)

        for _ in range(s_d_order):
            work_series = work_series.diff(periods=seasonal_period)
        for _ in range(d_order):
            work_series = work_series.diff(periods=1)

        return work_series.dropna()

    # ---------------------------------------------------------------------------------
    # Forward pipeline dispatcher
    # ---------------------------------------------------------------------------------

    def _apply_single_transform(
        self, series: pd.Series, step_name: str, column_name: str, fit: bool = False
    ) -> pd.Series:
        """
        Dispatches a single preprocessing step to the correct method.

        Args:
            series (pd.Series): The data series.
            step_name (str): The name of the step to apply.
            column_name (str): The name of the column.
            fit (bool): Passed to underlying methods to indicate fitting vs. transforming.

        Returns:
            pd.Series: The transformed series.
        """
        settings = self.preprocessing_steps.get(step_name, {})
        if not settings.get("enabled", False):
            return series
        if step_name == "log_transform":
            return self._apply_log_transform(series)
        elif step_name == "winsorize":
            return self._apply_winsorize(series)
        elif step_name == "scaling":
            return self._apply_scaling(
                series, column_name, method=self.preprocessing_steps[step_name]["method"], fit_scaler=fit
            )
        elif step_name == "differencing":
            return self._apply_differencing(series, column_name, fit_differencing=fit)
        return series

    # ---------------------------------------------------------------------------------
    # Inverse differencing (final, deduplicated, autoregressive)
    # ---------------------------------------------------------------------------------
    
    def _slice_context_for(
        self,
        series: pd.Series,
        context: pd.Series,
        d_order: int,
        s_d_order: int,
        seasonal_period: int,
    ) -> pd.Series:
        """
        Trim the context to the minimally required tail and avoid index overlap
        with the segment we are reconstructing.
        - If `series` has a DatetimeIndex, keep only strictly earlier timestamps
          than the first timestamp of `series`. This prevents duplicated indices
          when concatenating (history + reconstruction).
        - Keep only the minimal number of anchor points needed to seed inversion.
        """
        if context is None or context.empty:
            return context
        ctx = context.copy()
        if isinstance(series.index, pd.DatetimeIndex) and len(series.index) > 0:
            start_ts = series.index[0]
            # keep strictly earlier history only
            ctx = ctx.loc[ctx.index < start_ts]
        # minimal anchors required for inversion
        min_len = max(1, d_order + s_d_order * max(1, seasonal_period))
        if len(ctx) > min_len:
            ctx = ctx.iloc[-min_len:]
        return ctx

    def _inverse_standard_diff_autoregressive(
        self, series: pd.Series, d_order: int, context_series: pd.Series
    ) -> pd.Series:
        """
        Invert **standard** differencing of order `d` in an autoregressive fashion.
        We go one order at a time. At level k (from d down to 1) we take the context
        differenced (k-1) times, then invert a 1st difference via cumulative sum
        with the last known value as a seed:
            x_t = y_t + x_{t-1}  ->  x = cumsum(y) + seed
        Using `cumsum()+seed` is numerically stable and avoids issues with in-place
        writes through `iloc`.
        """
        if d_order == 0 or series.empty:
            return series

        rec = series.copy()
        for k in range(d_order, 0, -1):
            # context for this level = (k-1)-times differenced
            ctx_k = context_series.copy()
            for _ in range(k - 1):
                ctx_k = ctx_k.diff().dropna()
            if ctx_k.empty:
                raise ValueError("Context too short for standard inverse differencing.")
            seed = float(ctx_k.iloc[-1])
            # invert 1st difference
            rec = rec.cumsum() + seed
        return rec

    def _calculate_inverse_seasonal_diff(
            self,
            series: pd.Series,
            s_d_order: int,
            seasonal_period: int,
            context_series: pd.Series,
            is_future: bool = False,
    ) -> pd.Series:
        """
        Invert **seasonal** differencing (order D, period s) in an autoregressive fashion.
        """
        if s_d_order == 0 or series.empty:
            return series
        if seasonal_period <= 0:
            raise ValueError("seasonal_period must be > 0")

        rec = series.copy()
        # Invert from the highest seasonal order down to 1
        for k in range(s_d_order, 0, -1):
            ctx_k = context_series.copy()
            # Context for this level = (k-1)-times seasonally differenced
            for _ in range(k - 1):
                ctx_k = ctx_k.diff(periods=seasonal_period).dropna()

            if is_future:
                if len(ctx_k) < seasonal_period:
                    raise ValueError("Context too short for seasonal inverse differencing.")
                ctx_k = ctx_k.iloc[-seasonal_period:]
            else:
                ctx_k = self._slice_context_for(rec, ctx_k, d_order=0, s_d_order=1, seasonal_period=seasonal_period)
                if len(ctx_k) < seasonal_period:
                    raise ValueError("Context too short for seasonal inverse differencing.")

            combined = pd.concat([ctx_k, rec])
            original_diffs = rec.copy()
            rec_idx_ptr = 0

            start_combined_idx = len(ctx_k)
            for t in range(start_combined_idx, len(combined)):
                seed_value = combined.iloc[t - seasonal_period]
                combined.iloc[t] = original_diffs.iloc[rec_idx_ptr] + seed_value
                rec_idx_ptr += 1

            rec = combined.iloc[-len(series):]

        return rec

    def _perform_inverse_differencing(
            self,
            predictions_series: pd.Series,
            d_order: int,
            s_d_order: int,
            seasonal_period: int,
            context_series: pd.Series,
    ) -> pd.Series:
        """
        Reconstructs the original series from differenced predictions by inverting
        both standard and seasonal differencing.

        The forward transformation pipeline is applied as:
        `transformed_y = standard_diff^d(seasonal_diff^D(original_y))`
        Therefore, the inversion must be performed in the reverse order:
        1. Invert standard differencing (d), using a seasonally-differenced context.
        2. Invert seasonal differencing (D), using the original context.

        Args:
            predictions_series: The series of differenced predictions to reconstruct.
            d_order: The order of standard differencing (d) to invert.
            s_d_order: The order of seasonal differencing (D) to invert.
            seasonal_period: The seasonal period (s) used for differencing.
            context_series: The historical data series on the same scale as the
                           data *before* any differencing was applied. This serves
                           as the anchor for the reconstruction.

        Returns:
            The reconstructed series, transformed back to its pre-differenced scale.
        """
        if predictions_series.empty or (d_order == 0 and s_d_order == 0):
            return predictions_series.copy()

        if context_series is None or context_series.empty:
            raise ValueError("A valid context_series is required for inverse differencing.")

        original_context = context_series

        is_future = False
        try:
            is_future = (
                    isinstance(predictions_series.index, pd.DatetimeIndex)
                    and isinstance(original_context.index, pd.DatetimeIndex)
                    and len(predictions_series) > 0
                    and len(original_context) > 0
                    and predictions_series.index[0] > original_context.index[-1]
            )
        except Exception:
            is_future = False

        # Prepare the context for the standard inversion step. It must be on the
        # same scale as the data that standard differencing was originally applied to,
        # i.e., seasonally differenced data.
        context_for_std_inverse = original_context.copy()
        if d_order > 0 and s_d_order > 0:
            for _ in range(s_d_order):
                context_for_std_inverse = context_for_std_inverse.diff(periods=seasonal_period)
            context_for_std_inverse.dropna(inplace=True)

        # --- Step 1: Invert STANDARD differencing (d) ---
        if d_order > 0:
            std_anchor_ctx = self._slice_context_for(
                predictions_series, context_for_std_inverse, d_order=d_order, s_d_order=0, seasonal_period=1
            )
            reconstructed_series = self._inverse_standard_diff_autoregressive(
                predictions_series, d_order, std_anchor_ctx
            )
        else:
            reconstructed_series = predictions_series.copy()

        # --- Step 2: Invert SEASONAL differencing (D) ---
        if s_d_order > 0:
            reconstructed_series = self._calculate_inverse_seasonal_diff(
                reconstructed_series, s_d_order, seasonal_period, original_context, is_future=is_future
            )

        return reconstructed_series

    def _prepare_context_for_inverse_diff(self, context_df: pd.DataFrame, column_name: str) -> pd.Series:
        """
        Prepares a given historical context DataFrame for inverse differencing.

        This method takes the provided context data and applies all transformations
        that occur in order *before* the differencing step (e.g., logging, scaling).

        Args:
            context_df: The historical DataFrame to prepare. This can be the full
                        history for a future forecast or a subset for an in-sample fragment.
            column_name: The name of the column for which to prepare the context.

        Returns:
            The transformed historical data series, ready to be used as an anchor.
        """
        if context_df.empty:
            return pd.Series(dtype=float)

        context_series = context_df[column_name].copy()
        for step_name in self.active_steps:
            if self._get_step_order(step_name) >= self._get_step_order("differencing"):
                break
            # Use fit=False as scalers and other stateful transformers should already be fitted.
            context_series = self._apply_single_transform(context_series, step_name, column_name, fit=False)
        return context_series

    def _apply_inverse_differencing(self, predictions_series: pd.Series, column_name: str) -> pd.Series:
        """
        Handles the entire inverse differencing process, including dynamic context preparation.

        This method intelligently selects the correct historical context based on the
        timestamp of the `predictions_series`. For future predictions, it uses the full
        historical context. For in-sample (historical) fragments, it uses only the data
        that occurred *prior* to the fragment's start date.

        Args:
            predictions_series: The series of predictions to be inverse-differenced.
                                Must have a DatetimeIndex.
            column_name: The name of the target column.

        Returns:
            The predictions series after inverse differencing has been applied.
        """
        diff_settings = self.preprocessing_steps["differencing"]
        if not diff_settings.get("enabled", False):
            return predictions_series

        d_order = self.diff_orders.get(column_name, 0)
        s_d_order = self.seasonal_diff_orders.get(column_name, 0)

        if d_order == 0 and s_d_order == 0:
            return predictions_series

        # --- Dynamic context selection logic ---
        last_history_ts = self._full_raw_data_context.index[-1]
        prediction_start_ts = predictions_series.index[0]

        if prediction_start_ts > last_history_ts:
            # Scenario A: Future forecast. Use the full history as context.
            context_df_to_prepare = self._full_raw_data_context
        else:
            # Scenario B: In-sample (historical) fragment. Use only data *before* the fragment.
            context_df_to_prepare = self._full_raw_data_context.loc[
                self._full_raw_data_context.index < prediction_start_ts]

        if context_df_to_prepare.empty:
            raise ValueError(
                f"Cannot find historical context for predictions starting at {prediction_start_ts}. "
                "The prediction start date may be too early or the context data is missing."
            )

        # Prepare the dynamically selected context to create the anchor.
        prepared_context = self._prepare_context_for_inverse_diff(context_df_to_prepare, column_name)

        seasonal_period = int(diff_settings.get("seasonal_period", 1))

        return self._perform_inverse_differencing(
            predictions_series, d_order, s_d_order, seasonal_period, prepared_context
        )

    # ---------------------------------------------------------------------------------
    # Inverse of other transforms
    # ---------------------------------------------------------------------------------

    def _inverse_single_transform(self, series: pd.Series, step_name: str, column_name: str) -> pd.Series:
        """
        Dispatches a single inverse preprocessing step to the correct method.

        Args:
            series (pd.Series): The data series to be inverse-transformed.
            step_name (str): The name of the step to invert.
            column_name (str): The name of the column.

        Returns:
            pd.Series: The inverse-transformed series.
        """
        if step_name == "scaling":
            return self._apply_inverse_scaling(series, column_name)
        elif step_name == "log_transform":
            return self._apply_inverse_log_transform(series)
        # Winsorize is non-invertible, and differencing is handled separately.
        return series

    def _inverse_transforms_single_column(self, predictions_series: pd.Series, column_name: str) -> np.ndarray:
        """
        Applies all inverse transformations to a single column of data.

        Args:
            predictions_series (pd.Series): A Series containing the predicted values for a
                single column, with a proper DatetimeIndex.
            column_name (str): The name of the column to be reconstructed.

        Returns:
            np.ndarray: A 1D numpy array containing the predictions for the
                specified column, transformed back to their original scale.
        """
        series = predictions_series.copy()

        # 1. Inverse differencing (requires a proper DatetimeIndex).
        series = self._apply_inverse_differencing(series, column_name)

        # 2. Inverse point-wise transforms in reverse order.
        for step_name in reversed(self.active_steps):
            if step_name == "differencing":
                continue
            series = self._inverse_single_transform(series, step_name, column_name)

        return series.to_numpy()

    # ---------------------------------------------------------------------------------
    # Public API: forward / inverse
    # ---------------------------------------------------------------------------------

    def apply_transforms(
        self,
        train_series: pd.DataFrame,
        val_series: pd.DataFrame,
        test_series: pd.DataFrame,
        full_raw_data_context: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Applies the full pipeline of transformations to train, validation, and test sets.

        Args:
            train_series (pd.DataFrame): The training dataset.
            val_series (pd.DataFrame): The validation dataset.
            test_series (pd.DataFrame): The test dataset.
            full_raw_data_context (pd.DataFrame): The complete, original raw data
                (including training portion) needed for inverting differencing later.

        Returns:
            A tuple containing the transformed training, validation, and test DataFrames.
        """
        self._validate_dataframe(train_series, "training")
        self._validate_dataframe(val_series, "validation")
        self._validate_dataframe(test_series, "test")

        self._full_raw_data_context = full_raw_data_context.copy()

        datasets = {"train": train_series.copy(), "val": val_series.copy(), "test": test_series.copy()}

        for col_name in train_series.columns:
            for i, (name, df) in enumerate(datasets.items()):
                if col_name in df.columns:
                    should_fit = i == 0  # Fit only on the training set
                    for step_name in self.active_steps:
                        df[col_name] = self._apply_single_transform(
                            df[col_name], step_name, col_name, fit=should_fit
                        )

        # Drop any NaNs that resulted from transformations like differencing.
        for name in datasets:
            datasets[name].dropna(inplace=True)

        return datasets["train"], datasets["val"], datasets["test"]

    def transform(self, series: pd.DataFrame) -> pd.DataFrame:
        """
        Applies pre-fitted transformations to new data.

        Args:
            series (pd.DataFrame): The new data to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.

        Raises:
            RuntimeError: If the preprocessor has not been fitted yet.
        """
        if series.empty:
            return series.copy()
        self._validate_dataframe(series, "transform input")

        if self.preprocessing_steps["scaling"]["enabled"] and not self.scalers:
            raise RuntimeError("Preprocessor has not been fitted yet (scalers are missing).")
        if self.preprocessing_steps["differencing"]["enabled"] and not self.diff_orders:
            raise RuntimeError("Preprocessor has not been fitted yet (differencing orders are missing).")

        # Determine required context size for stateful transforms (differencing).
        max_lag = 0
        if self.preprocessing_steps["differencing"].get("enabled"):
            seasonal_period = self.preprocessing_steps["differencing"].get("seasonal_period", 1)
            d_order = max(self.diff_orders.values()) if self.diff_orders else 1
            s_order = max(self.seasonal_diff_orders.values()) if self.seasonal_diff_orders else 1
            max_lag = (seasonal_period * s_order) + d_order + 5  # small safety buffer

        context = pd.DataFrame()
        if max_lag > 0 and not self._full_raw_data_context.empty:
            context = self._full_raw_data_context.tail(max_lag)

        # Merge context + new series
        series_with_context = pd.concat([context, series]) if not context.empty else series.copy()
        series_with_context = series_with_context[~series_with_context.index.duplicated(keep="last")]

        transformed_series = series_with_context.copy()
        for col_name in series.columns:
            for step_name in self.active_steps:
                transformed_series[col_name] = self._apply_single_transform(
                    transformed_series[col_name], step_name, col_name, fit=False
                )

        # Return only rows of the original input index
        return transformed_series.loc[series.index].dropna()

    def inverse_transforms(
            self,
            predictions: Union[np.ndarray, pd.DataFrame],
            start_after: Optional[Union[pd.Timestamp,int]] = None
    ) -> pd.DataFrame:
        """
        Inverts all transformations for a given set of predictions.

        This method uses a robust strategy to determine the correct time index for the predictions:
        1.  If the `start_after` timestamp is provided, it is used as the single source of truth to
            generate a future index. This is the most reliable method, especially for NumPy array inputs.
        2.  If `start_after` is not provided, the method attempts to use the index from the
            input `predictions` DataFrame, assuming the caller has provided a meaningful index.
        3.  If the input is a NumPy array without a `start_after` timestamp, a ValueError is raised
           to prevent ambiguous guessing.

        Args:
            predictions: The model's output predictions, either as a raw NumPy array or a pandas DataFrame.
            start_after: Optional. The last known timestamp before the forecast begins. If provided,
                         it overrides any index on the input `predictions`, ensuring a correct
                         future forecast index.

        Returns:
            A DataFrame containing the predictions transformed back to their original scale,
            with a correct DatetimeIndex.

        Raises:
            RuntimeError: If the preprocessor has not been fitted with data.
            ValueError: If the time series frequency cannot be inferred, or if a NumPy array is
                        passed without the required `start_after` timestamp.
            TypeError: If the input format is unsupported when `start_after` is not provided.
        """
        if self._full_raw_data_context.empty:
            raise RuntimeError("Preprocessor has not been fitted. Call `apply_transforms` first.")

        context_index = self._full_raw_data_context.index
        n_rows = predictions.shape[0]
        columns = list(self._full_raw_data_context.columns)
        
        predictions_df: pd.DataFrame

        if start_after is not None:
            # Path A: `start_after` is provided. This is the unambiguous path.
            # This is where we handle both index types.
            if isinstance(context_index, pd.DatetimeIndex):
                freq = getattr(self, 'dataset_freq', None) or pd.infer_freq(context_index)
                if freq is None:
                    raise ValueError("Cannot infer frequency for DatetimeIndex.")
                pred_index = pd.date_range(start=start_after + pd.tseries.frequencies.to_offset(freq), periods=n_rows, freq=freq)
            elif isinstance(context_index, pd.RangeIndex):
                if not isinstance(start_after, int):
                    raise TypeError("For RangeIndex context, 'start_after' must be an integer.")
                pred_index = pd.RangeIndex(start=start_after + 1, stop=start_after + 1 + n_rows)
            else:
                raise TypeError(f"Unsupported index type for inverse transform: {type(context_index)}")

            if isinstance(predictions, np.ndarray):
                predictions_df = pd.DataFrame(predictions, index=pred_index, columns=columns)
            else: # is a DataFrame
                predictions_df = predictions.copy()
                predictions_df.index = pred_index
        else:
            # Path B: `start_after` is None. We rely on the input's own index.
            if isinstance(predictions, pd.DataFrame):
                # This works for DataFrames with either DatetimeIndex or RangeIndex.
                predictions_df = predictions.copy()
            elif isinstance(predictions, np.ndarray):
                # ORIGINAL LOGIC RESTORED: This is ambiguous and correctly raises an error.
                raise ValueError(
                    "Input is a NumPy array without timestamps. "
                    "Please provide the 'start_after' timestamp to specify the prediction start point."
                )
            else:
                raise TypeError("Unsupported input. Provide a DataFrame with a valid index or a NumPy array with 'start_after'.")

        # --- Reconstruction process (remains unchanged) ---
        reconstructed_df = predictions_df.copy()
        for col_name in reconstructed_df.columns:
            series_with_index = reconstructed_df[col_name]
            reconstructed_values = self._inverse_transforms_single_column(series_with_index, col_name)
            reconstructed_df[col_name] = reconstructed_values

        return reconstructed_df
