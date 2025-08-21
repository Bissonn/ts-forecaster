"""Module for loading and validating configuration files in the forecasting framework.

This module provides utilities to load YAML configuration files and validate their structure
for datasets, models, and preprocessing steps, ensuring compatibility with models like ARIMA,
VAR, LSTM, and Transformer.
"""

import logging
import os
from typing import Dict, Optional

import pandas as pd
import yaml
from schema import Schema, And, Use, Optional as SchemaOptional, Or, SchemaError
from pandas.tseries.frequencies import to_offset
import textwrap

logger = logging.getLogger(__name__)

# Model names for validation, aligned with model_registry.py
MODEL_NAMES = {
    "arima",
    "sarima",
    "var",
    "lstm_direct",
    "lstm_iterative",
    "transformer",
}

class ConfigValidationError(Exception):
    """Compact, human-readable configuration validation error."""
    pass

def _compact_schema_error(err: SchemaError) -> str:
    """
    Turn a verbose SchemaError into a short, readable message.
    We try err.code first (often the clearest), then fallback to str(err).
    """
    msg = (err.code or str(err) or "").strip()
    # Collapse newlines/indents to one or two lines for CLI readability
    msg = " ".join(msg.split())
    return msg

def is_valid_freq(x: str) -> bool:
    try:
        return to_offset(x) is not None
    except ValueError:
        return False

def _define_preprocessing_schema() -> Schema:
    """
    Define the schema for preprocessing configuration.

    Returns:
        Schema for validating preprocessing settings.
    """
    return Schema({
        SchemaOptional("log_transform"): {
            SchemaOptional("enabled"): bool,
            SchemaOptional("method"): And(str, lambda x: x in ["log", "log1p"]),
            SchemaOptional("epsilon"): And(float, lambda x: x > 0),
        },
        SchemaOptional("winsorize"): {
            SchemaOptional("enabled"): bool,
            SchemaOptional("limits"): And([float], lambda l: len(l) == 2 and 0 <= l[0] < l[1] <= 1),
        },
        SchemaOptional("scaling"): {
            SchemaOptional("enabled"): bool,
            SchemaOptional("method"): And(
                str,
                lambda x: x in ["minmax", "standard", "none"],
                error="`scaling.method` must be one of ['minmax', 'standard', 'none']",
            ),
            SchemaOptional("range"): And(
                [float],
                lambda l: len(l) == 2 and l[0] < l[1],
                error="`scaling.range` must be a list of two values with min < max",
            ),
        },
        SchemaOptional("differencing"): {
            SchemaOptional("enabled"): bool,
            SchemaOptional("auto"): And(str, lambda x: x in ["adf", "kpss", "none"]),
            SchemaOptional("order"): And(int, lambda x: x >= 0),
            SchemaOptional("seasonal_order"): And(int, lambda x: x >= 0),
            SchemaOptional("seasonal_period"): And(int, lambda x: x > 0),
            SchemaOptional("max_d"): And(int, lambda x: x >= 0),
            SchemaOptional("max_D"): And(int, lambda x: x >= 0),
            SchemaOptional("p_value_threshold"): And(float, lambda x: 0.0 < x < 1.0),
        },
    })

def validate_preprocessing(config: Dict, data: Optional[pd.DataFrame] = None) -> None:
    """
    Validate the preprocessing section of the configuration.

    Args:
        config: Preprocessing configuration dictionary.
        data: Optional input DataFrame for validation (e.g., for seasonal_period).

    Raises:
        SchemaError: If the preprocessing configuration is invalid.
        ValueError: If seasonal_period is invalid relative to data length.
    """
    try:
        preprocessing_schema = _define_preprocessing_schema()
        preprocessing_schema.validate(config)

        diff = config.get("differencing", {})
        if data is not None and diff.get("seasonal_order", 0) > 0:
            seasonal_period = diff.get("seasonal_period", 0)
            n_rows = len(data)
            if seasonal_period >= n_rows:
                raise ValueError(
                    f"`differencing.seasonal_period` ({seasonal_period}) must be less than data length ({n_rows})"
                )
        logger.debug("Validated preprocessing configuration: %s", config)
    except (SchemaError, ValueError) as e:
        logger.error("Preprocessing validation failed: %s", str(e))
        raise

def validate_config(config: Dict, data: Optional[pd.DataFrame] = None) -> Dict:
    """
    Validate the configuration for datasets, models, and experiments.

    Args:
        config: Configuration dictionary loaded from YAML.
        data: Optional input DataFrame for preprocessing validation.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If required configuration sections or parameters are missing or invalid.
        SchemaError: If the configuration does not match the schema.
    """
    # Shared schemas
    integer_range = Schema(
        And({
            "min": And(int, lambda x: x >= 0),
            "max": And(int, lambda x: x >= 0),
            SchemaOptional("step"): And(int, lambda x: x > 0),
        },
        lambda d: d["min"] <= d["max"],
        error="`min` must be less than or equal to `max` in integer range")
    )

    float_range = Schema(
        And({
            "min": And(float, lambda x: x >= 0),
            "max": And(float, lambda x: x >= 0),
            SchemaOptional("log"): bool,
            SchemaOptional("grid_steps"): And(int, lambda x: x >= 2),
        },
        lambda d: d["min"] <= d["max"],
        error="`min` must be less than or equal to `max` in float range")
    )

    # ARIMA/SARIMA parameter schema
    arima_param_schema = Schema({
        "window_size": Or([And(int, lambda x: x > 0)], integer_range),
        "p": Or([And(int, lambda x: x >= 0)], integer_range),
        "d": Or([And(int, lambda x: x >= 0)], integer_range),
        "q": Or([And(int, lambda x: x >= 0)], integer_range),
        SchemaOptional("P"): Or([And(int, lambda x: x >= 0)], integer_range),
        SchemaOptional("D"): Or([And(int, lambda x: x >= 0)], integer_range),
        SchemaOptional("Q"): Or([And(int, lambda x: x >= 0)], integer_range),
        SchemaOptional("seasonal_period"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("n_trials"): And(int, lambda x: x > 0),
    })

    var_param_schema = Schema({
        "window_size": Or([And(int, lambda x: x > 0)], integer_range),
        "max_lags": Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("ic"): Or(
            [And(str, lambda x: x in ["aic", "bic", "hqic", "fpe"])],
            {"values": And(list, lambda l: all(x in ["aic", "bic", "hqic", "fpe"] for x in l))},
        ),
        SchemaOptional("n_trials"): And(int, lambda x: x > 0),
    })

    lstm_opt_param_schema = Schema({
        "window_size": Or([And(int, lambda x: x > 0)], integer_range),
        "hidden_size": Or([And(int, lambda x: x > 0)], integer_range),
        "num_layers": Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("dropout"): Or([And(float, lambda x: 0 <= x < 1)], float_range),
        SchemaOptional("batch_size"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("learning_rate"): Or([And(float, lambda x: x > 0)], float_range),
        SchemaOptional("epochs"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("early_stopping_patience"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("weight_decay"): Or([And(float, lambda x: x >= 0)], float_range),
        SchemaOptional("n_trials"): And(int, lambda x: x > 0),
    })

    transformer_opt_param_schema = Schema({
        "window_size": Or([And(int, lambda x: x > 0)], integer_range),
        "hidden_size": Or([And(int, lambda x: x > 0)], integer_range),
        "num_heads": Or([And(int, lambda x: x > 0)], integer_range),
        "num_encoder_layers": Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("num_decoder_layers"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("architecture"): And(str, lambda s: s in ['encoder-only', 'encoder-decoder']),
        SchemaOptional("strategy"): And(str, lambda s: s in ['direct', 'iterative']),
        "dim_ff_multiplier": Or([And(float, lambda x: x > 0)], float_range),
        SchemaOptional("dropout"): Or([And(float, lambda x: 0 <= x < 1)], float_range),
        SchemaOptional("batch_size"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("learning_rate"): Or([And(float, lambda x: x > 0)], float_range),
        SchemaOptional("epochs"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("early_stopping_patience"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("weight_decay"): Or([And(float, lambda x: x >= 0)], float_range),
        SchemaOptional("n_trials"): And(int, lambda x: x > 0),
    })

    auto_arima_schema = Schema({
        SchemaOptional("stepwise"): bool,
        SchemaOptional("suppress_warnings"): bool,
        SchemaOptional("error_action"): And(str, lambda x: x in ["ignore", "warn", "raise"]),
        SchemaOptional("trace"): bool,
        SchemaOptional("maxiter"): And(int, lambda x: x > 0),
        SchemaOptional("n_jobs"): And(int, lambda x: x != 0),
        SchemaOptional("n_fits"): And(int, lambda x: x > 0),
        SchemaOptional("information_criterion"): And(str, lambda x: x in ["aic", "bic", "hqic", "oob"]),
        SchemaOptional("seasonal_test"): And(str, lambda x: x in ["ocsb"]),
        SchemaOptional("enforce_stationarity"): bool,
        SchemaOptional("enforce_invertibility"): bool,
        SchemaOptional("with_intercept"): Or(str, bool, lambda x: x in ["auto", True, False]),
    })

    # Model-specific schemas
    MODEL_SCHEMAS = {
        "arima": Schema({
            "p": And(int, lambda x: x >= 0),
            "d": And(int, lambda x: x >= 0),
            "q": And(int, lambda x: x >= 0),
            "window_size": And(int, lambda x: x > 0),
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda x: x in ["grid", "random", "optuna", "auto_arima"]),
                "params": arima_param_schema,
                SchemaOptional("auto_arima"): auto_arima_schema,
            },
            SchemaOptional("preprocessing"): _define_preprocessing_schema(),
        }),
        "sarima": Schema({
            "p": And(int, lambda x: x >= 0),
            "d": And(int, lambda x: x >= 0),
            "q": And(int, lambda x: x >= 0),
            "P": And(int, lambda x: x >= 0),
            "D": And(int, lambda x: x >= 0),
            "Q": And(int, lambda x: x >= 0),
            "seasonal_period": And(int, lambda x: x > 0),
            "window_size": And(int, lambda x: x > 0),
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda x: x in ["grid", "random", "optuna", "auto_arima"]),
                "params": arima_param_schema,
                SchemaOptional("auto_arima"): auto_arima_schema,
            },
            SchemaOptional("preprocessing"): _define_preprocessing_schema(),
        }),
        "var": Schema({
            "window_size": And(int, lambda x: x > 0),
            "max_lags": And(int, lambda x: x > 0),
            SchemaOptional("ic"): And(str, lambda x: x in ["aic", "bic", "hqic", "fpe"]),
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda x: x in ["grid", "random", "optuna"]),
                "params": var_param_schema,
            },
            SchemaOptional("preprocessing"): _define_preprocessing_schema(),
        }),
        "lstm_direct": Schema({
            "window_size": And(int, lambda x: x > 0),
            "hidden_size": And(int, lambda x: x > 0),
            "num_layers": And(int, lambda x: x > 0),
            SchemaOptional("dropout"): And(float, lambda x: 0 <= x < 1),
            SchemaOptional("batch_size"): And(int, lambda x: x > 0),
            SchemaOptional("learning_rate"): And(float, lambda x: x > 0),
            SchemaOptional("epochs"): And(int, lambda x: x > 0),
            SchemaOptional("early_stopping_patience"): And(int, lambda x: x > 0),
            SchemaOptional("weight_decay"): And(float, lambda x: x >= 0),
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda x: x in ["grid", "random", "optuna"]),
                "params": lstm_opt_param_schema,
            },
            SchemaOptional("preprocessing"): _define_preprocessing_schema(),
        }),
        "lstm_iterative": Schema({
            "window_size": And(int, lambda x: x > 0),
            "hidden_size": And(int, lambda x: x > 0),
            "num_layers": And(int, lambda x: x > 0),
            SchemaOptional("dropout"): And(float, lambda x: 0 <= x < 1),
            SchemaOptional("batch_size"): And(int, lambda x: x > 0),
            SchemaOptional("learning_rate"): And(float, lambda x: x > 0),
            SchemaOptional("epochs"): And(int, lambda x: x > 0),
            SchemaOptional("early_stopping_patience"): And(int, lambda x: x > 0),
            SchemaOptional("weight_decay"): And(float, lambda x: x >= 0),
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda x: x in ["grid", "random", "optuna"]),
                "params": lstm_opt_param_schema,
            },
            SchemaOptional("preprocessing"): _define_preprocessing_schema(),
        }),
        "transformer": Schema({
            "window_size": And(int, lambda x: x > 0),
            "hidden_size": And(int, lambda x: x > 0),
            "num_heads": And(int, lambda x: x > 0),
            "num_encoder_layers": And(int, lambda x: x > 0),
            SchemaOptional("num_decoder_layers"): Or([And(int, lambda x: x > 0)], integer_range),
            SchemaOptional("architecture"): And(str, lambda s: s in ['encoder-only', 'encoder-decoder']),
            SchemaOptional("strategy"): And(str, lambda s: s in ['direct', 'iterative']),
            "dim_ff_multiplier": And(float, lambda x: x > 0),
            SchemaOptional("dropout"): And(float, lambda x: 0 <= x < 1),
            SchemaOptional("batch_size"): And(int, lambda x: x > 0),
            SchemaOptional("learning_rate"): And(float, lambda x: x > 0),
            SchemaOptional("epochs"): And(int, lambda x: x > 0),
            SchemaOptional("early_stopping_patience"): And(int, lambda x: x > 0),
            SchemaOptional("weight_decay"): And(float, lambda x: x >= 0),
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda x: x in ["grid", "random", "optuna"]),
                "params": transformer_opt_param_schema,
            },
            SchemaOptional("preprocessing"): _define_preprocessing_schema(),
        }),
    }

    experiment_schema = Schema({
        "name": And(str, len),
        "description": And(str, len),
        SchemaOptional("dataset"): str,
        SchemaOptional("models"): [And(str, lambda x: x in MODEL_NAMES)],
        "validation_setup": {
            "forecast_steps": And(int, lambda n: n > 0),
            "n_folds": And(int, lambda n: n > 0),
            "max_window_size": And(int, lambda n: n > 0),
            SchemaOptional("early_stopping_validation_percentage"): And(
                Or(int, float), lambda x: 0 < x <= 100
            ),
        },
    })

    schema = Schema({
        SchemaOptional("experiments"): [experiment_schema],
        SchemaOptional("training"): {
            SchemaOptional("horizons"): [And(int, lambda x: x > 0)],
            SchemaOptional("cv_folds"): And(int, lambda x: x > 0),
        },
        "datasets": {
            str: {
                "path": And(str, lambda x: os.path.exists(x), error="Dataset file path does not exist"),
                "columns": [And(str, len)],
                SchemaOptional("freq"): And(str, is_valid_freq, error="Invalid frequency string"),
                SchemaOptional("preprocessing"): _define_preprocessing_schema(),
            },
        },
        "models": {
            SchemaOptional("arima"): MODEL_SCHEMAS["arima"],
            SchemaOptional("sarima"): MODEL_SCHEMAS["sarima"],
            SchemaOptional("var"): MODEL_SCHEMAS["var"],
            SchemaOptional("lstm_direct"): MODEL_SCHEMAS["lstm_direct"],
            SchemaOptional("lstm_iterative"): MODEL_SCHEMAS["lstm_iterative"],
            SchemaOptional("transformer"): MODEL_SCHEMAS["transformer"],
        },
    })

    try:
        validated_config = schema.validate(config)
        for dataset_name, dataset_config in config.get("datasets", {}).items():
            if "preprocessing" in dataset_config:
                validate_preprocessing(dataset_config["preprocessing"], data)
        for model_name, model_config in config.get("models", {}).items():
            if model_name not in MODEL_NAMES:
                raise ValueError(f"Invalid model name: {model_name}. Must be one of {MODEL_NAMES}")
            if "preprocessing" in model_config:
                validate_preprocessing(model_config["preprocessing"], data)
            opt_params = model_config.get("optimization", {}).get("params", {})
            all_params = set(model_config.keys()) | set(opt_params.keys())
            model_required_params = {
                "arima": {"window_size", "p", "d", "q"},
                "sarima": {"window_size", "p", "d", "q", "P", "D", "Q", "seasonal_period"},
                "var": {"window_size", "max_lags"},
                "lstm_direct": {"window_size", "hidden_size", "num_layers"},
                "lstm_iterative": {"window_size", "hidden_size", "num_layers"},
                "transformer": {"window_size", "hidden_size", "num_heads"},
            }
            required_params = model_required_params.get(model_name, set())
            missing_params = required_params - all_params
            if missing_params:
                raise SchemaError(
                    f"Missing required parameter(s) {missing_params} for {model_name}. "
                    f"Specify in models.{model_name} or optimization.params."
                )
        logger.info("Configuration validation passed successfully")
        return validated_config
    except (SchemaError, ValueError) as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise

def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load and validate a configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file. Defaults to 'config.yaml'.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
        SchemaError: If the configuration does not match the schema.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        if not config:
            logger.error("Configuration file is empty")
            raise ValueError("Configuration file is empty")
        return validate_config(config)
    except (yaml.YAMLError, SchemaError) as e:
#        logger.error(f"Failed to parse YAML file {config_path}: {str(e)}")
        # Wrap with a short message (no stack trace) for the caller.
        raise ConfigValidationError(_compact_schema_error(e))
#        raise
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
        raise

def get_model_config(model_name: str, config_path: str = "config.yaml") -> Dict:
    """
    Load configuration for a specific model from a YAML file.

    Args:
        model_name: Name of the model (e.g., 'arima', 'transformer').
        config_path: Path to the YAML configuration file. Defaults to 'config.yaml'.

    Returns:
        Model configuration dictionary, with default optimization settings if not specified.

    Raises:
        ValueError: If model_name is invalid or no configuration is found.
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Invalid model name: {model_name}. Must be one of {MODEL_NAMES}")
    config = load_config(config_path)
    model_config = config.get("models", {}).get(model_name, {})
    if not model_config:
        logger.warning(f"No configuration found for model {model_name}. Using empty config.")
    model_config.setdefault("optimization", {"method": "grid", "params": {}})
    return model_config
