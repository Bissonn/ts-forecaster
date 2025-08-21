"""
Module for training and evaluating time series forecasting models.

This module provides functions to train, save, load, and evaluate time series forecasting
models using a ModelFactory pattern. It supports both statistical (e.g., ARIMA, SARIMA)
and neural (e.g., LSTM, Transformer) models, with hyperparameter optimization, dataset
preparation, and visualization of predictions.
"""

import os
import torch
import argparse
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from utils.dataset import TimeSeriesDataset
from utils.config_utils import load_config, get_model_config
from utils.visualizer import Visualizer
from utils.metrics import calculate_metrics
from models.factory import ModelFactory
from models.base import TSForecaster, NeuralTSForecaster
from utils.preprocessor import Preprocessor
from models.model_registry import model_registry
from utils.data_utils import create_sliding_window
import pickle

# Explicitly import model files to ensure registration
import models.arima
import models.sarima
import models.transformer
import models.lstm
import models.var

def setup_logging(log_dir: str = "results/logs") -> None:
    """
    Configure logging to file and console.

    Args:
        log_dir (str): Directory to store log files. Defaults to 'results/logs'.
    """
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

def normalize_model_name(name: str) -> str:
    """
    Normalize a model name to the standard snake_case format expected by ModelFactory.

    Args:
        name (str): Input model name (e.g., 'LSTMDirect', 'lstmdirect', 'lstm_direct').

    Returns:
        str: Normalized model name in snake_case (e.g., 'lstm_direct').

    Raises:
        ValueError: If the model name is not recognized.
    """
    model_name_map = {
        'arima': 'arima',
        'sarima': 'sarima',
        'transformer': 'transformer',
        'lstm_direct': 'lstm_direct',
        'lstmdirect': 'lstm_direct',
        'lstm_iterative': 'lstm_iterative',
        'lstmiterative': 'lstm_iterative',
        'var': 'var'
    }
    normalized_input = name.lower().replace(' ', '_').replace('-', '_')
    if normalized_input in model_name_map:
        return model_name_map[normalized_input]
    raise ValueError(f"Unrecognized model name: {name}. Supported models: {sorted(set(model_name_map.values()))}")

def save_model(model: TSForecaster, path: str) -> None:
    """
    Save a trained model and its preprocessor state to a file.

    Args:
        model (TSForecaster): The trained forecasting model.
        path (str): File path to save the model.

    Raises:
        OSError: If saving the model fails.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        preprocessor_state = {
            'config': model.preprocessor.config,
            'scalers': model.preprocessor.scalers,
            'diff_orders': model.preprocessor.diff_orders,
            'seasonal_diff_orders': model.preprocessor.seasonal_diff_orders,
            '_full_raw_data_context': model.preprocessor._full_raw_data_context
        }
        
        state = {
            'state_dict': getattr(model.model, 'state_dict', lambda: None)(),
            'model_config': model.model_params,
            'model_class': type(model).__name__,
            'fitted': model.fitted,
            'preprocessor_state': preprocessor_state,
            'last_series': getattr(model, 'last_series', None),
            'statsmodel': None
        }
        
        if model.__class__.__name__ in ['ARIMAForecaster', 'SARIMAForecaster', 'VARForecaster']:
            state['statsmodel'] = model.model
            
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {str(e)}", exc_info=True)
        raise

def load_model(path: str, model_name: str, model_params: Dict, num_features: int, forecast_steps: int,
               config_path: str) -> TSForecaster:
    """
    Load a saved model and its preprocessor state from a file.

    Args:
        path (str): File path to the saved model.
        model_name (str): Name of the model to load.
        model_params (Dict): Model configuration parameters.
        num_features (int): Number of features in the dataset.
        forecast_steps (int): Forecast horizon.
        config_path (str): Path to the configuration file.

    Returns:
        TSForecaster: The loaded forecasting model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model name is not supported.
        RuntimeError: If loading fails.
    """
    model_name_normalized = normalize_model_name(model_name)
    supported_models = ['arima', 'sarima', 'transformer', 'lstm_direct', 'lstm_iterative', 'var']
    if model_name_normalized not in supported_models:
        logger.error(f"Model {model_name_normalized} is not supported.")
        raise ValueError(f"Model {model_name_normalized} is not supported.")

    try:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        forecaster = ModelFactory.create(model_name_normalized, state['model_config'], num_features, forecast_steps)

        if state['statsmodel'] is not None:
            forecaster.model = state['statsmodel']
        elif state['state_dict'] is not None:
            forecaster.model.load_state_dict(state['state_dict'])
            forecaster.model.to(device)
        forecaster.fitted = state['fitted']

        # Restore preprocessor state
        if 'preprocessor_state' in state:
            p_state = state['preprocessor_state']
            forecaster.preprocessor.config = p_state.get('config', {})
            forecaster.preprocessor.scalers = p_state.get('scalers', {})
            forecaster.preprocessor.diff_orders = p_state.get('diff_orders', {})
            forecaster.preprocessor.seasonal_diff_orders = p_state.get('seasonal_diff_orders', {})
            forecaster.preprocessor._full_raw_data_context = p_state.get('_full_raw_data_context', pd.Series(dtype=float))
        else:
            logger.warning(f"No preprocessor state found in saved model {path}.")

        # Restore model-specific attributes
        forecaster.last_series = state.get('last_series', None)
        
        logger.info(f"Loaded model from {path} with fitted={forecaster.fitted}")
        return forecaster
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {str(e)}", exc_info=True)
        raise

def new_model(
        model_name: str,
        dataset_identifier: str,
        dataset: TimeSeriesDataset,
        config: Dict,
        forecast_steps: int,
        model_path: str,
        config_path: str,
        validation_params: Dict,
        optimized: bool = False
) -> TSForecaster:
    """
    Create and train a new forecasting model.

    Args:
        model_name (str): Name of the model to train.
        dataset_identifier (str): Identifier for the dataset (e.g., column name or 'multi').
        dataset (TimeSeriesDataset): Dataset containing time series data.
        config (Dict): Training configuration.
        forecast_steps (int): Forecast horizon.
        model_path (str): Path to save the trained model.
        config_path (str): Path to the configuration file.
        validation_params (Dict): Parameters for validation (e.g., cross-validation settings).
        optimized (bool): Whether to perform hyperparameter optimization. Defaults to False.

    Returns:
        TSForecaster: The trained forecasting model.

    Raises:
        ValueError: If forecast_steps is not positive or model_name is invalid.
        RuntimeError: If training fails.
    """
    if forecast_steps <= 0:
        raise ValueError("forecast_steps must be a positive integer")

    try:
        logger.info(f"Training new {model_name} for {dataset_identifier} with forecast_steps={forecast_steps}")
        model_params = get_model_config(model_name, config_path)
        
        best_params = model_params
        if optimized:
            forecaster_for_optim = ModelFactory.create(model_name, model_params, dataset.series.shape[1], forecast_steps)
            logger.info(f"Optimizing hyperparameters for {model_name}...")
            best_params, best_score = forecaster_for_optim.optimize_hyperparameters(
                dataset, model_params, validation_params
            )
            logger.info(f"Finished optimization for {model_name}. Best CV score: {best_score:.6f}")
            logger.info(f"Best params found: {best_params}")
        else:
            logger.info(f"Skipping hyperparameter optimization for {model_name}.")

        # Create final model instance with best parameters
        final_forecaster = ModelFactory.create(
            model_name, best_params, dataset.series.shape[1], forecast_steps
        )
        
        # Final training on the entire development set
        logger.info("Fitting final model on the entire development set...")
        final_forecaster._fit_and_evaluate_fold(
            train_fold=dataset.development_data,
            validation_params=validation_params,
            is_final_fit=True
        )
    
        save_model(final_forecaster, model_path)
        return final_forecaster

    except Exception as e:
        logger.error(f"Failed to train {model_name}: {str(e)}", exc_info=True)
        raise

def create_or_load_model(
        model_name: str,
        dataset: TimeSeriesDataset,
        config: Dict,
        forecast_steps: int,
        config_path: str,
        validation_params: Dict,
        optimized: bool,
        force_train: bool
) -> TSForecaster:
    """
    Create a new model or load an existing one from disk.

    Args:
        model_name (str): Name of the model.
        dataset (TimeSeriesDataset): Dataset containing time series data.
        config (Dict): Training configuration.
        forecast_steps (int): Forecast horizon.
        config_path (str): Path to the configuration file.
        validation_params (Dict): Parameters for validation.
        optimized (bool): Whether to optimize hyperparameters.
        force_train (bool): Whether to force training even if a saved model exists.

    Returns:
        TSForecaster: The trained or loaded forecasting model.

    Raises:
        ValueError: If forecast_steps is not positive or model_name is invalid.
        FileNotFoundError: If config_path or model file is invalid.
    """
    if forecast_steps <= 0:
        raise ValueError("forecast_steps must be a positive integer")

    dataset_identifier = dataset.series.columns[0] if dataset.series.shape[1] == 1 else 'multi'
    model_path = f"{config.get('results_dir', 'results/trained_models')}/{model_name}_{dataset_identifier}_horizon_{forecast_steps}.pkl"
    
    if os.path.exists(model_path) and not force_train:
        logger.info(f"Loading existing model from {model_path}")
        model_params = get_model_config(model_name, config_path)
        return load_model(model_path, model_name, model_params, dataset.series.shape[1], forecast_steps, config_path)
    else:
        return new_model(
            model_name, dataset_identifier, dataset, config, forecast_steps,
            model_path, config_path, validation_params, optimized
        )

def train_model(
        model_name: str,
        dataset: TimeSeriesDataset,
        config: Dict,
        forecast_steps: int,
        config_path: str,
        validation_params: Dict,
        optimized: bool = False,
        force_train: bool = False
) -> np.ndarray:
    """
    Train a model and return its predictions.

    Args:
        model_name (str): Name of the model to train.
        dataset (TimeSeriesDataset): Dataset containing time series data.
        config (Dict): Training configuration.
        forecast_steps (int): Forecast horizon.
        config_path (str): Path to the configuration file.
        validation_params (Dict): Parameters for validation.
        optimized (bool): Whether to optimize hyperparameters. Defaults to False.
        force_train (bool): Whether to force training even if a saved model exists. Defaults to False.

    Returns:
        np.ndarray: Model predictions for the test set.

    Raises:
        ValueError: If inputs are invalid (e.g., forecast_steps <= 0).
        RuntimeError: If training or prediction fails.
    """
    if forecast_steps <= 0:
        raise ValueError("forecast_steps must be a positive integer")

    model_name_normalized = normalize_model_name(model_name)
    model_params = get_model_config(model_name_normalized, config_path)

    # Decide how many features to pass to the temporary instance used only for prepare_data().
    # If the model is univariate, we must pass num_features=1 here; otherwise, constructors like ARIMA/SARIMA will raise.
    model_cls = model_registry.get(model_name_normalized)
    if model_cls is None:
        raise ValueError(f"Model '{model_name_normalized}' is not registered.")
    
    is_univariate_model = getattr(model_cls, "is_univariate", False)
    stub_num_features = 1 if is_univariate_model else dataset.series.shape[1]
    
    # Create a temporary instance to call prepare_data (it may split multivariate data into per-column datasets).
    forecaster_instance = ModelFactory.create(
        model_name_normalized, model_params, stub_num_features, forecast_steps
    )
    
    datasets = forecaster_instance.prepare_data(dataset)
    logger.info(f"Prepared {len(datasets)} local dataset(s) for training.")

    all_predictions = []
    for local_dataset in datasets:
        forecaster = create_or_load_model(
            model_name_normalized, local_dataset, config, forecast_steps, config_path,
            validation_params, optimized, force_train
        )
        
        # Get window size for prediction input
        window_size = forecaster.model_params.get('window_size')
        if window_size is None:
            raise ValueError(f"window_size not defined for model {model_name_normalized}")
            
        input_data_raw = local_dataset.development_data.iloc[-window_size:] if window_size else pd.DataFrame()
        
        # Generate predictions
        predictions = forecaster.predict(input_data_raw)
        
        # Save results using dataset's method
        local_dataset._save_results(predictions.values, model_name_normalized, forecast_steps, calculate_metrics)
        all_predictions.append(predictions.values)

    # Concatenate predictions for multivariate series
    if len(all_predictions) > 1:
        if not all(pred.shape[1] == all_predictions[0].shape[1] for pred in all_predictions):
            raise ValueError("Inconsistent prediction shapes for multivariate series")
        return np.concatenate(all_predictions, axis=1)
    return all_predictions[0]

def initialize_environment(config: Dict = None) -> None:
    """
    Initialize the environment by setting up logging and random seeds.

    Args:
        config (Dict, optional): Configuration dictionary with optional 'results_dir' and 'log_dir' keys.
    """
    log_dir = config.get('log_dir', 'results/logs') if config else 'results/logs'
    setup_logging(log_dir)
    np.random.seed(42)
    torch.manual_seed(42)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for training.

    Returns:
        argparse.Namespace: Parsed arguments.

    Raises:
        ValueError: If required arguments are invalid.
    """
    parser = argparse.ArgumentParser(description="Train time series forecasting models")
    parser.add_argument('--experiment', type=str, default=None, help="Name of the experiment to run from config.yaml. If None, runs all experiments.")
    parser.add_argument('--models', nargs='+', default=['lstm_direct'], help="List of model names to train")
    parser.add_argument('--dataset', default='gemini_data', help="Name of the dataset")
    parser.add_argument('--columns', nargs='+', default=None, help="List of columns to use from the dataset")
    parser.add_argument('--optimized', action='store_true', help="Enable hyperparameter optimization")
    parser.add_argument('--force-train', action='store_true', help="Force training even if a saved model exists")
    parser.add_argument('--config-path', default='config.yaml', help="Path to the configuration file")
    args = parser.parse_args()

    if not args.models:
        raise ValueError("At least one model must be specified")
    if not args.dataset:
        raise ValueError("Dataset name must be specified")
    return args

def load_and_prepare_dataset(args: argparse.Namespace, config_path: str) -> Tuple[Dict, TimeSeriesDataset]:
    """
    Load and prepare the dataset based on command-line arguments and configuration.

    Args:
        args (argparse.Namespace): Command-line arguments.
        config_path (str): Path to the configuration file.

    Returns:
        Tuple[Dict, TimeSeriesDataset]: Configuration dictionary and prepared dataset.

    Raises:
        FileNotFoundError: If config_path or dataset is invalid.
    """
    config = load_config(config_path)
    dataset_name = args.dataset
    columns = args.columns if args.columns is not None else config['datasets'][dataset_name].get('columns')
    dataset = TimeSeriesDataset(dataset_name, config, columns=columns)
    return config, dataset

def train_and_visualize(
        model_name: str,
        dataset: TimeSeriesDataset,
        config: Dict,
        forecast_steps: int,
        config_path: str,
        optimized: bool,
        force_train: bool,
        validation_params: Dict
) -> None:
    """
    Train a model, generate predictions, and visualize results.

    Args:
        model_name (str): Name of the model to train.
        dataset (TimeSeriesDataset): Dataset containing time series data.
        config (Dict): Training configuration.
        forecast_steps (int): Forecast horizon.
        config_path (str): Path to the configuration file.
        optimized (bool): Whether to optimize hyperparameters.
        force_train (bool): Whether to force training.
        validation_params (Dict): Parameters for validation.

    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If training or visualization fails.
    """
    if forecast_steps <= 0:
        raise ValueError("forecast_steps must be a positive integer")

    logger.info(f"Processing model: {model_name} on dataset: {dataset.name} with columns: {dataset.columns}")
    predictions = train_model(model_name, dataset, config, forecast_steps, config_path, validation_params, optimized, force_train)
    if predictions is None:
        logger.warning(f"No predictions generated for model {model_name}")
        return
    model_label = f"{model_name}_horizon_{forecast_steps}"
    Visualizer.plot_predictions(dataset.name, model_label, dataset.test_data, predictions, dataset.series.columns, forecast_steps)
    Visualizer.plot_error_accumulation(dataset.name, model_label, dataset.test_data, predictions, dataset.series.columns, forecast_steps)

def main() -> None:
    """
    Main function to run the training pipeline.

    Parses arguments, loads configuration, prepares dataset, and runs experiments.
    """
    args = parse_arguments()
    config = load_config(args.config_path)
    initialize_environment(config)

    all_experiments = config.get('experiments', [])
    if not all_experiments:
        logger.info("No 'experiments' section found. Running in legacy mode with horizons.")
        models = [normalize_model_name(model) for model in args.models]
        horizons = config.get('training', {}).get('horizons', [10])
        _, dataset = load_and_prepare_dataset(args, args.config_path)
        for forecast_steps in horizons:
            dataset.split_data(forecast_steps=forecast_steps)
            for model_name in models:
                train_and_visualize(model_name, dataset, config, forecast_steps, args.config_path, args.optimized, args.force_train, validation_params={})
        return

    experiments_to_run = all_experiments
    if args.experiment:
        experiments_to_run = [exp for exp in all_experiments if exp.get('name') == args.experiment]
        if not experiments_to_run:
            logger.error(f"Experiment '{args.experiment}' not found.")
            return

    config, dataset = load_and_prepare_dataset(args, args.config_path)

    for experiment_config in experiments_to_run:
        exp_name = experiment_config.get('name', 'unnamed_experiment')
        validation_setup = experiment_config['validation_setup']
        forecast_steps = validation_setup['forecast_steps']
        
        logger.info(f"================== Starting Experiment: {exp_name} ==================")
        
        dataset.split_data(forecast_steps=forecast_steps)

        models_to_run = experiment_config.get('models', [normalize_model_name(m) for m in args.models])
        
        for model_name in models_to_run:
            if model_name in config.get('models', {}):
                try:
                    train_and_visualize(
                        model_name, dataset, config, forecast_steps, args.config_path, 
                        args.optimized, args.force_train, validation_setup
                    )
                except Exception as e:
                    logger.error(f"Failed to process model {model_name} for experiment '{exp_name}': {e}", exc_info=True)
                    continue
            else:
                logger.warning(f"Model '{model_name}' not found in config.yaml. Skipping.")

if __name__ == "__main__":
    main()
