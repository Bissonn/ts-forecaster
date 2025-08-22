# TS-Forecaster: Advanced Time Series Forecasting Framework

## 🤖 Project Overview

This project provides a flexible and extensible Python-based framework for time series forecasting. It is designed to facilitate the training, evaluation, and comparison of a wide range of models—from classical statistical methods like **ARIMA** and **VAR** to advanced deep learning architectures, including **LSTM** and **Transformer** networks.

Key features include:
- **Unified Model Interface**: A model registry system allows for the easy integration of new architectures without altering the core training logic.
- **Automated Preprocessing Pipeline**: A powerful, configurable preprocessing engine handles data transformations such as scaling, differencing, log transforms, and outlier capping.
- **Hyperparameter Optimization**: Built-in support for `Grid Search`, `Random Search`, and `Optuna` enables automated model tuning to find the best-performing parameters.
- **YAML-Based Configuration**: The entire workflow—from dataset definitions to model parameters and experiment setups—is managed through a single, human-readable `config.yaml` file.
- **Comprehensive Evaluation & Visualization**: The framework automatically calculates key performance metrics (MAE, RMSE, SMAPE, MASE) and generates plots comparing forecasts against actual values.

---

## 📁 Directory Structure

The project is organized into a modular structure to promote clarity and ease of extension.

mag/
├── config.yaml                # Main configuration file for all experiments.
├── data/                      # Directory for storing raw dataset files (e.g., .csv).
├── models/                    # Contains all forecasting model implementations.
├── results/                   # Default output directory for all generated artifacts.
├── scripts/                   # Main executable scripts for running the pipeline.
├── tests/                     # Unit and integration tests mirroring the project structure.
└── utils/                     # Core utilities and helper modules.


- **`models/`**: This is where the logic for each forecasting model resides. The `base.py` file defines the abstract classes that all models must inherit from, ensuring a consistent API. The `model_registry.py` handles the registration of new models so they are accessible to the factory.
- **`results/`**: All outputs from the training and evaluation process are saved here. This includes logs, performance metrics, visualizations, and serialized trained models.
- **`scripts/`**: Contains the main entry point for the application, `train.py`, which orchestrates the entire process from configuration loading to model training and prediction.
- **`tests/`**: Holds all tests for the project. The structure of this directory mirrors the main project structure to make locating tests for specific modules intuitive.
- **`utils/`**: A collection of helper modules responsible for tasks like loading configurations (`config_utils.py`), managing datasets (`dataset.py`), preprocessing data (`preprocessor.py`), calculating metrics (`metrics.py`), and running hyperparameter searches (`hyperopt/`).

---

## 🚀 How to Use

The primary entry point for all operations is the `scripts/train.py` script.

**Step 1: Configure Your Environment**

First, adjust the `config.yaml` file to define your datasets, models, and experiment parameters. See the detailed configuration section below for all available options.

**Step 2: Run the Training Script**

Execute the `train.py` script from your terminal using the following commands.

- **Train a single model on a specific dataset:**
  ```bash
  python -m scripts.train --model arima --dataset ETTh1
Train multiple models simultaneously:

Bash

python -m scripts.train --models arima lstm_direct --dataset ETTh1
Run training with hyperparameter optimization enabled:
(Ensure the optimize: true flag and optimization block are set in config.yaml for the model)

Bash

python -m scripts.train --model lstm_direct --dataset gemini_data --optimized
Force retraining, ignoring any saved model files:

Bash

python -m scripts.train --model var --dataset ETTh1 --force-train
All results, including metrics, plots, and trained models, will be saved to the results/ directory.

🛠️ Configuration (config.yaml)
The config.yaml file is the central control panel for the entire framework. It is validated against a schema defined in utils/config_utils.py.

Main Sections
experiments
This section defines the overall validation and evaluation strategy.

name (string): A descriptive name for the experiment.

description (string): A longer description of the experiment's goal.

validation_setup:

forecast_steps (integer): The number of future time steps to predict (the forecast horizon).

n_folds (integer): The number of folds to use in walk-forward cross-validation for hyperparameter tuning.

max_window_size (integer): The size of the initial training window in the first fold of cross-validation.

early_stopping_validation_percentage (float): The percentage of data from each fold to use as a validation set for early stopping in neural network training.

datasets
This section defines all available datasets.

<dataset_name>: A unique identifier for the dataset.

path (string): The file path to the CSV dataset. The path is validated for existence.

columns (list of strings): The specific columns from the CSV file to be used as features.

freq (string, optional): The frequency of the time series data (e.g., 'H' for hourly, 'D' for daily). If not provided, the framework will attempt to infer it. See Pandas Offset Aliases for all options.

preprocessing (dict, optional): Dataset-specific preprocessing steps that will apply to all models trained on this data. The structure is the same as the model-level preprocessing block described below.

models
This is where you configure the parameters for each forecasting model.

<model_name>: The name must match a registered model (e.g., arima, lstm_direct).

Base Parameters: Each model has its own set of required and optional parameters (e.g., p, d, q for ARIMA; window_size, hidden_size for LSTM). These are used when optimization is disabled.

optimize (boolean): Set to true to enable hyperparameter optimization for this model.

optimization block (dict, optional):

method (string): The optimization strategy. Must be one of grid, random, or optuna.

params (dict): The hyperparameter search space.

For discrete values, provide a list: p: [1, 2, 3].

For integer ranges, provide a dictionary: hidden_size: {"min": 32, "max": 128, "step": 32}.

For float ranges, provide a dictionary: learning_rate: {"min": 0.0001, "max": 0.01, "log": true}. log: true enables logarithmic sampling.

preprocessing block (dict, optional): Model-specific data transformations.

log_transform (enabled: bool, method: 'log' or 'log1p').

winsorize (enabled: bool, limits: [lower_quantile, upper_quantile]).

scaling (enabled: bool, method: 'minmax' or 'standard').

differencing (enabled: bool, auto: 'adf', 'kpss', or 'none', order: int, seasonal_order: int, seasonal_period: int).

Configuration Examples
Example 1: Simple ARIMA with Grid Search Optimization

YAML

models:
  arima:
    p: 5
    d: 1
    q: 0
    window_size: 30
    optimize: true
    optimization:
      method: grid
      params:
        p: [1, 2, 3]
        d: [1]
        q: [0, 1]
    preprocessing:
      scaling:
        enabled: true
        method: 'standard'
Example 2: LSTM with Optuna Optimization and Complex Preprocessing

YAML

models:
  lstm_direct:
    window_size: 90
    hidden_size: 128
    num_layers: 2
    optimize: true
    optimization:
      method: optuna
      params:
        window_size: [30, 60, 90] # Optuna treats lists as categorical choices
        hidden_size:
          min: 64
          max: 256
          step: 32
        learning_rate:
          min: 0.0001
          max: 0.01
          log: true
        n_trials: 20 # n_trials is part of the params block
    preprocessing:
      scaling:
        enabled: true
        method: 'minmax'
      differencing:
        enabled: true
        auto: 'adf'
        max_d: 2
✨ Adding New Models
The framework is designed for easy extension. To add a custom forecasting model, follow these steps:

Create the Model File: In the mag/models/ directory, create a new Python file (e.g., my_new_model.py).

Implement the Forecaster Class:

Your new class must inherit from either StatTSForecaster (for statistical models) or NeuralTSForecaster (for deep learning models), which are defined in models/base.py.

Implement all the abstract methods required by the base class, such as fit and predict.

Call the parent constructor (super().__init__(...)) within your model's __init__ method.

Register Your Model:

Decorate your class with @register_model from models.model_registry.

Provide a unique string name (this name will be used in config.yaml) and set the is_univariate flag if your model can only handle a single time series at a time.

Python

# In mag/models/my_new_model.py
from models.base import StatTSForecaster
from models.model_registry import register_model

@register_model("my_model", is_univariate=False)
class MyNewModelForecaster(StatTSForecaster):
    def __init__(self, model_params, num_features, forecast_steps):
        super().__init__(model_params, num_features, forecast_steps)
        # ... custom initialization ...

    def fit(self, train_series, val_series=None):
        # ... model fitting logic ...
        self.fitted = True

    def predict(self, input_data=None, forecast_steps=None):
        # ... prediction logic ...
        # return pd.DataFrame(...)
Make the Model Discoverable: Add an import statement for your new module in mag/scripts/train.py to ensure the registration code runs when the application starts.

Python

# In mag/scripts/train.py
import models.my_new_model
Add Configuration: Finally, add a new section for my_model in your config.yaml file under the models key to define its parameters.

Your new model is now fully integrated into the framework and can be used just like any of the built-in models.

🧪 Testing Strategy
The project maintains a comprehensive test suite to ensure code quality and reliability. The testing philosophy is based on a parallel directory structure.

Parallel Structure: For every file in the main project (e.g., utils/dataset.py), there is a corresponding test file in the tests/ directory (e.g., tests/utils/test_dataset.py).

Naming Convention: Test files are always prefixed with test_. For example, the tests for utils/preprocessor.py are located in tests/utils/test_preprocessor.py.

Test Types:

Unit Tests: The majority of tests are unit tests, which verify the functionality of individual functions and classes in isolation. Dependencies are heavily mocked using pytest and unittest.mock to ensure tests are fast and deterministic.

Integration Tests: A smaller set of integration tests (e.g., tests/test_integration.py) verifies that different components of the system work together correctly in a full fit -> predict pipeline.