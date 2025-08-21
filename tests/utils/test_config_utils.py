import pytest
from schema import SchemaError
from utils.config_utils import validate_config, get_model_config
import os
import yaml

@pytest.fixture
def valid_config_dict(dummy_data_path):
    """Pytest fixture to provide a base valid configuration using a temporary data file."""
    # This fixture uses the path to the dynamically created dummy file
    return {
        'experiments': [{
            'name': 'test_experiment',
            'description': 'This is a test experiment.',
            'dataset': 'my_data',
            'models': ['arima'],
            'validation_setup': {
                'forecast_steps': 10,
                'n_folds': 3,
                'max_window_size': 50
            }
        }],
        'datasets': {
            'my_data': {
                'path': dummy_data_path, # Use the path from the fixture
                'columns': ['value'],
                'freq': 'D'
            }
        },
        'models': {
            'arima': {
                'p': 1, 'd': 1, 'q': 1, 'window_size': 12,
                'preprocessing': {'scaling': {'enabled': True}}
            }
        }
    }

@pytest.fixture
def dummy_data_path(tmp_path):
    """Creates a dummy data file for config validation and returns its path."""
    data_file = tmp_path / "data.csv"
    data_file.write_text("date,value\n2023-01-01,10")
    # Return the path as a string, as it would appear in the YAML config
    return str(data_file)

def test_config_validation_succeeds_with_valid_config(valid_config_dict):
    """Test that a correctly structured config passes validation."""
    try:
        validated = validate_config(valid_config_dict)
        assert isinstance(validated, dict)
    except Exception as e:
        pytest.fail(f"validate_config() raised an unexpected exception: {e}")

def test_config_validation_fails_on_missing_top_level_key(valid_config_dict):
    """Test that a config missing a required top-level key (e.g., 'datasets') fails."""
    invalid_config = valid_config_dict.copy()
    del invalid_config['datasets']
    with pytest.raises(SchemaError, match="Missing key: 'datasets'"):
        validate_config(invalid_config)

def test_config_validation_fails_on_missing_nested_key(valid_config_dict):
    """Test that a config missing a required nested key (e.g., 'path') fails."""
    invalid_config = valid_config_dict.copy()
    del invalid_config['datasets']['my_data']['path']
    with pytest.raises(SchemaError, match="Missing key: 'path'"):
        validate_config(invalid_config)

def test_config_validation_fails_on_incorrect_data_type(valid_config_dict):
    """Test that a config with an incorrect data type for a key fails."""
    invalid_config = valid_config_dict.copy()
    invalid_config['experiments'][0]['validation_setup']['forecast_steps'] = "ten"
    with pytest.raises(SchemaError):
        validate_config(invalid_config)

def test_get_model_config_retrieves_correctly(valid_config_dict, mocker):
    """Test retrieval of model config using a mocked load_config."""
    # Mock load_config to isolate the get_model_config function
    mocker.patch('utils.config_utils.load_config', return_value=valid_config_dict)

    model_config = get_model_config('arima', config_path="dummy_path.yaml")

    expected = {
        'p': 1, 'd': 1, 'q': 1, 'window_size': 12,
        'preprocessing': {'scaling': {'enabled': True}},
        'optimization': {'method': 'grid', 'params': {}} # This is added by default
    }
    assert model_config == expected

def test_get_model_config_raises_error_for_nonexistent_model(valid_config_dict, mocker):
    """Test that requesting a nonexistent model correctly raises a ValueError."""
    mocker.patch('utils.config_utils.load_config', return_value=valid_config_dict)

    with pytest.raises(ValueError, match="Invalid model name: nonexistent_model"):
        get_model_config('nonexistent_model', config_path="dummy_path.yaml")
