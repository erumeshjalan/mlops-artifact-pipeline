import pytest
import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from utils import load_config, load_data, train_model, save_model, load_model, evaluate_model

class TestTrainingPipeline:
    """Test suite for training pipeline components."""

    def test_config_file_loading(self):
        """Test that configuration file loads successfully."""
        config = load_config('config/config.json')

        # Check that config is loaded
        assert config is not None
        assert isinstance(config, dict)

        # Check required hyperparameters exist
        required_params = ['C', 'solver', 'max_iter']
        for param in required_params:
            assert param in config, f"Missing required parameter: {param}"

        # Check data types
        assert isinstance(config['C'], (int, float)), "C should be numeric"
        assert isinstance(config['solver'], str), "solver should be string"
        assert isinstance(config['max_iter'], int), "max_iter should be integer"

        # Check reasonable values
        assert config['C'] > 0, "C should be positive"
        assert config['max_iter'] > 0, "max_iter should be positive"

    def test_model_creation(self):
        """Test that model creation returns LogisticRegression object."""
        X, y = load_data()
        config = load_config('config/config.json')

        model = train_model(X, y, config)

        # Verify model type
        assert isinstance(model, LogisticRegression), "Model should be LogisticRegression instance"

        # Check model is fitted
        assert hasattr(model, 'coef_'), "Model should have coef_ after fitting"
        assert hasattr(model, 'classes_'), "Model should have classes_ after fitting"

        # Match model params with config
        assert model.C == config['C']
        assert model.solver == config['solver']
        assert model.max_iter == config['max_iter']

    def test_model_accuracy(self):
        """Test that model achieves reasonable accuracy."""
        X, y = load_data()
        config = load_config('config/config.json')
        model = train_model(X, y, config)
        accuracy, f1 = evaluate_model(model, X, y)

        assert accuracy > 0.85, f"Accuracy {accuracy:.4f} below threshold"
        assert f1 > 0.85, f"F1 Score {f1:.4f} below threshold"

        # Ensure output shape
        predictions = model.predict(X)
        assert predictions.shape[0] == X.shape[0]
        assert set(predictions).issubset(set(y))

    def test_data_loading(self):
        """Test data loading functionality."""
        X, y = load_data()

        assert X.shape[0] > 0
        assert X.shape[1] == 64  # digits dataset = 8x8 image
        assert len(y) == X.shape[0]
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        # Ensure labels are 0–9
        assert set(y) == set(range(10))

    def test_model_save_load(self):
        """Test saving and loading of model."""
        X, y = load_data()
        config = load_config('config/config.json')
        model = train_model(X, y, config)
        test_model_path = 'test_model.pkl'

        save_model(model, test_model_path)
        assert os.path.exists(test_model_path)

        loaded_model = load_model(test_model_path)
        assert isinstance(loaded_model, LogisticRegression)

        # Check prediction consistency
        original_pred = model.predict(X[:10])
        loaded_pred = loaded_model.predict(X[:10])
        np.testing.assert_array_equal(original_pred, loaded_pred)

        # Cleanup
        os.remove(test_model_path)
