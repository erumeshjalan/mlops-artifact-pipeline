import os
import pytest
from src.train import load_config, train_model

def test_load_config():
    config = load_config("config/config.json")
    assert isinstance(config, dict)
    assert "C" in config
    assert "solver" in config

def test_train_model_output():
    config = load_config("config/config.json")
    model = train_model(config)
    assert model is not None
