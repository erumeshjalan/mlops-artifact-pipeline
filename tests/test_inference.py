import pytest
from src.inference_module import load_model, predict

def test_load_model():
    model = load_model("models/model_train.pkl")
    assert model is not None

def test_predict_output():
    model = load_model("models/model_train.pkl")
    sample_input = [5.1, 3.5, 1.4, 0.2]
    prediction = predict(model, sample_input)
    assert len(prediction) == 1
