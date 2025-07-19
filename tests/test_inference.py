import os
import sys
import numpy as np
from sklearn.datasets import load_digits

# Ensure 'src' is on the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from inference_module import load_model, predict

def test_predict_output():
    model = load_model("models/model_train.pkl")

    # Use real data for realistic testing
    X, _ = load_digits(return_X_y=True)
    input_data = X[0]  # a valid 64-feature input

    prediction = predict(model, input_data)

    assert prediction.shape == (1,), "Prediction should return one result"
    assert isinstance(prediction[0], (int, np.integer)), "Prediction should be an integer class label"
