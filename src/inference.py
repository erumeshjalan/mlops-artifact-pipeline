import joblib
import numpy as np
import os

def load_model(model_path: str = "model_train.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    model = joblib.load(model_path)
    return model

def predict(model, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    return model.predict(input_array)

if __name__ == "__main__":
    # Example input (must match training features)
    sample_input = [5.1, 3.5, 1.4, 0.2]  # Replace with your dataset's actual feature size
    model = load_model()
    prediction = predict(model, sample_input)
    print(f"Prediction: {prediction}")
