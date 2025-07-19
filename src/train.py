import json
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


def load_config(path='config/config.json'):
    with open(path, 'r') as f:
        return json.load(f)


def train_model(X_train, y_train, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"],
        random_state=config["random_state"]
    )
    model.fit(X_train, y_train)
    return model


def main():
    # Load config
    config = load_config()

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["random_state"]
    )

    # Train model
    model = train_model(X_train, y_train, config)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_train.pkl")

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
