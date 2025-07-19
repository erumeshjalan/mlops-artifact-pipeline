from src.utils import (
    load_config,
    load_data,
    train_model,
    save_model,
    evaluate_model
)

def main():
    # Load configuration
    config = load_config("config/config.json")

    # Load dataset
    X, y = load_data()

    # Train model
    model = train_model(X, y, config)

    # Save model
    save_model(model, "models/model_train.pkl")

    # Evaluate model
    accuracy, f1 = evaluate_model(model, X, y)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
