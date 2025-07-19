from src.utils import load_config, load_data, train_model, evaluate_model

def test_train_model_output():
    config = load_config("config/config.json")
    X, y = load_data()
    model = train_model(X, y, config)
    assert model is not None

def test_model_evaluation():
    config = load_config("config/config.json")
    X, y = load_data()
    model = train_model(X, y, config)
    accuracy, f1 = evaluate_model(model, X, y)
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= f1 <= 1.0
