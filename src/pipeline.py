from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.modeling import get_models
from src.training import train_models
from src.evaluation import evaluate_all
from src.eda import run_eda
from src.persistence import save_model, save_metrics


def run_pipeline():
    df = load_data()

    run_eda(df)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = get_models()
    trained = train_models(models, X_train, y_train)

    results = evaluate_all(trained, X_test, y_test)

    for name, model in trained.items():
        save_model(model, name)

    save_metrics(results)

    return results