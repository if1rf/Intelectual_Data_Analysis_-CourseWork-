import os
import joblib
import json

from src.config import MODELS_DIR, METRICS_DIR


def save_model(model, name):
    os.makedirs(MODELS_DIR, exist_ok=True)

    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)

    print(f"[INFO] Model saved: {path}")


def load_model(name):
    path = MODELS_DIR / f"{name}.pkl"
    return joblib.load(path)


def save_metrics(metrics):
    os.makedirs(METRICS_DIR, exist_ok=True)

    path = METRICS_DIR / "metrics.json"

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Metrics saved: {path}")