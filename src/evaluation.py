import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from src.config import FIGURES_DIR


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)

    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(FIGURES_DIR / f"{name}_cm.png")
    plt.close()


def plot_roc_curve(y_test, y_proba, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(FIGURES_DIR / f"{name}_roc.png")
    plt.close()


def evaluate_all(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)

        results[name] = metrics

        print(f"\n[RESULT] {name}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        plot_confusion_matrix(y_test, y_pred, name)
        plot_roc_curve(y_test, y_proba, name)

    return results