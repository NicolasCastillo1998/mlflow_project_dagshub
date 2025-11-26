import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
import seaborn as sns
import numpy as np

def evaluate(model_path, X_test, y_test, output_metrics="metrics.json"):
    # Cargar modelo
    model = joblib.load(model_path)

    # Predicciones
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Métricas
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    # Guardar métricas
    with open(output_metrics, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Métricas guardadas:", metrics)

    # ---------------- PLOT ROC CURVE ----------------
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % metrics["roc_auc"])
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.savefig("artifacts/roc_curve.png")
    plt.close()

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Matriz de Confusión")
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()

    print("Plots guardados en carpeta artifacts/")
