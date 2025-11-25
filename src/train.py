import json, joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from pathlib import Path
import mlflow

# Rutas fijas (coinciden con DVC)
data_path = "scripts/data/telco_churn_processed.csv"
target = "Churn"

def main():
    # Cargar dataset procesado
    df = pd.read_csv(data_path)
    y = df[target]
    X = df.drop(columns=[target])

    # Configuración fija del split
    test_size = 0.2
    random_state = 42
    stratify = y

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Modelo: regresión logística
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predicciones
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    # Métricas
    metrics = {
        "auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }

    # Crear directorio de salida (DVC lo usa)
    Path('scripts').mkdir(exist_ok=True, parents=True)

    # Guardar métricas
    with open('scripts/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Guardar modelo
    joblib.dump(clf, 'scripts/model.joblib')

    print("✅ Modelo entrenado con éxito")
    print("📊 Métricas:", metrics)

if __name__ == "__main__":
    main()