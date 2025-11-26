Proyecto de Machine Learning con MLflow, DVC y GitHub Actions

Autor: Nicolás Castillo

Año: 2025

Este proyecto implementa un pipeline completo de Machine Learning reproducible, utilizando:

MLflow para el seguimiento de experimentos
DVC (Data Version Control) para gestión versionada de datos y artefactos
GitHub Actions para CI/CD del pipeline
Dagshub como repositorio remoto de datos y experimentos

El objetivo es entrenar, evaluar y comparar modelos para predecir churn en un dataset de telecomunicaciones, asegurando trazabilidad, reproducibilidad y buenas prácticas de MLOps.

1. Estructura general del proyecto

```bash
├── data/
│   ├── raw/                      # Dataset original versionado con DVC
│   ├── data-file.bin             # Datos procesados (DVC)
│
├── scripts/
│   ├── data/                     # Procesamientos intermedios
│   ├── model.joblib              # Modelo final entrenado
│   ├── telco_churn_processed.csv # Dataset procesado
│
├── artifacts/                    
│   ├── roc_curve.png             # Curva ROC
│   ├── confusion_matrix.png      # Matriz de confusión
│   ├── metrics.json              # Métricas de evaluación
│
├── src/
│   ├── train.py                  # Entrenamiento de modelo
│   ├── evaluate.py               # Evaluación + generación de plots
│
├── params.yaml                   # Hiperparámetros del modelo
├── dvc.yaml                      # Pipeline completo de DVC
├── dvc.lock                      # Lock del pipeline
├── .github/workflows/ci.yml      # CI/CD con GitHub Actions
├── mlruns/                       # Tracking de MLflow
├── README.md                     # Documentación del proyecto
```

2. Pipeline de DVC

El pipeline versionado en dvc.yaml consta de tres etapas principales:

1) data_prep
Lee el dataset crudo raw/telco_churn.csv.dvc
Aplica encoding, limpieza y escalado
Guarda datos procesados en data/data-file.bin
Reproducible: cualquier cambio en raw dispara el pipeline

2) train
Lee hiperparámetros desde params.yaml
Entrena un modelo de Logistic Regression
Guarda model.joblib
Registra métricas en MLflow

3) evaluate
Genera métricas adicionales:
Accuracy
F1-score
ROC-AUC

Produce artefactos:
roc_curve.png
confusion_matrix.png
metrics.json
Se versionan con DVC

Ejecutar el pipeline completo:
dvc repro

3. Experimentos con MLflow

Todos los entrenamientos quedan registrados bajo mlruns/.

Métricas registradas automáticamente:
auc
accuracy
f1
Tamaño de train/test
Artefactos de cada modelo
Hiperparámetros utilizados

Esto permite comparar objetivamente cada variante del modelo.

4. CI/CD con GitHub Actions

Se configuró un workflow en:
.github/workflows/ci.yml

Este job ejecuta automáticamente en cada Pull Request:
Instala dependencias
Autentica DVC contra Dagshub usando Secrets:
DAGSHUB_USER
DAGSHUB_TOKEN

Ejecuta:
Este job ejecuta automáticamente en cada Pull Request:

Instala dependencias

Autentica DVC contra Dagshub usando Secrets:

DAGSHUB_USER

DAGSHUB_TOKEN

Ejecuta:
dvc pull
dvc repro

Verifica que el pipeline NO falle
Publica logs y métricas
El PR sólo puede mergearse si el pipeline pasa correctamente.

5. Iteración colaborativa (feat-branches)
Se trabajó siguiendo buenas prácticas de Gitflow:

Ramas creadas:
feat-ci
feat-C05
feat-C1
feat-solver-sag
feat-best-model

Cada rama contiene modificaciones concretas en params.yaml o mejoras del pipeline.
Todos los PR fueron evaluados por CI y mergeados a main tras aprobarse.

Ejemplos de experimentos:
| Rama              | Cambio realizado                                     | Resultado              |
| ----------------- | ---------------------------------------------------- | ---------------------- |
| `feat-C05`        | `C=0.5`                                              | + buena regularización |
| `feat-C1`         | `C=1.0`                                              | baseline sólido        |
| `feat-solver-sag` | solver="sag" + balanceo                              | mejor AUC              |
| `feat-best-model` | combinación de sag + balanced + C=0.5 + max_iter=200 | modelo final           |


6. Modelo final
El mejor rendimiento se obtuvo con:
solver: sag
class_weight: balanced
C: 0.5
max_iter: 200

Métricas finales
Accuracy: 0.794
F1-score: 0.583
ROC-AUC: 0.840

Artefactos generados

ROC Curve: artifacts/roc_curve.png
Matriz de Confusión: artifacts/confusion_matrix.png
Métricas JSON versionadas: metrics.json

Todos los artefactos están versionados bajo DVC y disponibles vía dvc pull.

7. Instrucciones de despliegue (FastAPI / Real Time Prediction)
Ejemplo básico (no implementado en este repo pero documentado para producción):
app.py

from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("scripts/model.joblib")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}

Ejecutar servicio:
uvicorn app:app --reload

Esto habilita un endpoint para predicción en tiempo real con el modelo final.

8. Cómo reproducir el proyecto
Clonar el repo:

git clone https://github.com/NicolasCastillo1998/mlflow_project_dagshub.git
cd mlflow_project_dagshub

Instalar dependencias:
pip install -r requirements.txt

Obtener datos y artefactos:
dvc pull

Correr pipeline completo:
dvc repro

9. Conclusión

Este proyecto integra MLOps moderno aplicado en un contexto real, incluyendo:
pipelines reproducibles
versionado de datos
experimentación controlada
automatización CI/CD
trazabilidad completa del modelo

Representa un workflow profesional y escalable para machine learning en producción.
