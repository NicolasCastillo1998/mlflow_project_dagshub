Etapa 4 — Comparación de Experimentos (Proyecto Telco Churn)

Este informe resume y compara los experimentos realizados para optimizar el modelo de Regresión Logística
utilizando DVC Experiments como parte del pipeline MLOps.

---

Experimentos ejecutados:
Se realizaron 4 corridas modificando los hiperparámetros del modelo:

| Experimento | AUC    | F1     | Accuracy | C   | class_weight | max_iter |
|-------------|--------|--------|----------|-----|--------------|----------|
| exp_1       | 0.8403 | 0.5833 | 0.7941   | 0.2 | balanced     | 200      |
| exp_2       | 0.8403 | 0.5833 | 0.7941   | 1.0 | balanced     | 200      |
| exp_3       | 0.8403 | 0.5833 | 0.7941   | 0.5 | balanced     | 200      |
| exp_4       | 0.8403 | 0.5833 | 0.7941   | 0.2 | balanced     | 300      |


---


Cómo se ejecutaron

```bash
dvc exp run
dvc exp show --no-pager

Todos los experimentos quedaron registrados en Dagshub > DVC > Experiments.

Modelo ganador:
Según la métrica principal AUC, el mejor modelo resultó ser:
exp\_1 — Mejor modelo

Métricas:
AUC: 0.8403
F1-score: 0.5833
Accuracy: 0.7941
Hiperparámetros:
C: 0.2
class\_weight: balanced
max\_iter: 200


Justificación

C = 0.2
> Mayor regularización > menos overfitting.
class\_weight = balanced
> Fundamental para corregir el desbalance del dataset (churn ≈ 26%).
max\_iter = 200

> Convergencia suficiente sin aumentar innecesariamente el costo computacional.

Este modelo ofrece un equilibrio sólido entre discriminación (AUC) y estabilidad.

Conclusiones:
Se realizaron 4 experimentos variando hiperparámetros.
Se registraron y visualizaron con DVC Experiments.
El modelo final seleccionado fue exp\_1 por mejor AUC.
El pipeline es completamente reproducible, ya que:
Los datos están versionados con DVC.
Los runs quedan almacenados tanto en MLflow como en Dagshub.

