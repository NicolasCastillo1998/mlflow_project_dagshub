import yaml, joblib, pandas as pd
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
from pathlib import Path

params = yaml.safe_load(open('params.yaml'))
data_path = params['data']['processed_path']
target = params['data']['target']

def main():
    df = pd.read_csv(data_path)
    y = df[target]
    X = df.drop(columns=[target])

    model = joblib.load('models/model.joblib')
    proba = model.predict_proba(X)[:,1]

    fpr, tpr, _ = roc_curve(y, proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    Path('reports').mkdir(exist_ok=True, parents=True)
    plt.title('ROC Curve')
    plt.savefig('reports/roc_curve.png', bbox_inches='tight')
    print("Saved reports/roc_curve.png")

if __name__ == '__main__':
    main()
