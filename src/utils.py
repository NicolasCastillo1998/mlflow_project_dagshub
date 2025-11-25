import pandas as pd
from pathlib import Path

CATEGORICAL = [
    "gender","region","contract_type","internet_service",
    "phone_service","multiple_lines","payment_method"
]

NUMERIC = ["age","tenure_months","monthly_charges","total_charges"]

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
