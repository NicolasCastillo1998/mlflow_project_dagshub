import os
import pandas as pd

RAW = "data/raw/telco_churn.csv"
CLEAN = "scripts/data/telco_churn_processed.csv"

os.makedirs("scripts/data", exist_ok=True)

# Cargar
df = pd.read_csv(RAW)

# Quitar filas vacías
df = df.dropna(how="any")

# Quitar ID si existe
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Asegurar que 'Churn' sea binaria 0/1
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().str.lower().map({"yes": 1, "no": 0})

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Guardar
df.to_csv(CLEAN, index=False)
print(f"✅ Dataset limpio guardado en {CLEAN} - shape={df.shape}")
