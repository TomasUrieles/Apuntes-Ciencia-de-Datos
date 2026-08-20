import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Cargar CSV
df = pd.read_csv("pacientes.csv")

# 2. Seleccionar las columnas requeridas
df = df[["edad", "colesterol", "problema_cardiaco"]]

# 3. Eliminar registros con datos faltantes
df = df.dropna(subset=["edad", "colesterol"])

# 4. Renombrar columnas
df = df.rename(columns={
    "edad": "x",
    "colesterol": "y",
    "problema_cardiaco": "label"
})

# 5. Convertir target:
# 0 -> -1
# 1 -> 1
df["label"] = df["label"].map({
    0: -1,
    1: 1
})

# 6. Estandarización Z-score
scaler = StandardScaler()

df[["x", "y"]] = scaler.fit_transform(df[["x", "y"]])

# 7. Multiplicar por 2
df[["x", "y"]] = df[["x", "y"]] * 2

# 8. Convertir a JSON
df.to_json(
    "pacientes_preprocesados.json",
    orient="records",
    indent=2
)

print("Archivo generado correctamente.")
print("\nPrimeros registros:")
print(df.head())

print("\nCantidad final de registros:")
print(len(df))

print("\nDistribución de clases:")
print(df["label"].value_counts())


print(df.columns)
print(df["label"].unique())

