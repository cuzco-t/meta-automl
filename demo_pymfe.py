from matplotlib import pyplot as plt
import pandas as pd

# Cargar el dataset
df = pd.read_csv("./data/descriptive_practice_dataset_large.csv", encoding="utf-8")
print(df.dtypes)

df.to_numpy()
print(df.dtypes)


# # Seleccionar solo columnas numéricas
# numeric_cols = df.select_dtypes(include='number').columns
# print("Columnas numéricas:", numeric_cols)

# # Dibujar un boxplot por cada columna numérica
# for col in numeric_cols:
#     plt.figure(figsize=(6, 4))
#     df.boxplot(column=col)
#     plt.title(f"Boxplot de {col}")
#     plt.show()
