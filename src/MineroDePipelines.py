import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from .Preprocesamiento import Preprocesamiento
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

class MineroDePipelines:
    def __init__(self):
        load_dotenv()
        self.preprocesamiento = Preprocesamiento()
        self._SEMILLA = int(os.getenv("SEMILLA_ALEATORIA", "42"))


    def _leer_dataset(self, ruta_absoluta, target) -> tuple[pd.DataFrame, pd.Series]:
        """
        Lee el dataset desde la ruta absoluta proporcionada y separa las características (X) del target (y).
        
        :param self: Referencia de la instancia de la clase.
        :param ruta_absoluta: Ruta absoluta del archivo CSV que contiene el dataset.
        :param target: Nombre de la columna que se utilizará como target.
        :return: Tuple con las características (X) y el target (y) del dataset
        :rtype: tuple
        """
        self.df = pd.read_csv(ruta_absoluta, encoding="utf-8")
        X_df = self.df.drop(columns=[target])
        y_df = self.df[target]

        return X_df, y_df

    def construir_pipeline_clasificacion(self, ruta_absoluta, target):
        # Leer el dataset y separar características y target
        X, y = self._leer_dataset(ruta_absoluta, target)
        print(f"Tamaño del dataset original: {X.shape}, {y.shape}")
        print(f"Tipos de datos originales: {X.dtype}, {y.dtype}")
        # Aquí se podrían agregar más pasos al pipeline, como selección de características, etc.
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self._SEMILLA)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            X_preprocesado, y_preprocesado, secuencia = self.preprocesamiento.preprocesar_datos(X_train, y_train)
            print(f"Fold {fold} - Secuencia de preprocesamiento: {secuencia}")
            print(f"Fold {fold} - Tamaño del conjunto de entrenamiento preprocesado: {X_preprocesado.shape}, {y_preprocesado.shape}")
            print(f"Fold {fold} - Tamaño del conjunto de validación: {X_val.shape}, {y_val.shape}")
            print("-" * 50)

        return None
    
    def construir_pipeline_regresion(self, ruta_absoluta, target):
        # Leer el dataset y separar características y target
        X_df, y_df = self._leer_dataset(ruta_absoluta, target)
        print(f"Tamaño del dataset original: {X_df.shape}, {y_df.shape}")
        print(f"Tipos de datos originales:")
        for i in range(X_df.shape[1]):
            print(f"Columna {i}: dtype = {X_df.iloc[:, i].dtype}")

        # Se divide en dataset en 80% para entrenamiento y 20% para validación
        X_train, X_val, y_train, y_val = train_test_split(
            X_df, y_df, test_size=0.2, random_state=self._SEMILLA
        )

        print(f"Tamaño del conjunto de entrenamiento: X={X_train.shape}, y={y_train.shape}")
        print("Tipos de datos del conjunto de entrenamiento:")

        for i in range(X_train.shape[1]):
            print(f"Columna {i}: dtype = {X_train.iloc[:, i].dtype}")


        # Aquí se podrían agregar más pasos al pipeline, como selección de características, etc.
        X_preprocesado, y_preprocesado, secuencia = self.preprocesamiento.preprocesar_datos(X_train, y_train, tarea="regresion")
        print(f"Tamaño del conjunto de entrenamiento preprocesado: {X_preprocesado.shape}, {y_preprocesado.shape}")
        print(f"Tamaño del conjunto de validación: {X_val.shape}, {y_val.shape}")
        print("-" * 50)
        print("\nSecuencia de preprocesamiento:")
        for clave, valor in secuencia.items():
            print(f"{clave}: {valor}")
    