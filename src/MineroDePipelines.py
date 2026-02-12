import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from .SecuenciaPreprocesamiento import SecuenciaPreprocesamiento

from src.preprocesamiento.BalanceadorDeClases import BalanceadorDeClases
from src.preprocesamiento.TratarDuplicados import TratarDuplicados
from src.preprocesamiento.TratarFaltantesNumericos import TratarFaltantesNumericos
from src.preprocesamiento.TratarFaltantesStrings import TratarFaltantesStrings

from src.preprocesamiento.CodificarVariablesBinarias import CodificarVariablesBinarias
from src.preprocesamiento.CodificarVariablesCategoricasRangoBajo import CodificarVariablesCategoricasRangoBajo
from src.preprocesamiento.CodificarVariablesCategoricasRangoMedio import CodificarVariablesCategoricasRangoMedio
from src.preprocesamiento.CodificarVariablesCategoricasRangoAlto import CodificarVariablesCategoricasRangoAlto

from src.preprocesamiento.TratarOutliersNumericos import TratarOutliersNumericos
from src.preprocesamiento.EscalarDatosNumericos import EscalarDatosNumericos
from src.preprocesamiento.NormalizarDatosNumericos import NormalizarDatosNumericos
from src.preprocesamiento.CrearNuevaVariable import CrearNuevaVariable
from src.preprocesamiento.SeleccionarVariables import SeleccionarVariables

class MineroDePipelines:
    def __init__(self):
        load_dotenv()
        # self._SEMILLA = int(os.getenv("SEMILLA_ALEATORIA", "42"))
        self._SEMILLA = None


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

        PERMITIR_NONE = False
        pipeline = Pipeline([
            ("tratar_duplicados", TratarDuplicados(PERMITIR_NONE, self._SEMILLA)),
            ("tratar_faltantes_numericos", TratarFaltantesNumericos(PERMITIR_NONE, self._SEMILLA)),
            ("tratar_faltantes_strings", TratarFaltantesStrings(PERMITIR_NONE, self._SEMILLA)),
            ("codificar_variables_binarias", CodificarVariablesBinarias(PERMITIR_NONE, self._SEMILLA)),
            ("codificar_variables_categoricas_rango_bajo", CodificarVariablesCategoricasRangoBajo(PERMITIR_NONE, self._SEMILLA)),
            ("codificar_variables_categoricas_rango_medio", CodificarVariablesCategoricasRangoMedio(PERMITIR_NONE, self._SEMILLA)),
            ("codificar_variables_categoricas_rango_alto", CodificarVariablesCategoricasRangoAlto(PERMITIR_NONE, self._SEMILLA)),
            ("tratar_outliers_numericos", TratarOutliersNumericos(PERMITIR_NONE, self._SEMILLA)),
            ("escalar_datos_numericos", EscalarDatosNumericos(PERMITIR_NONE, self._SEMILLA)),
            ("normalizar_datos_numericos", NormalizarDatosNumericos(PERMITIR_NONE, self._SEMILLA)),
            ("crear_nueva_variable", CrearNuevaVariable(PERMITIR_NONE, self._SEMILLA)),
            # ("seleccionar_variables", SeleccionarVariables(PERMITIR_NONE, self._SEMILLA))
        ])

        X_preprocesado = pipeline.fit_transform(X_train, y_train)
        print(f"Tamaño del conjunto de entrenamiento preprocesado: {X_preprocesado.shape}")
        print(f"Tipos de datos del conjunto de entrenamiento preprocesado: \n{X_preprocesado.dtypes}")

        print("Secuencia de preprocesamiento aplicada:")
        secuencia = SecuenciaPreprocesamiento()
        secuencia.imprimir_secuencia()

        return None

    