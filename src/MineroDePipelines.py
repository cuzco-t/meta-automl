import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score

from .SecuenciaPreprocesamiento import SecuenciaPreprocesamiento
from .cash.SelectorModeloRegresion import SelectorModeloRegresion

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
        self._N_SPLITS = 3

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
        X_df, y_df = self._leer_dataset(ruta_absoluta, target)

        # Se divide el dataset en 3 folds normales
        skf = StratifiedKFold(n_splits=self._N_SPLITS, shuffle=True, random_state=self._SEMILLA)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y_df), 1):
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_df.iloc[train_idx], y_df.iloc[val_idx]

            print(f"Fold {fold}")
            print(f"\tTamaño del conjunto de entrenamiento: X={X_train.shape}, y={y_train.shape}")
            print(f"\tTipos de datos del conjunto de entrenamiento:\n{X_train.dtypes}")
            print("="*100)

            X_seleccionado, y_train = self._preprocesar_datos(X_train, y_train)
            SecuenciaPreprocesamiento().guardar_secuencia()
            # ====================================================================================================
            # Hasta aquí se ha aplicado la secuencia de preprocesamiento al conjunto de entrenamiento. 
            # Ahora se eligirá el modelo de ML y sus hiperparámetros, para realizar el entrenamiento del modelo.
            # ====================================================================================================
            selector_modelo = SelectorModeloRegresion(self._SEMILLA)
            modelo_ml = selector_modelo.get_modelo_ml(X_seleccionado, y_train)

            modelo_ml.fit(X_seleccionado, y_train)

            print()
            print("Modelo entrenado. Evaluando en conjunto de validación...".upper())
            print()

            X_val, y_val = self._preprocesar_datos(X_val, y_val)
            predicciones = modelo_ml.predict(X_val)

            mae = mean_absolute_error(y_val, predicciones)
            mse = mean_squared_error(y_val, predicciones)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, predicciones)
            medae = median_absolute_error(y_val, predicciones)
            ev = explained_variance_score(y_val, predicciones)

            acc = accuracy_score(y_val, predicciones)
            prec = precision_score(y_val, predicciones)
            rec = recall_score(y_val, predicciones)
            f1 = f1_score(y_val, predicciones)

            accuracy_scores.append(acc)
            precision_scores.append(prec)
            recall_scores.append(rec)
            f1_scores.append(f1)
            

            print(f"Fold {fold}")
            print(f"\tAccuracy : {acc:.4f}")
            print(f"\tPrecision: {prec:.4f}")
            print(f"\tRecall   : {rec:.4f}")
            print(f"\tF1       : {f1:.4f}")
            print("=" * 100)

        print("Promedios finales:")
        print("Accuracy :", np.mean(accuracy_scores))
        print("Precision:", np.mean(precision_scores))
        print("Recall   :", np.mean(recall_scores))
        print("F1       :", np.mean(f1_scores))
        
        self._reiniciar_fases_pipeline()

        return None
    
    def construir_pipeline_regresion(self, ruta_absoluta, target):
        # Leer el dataset y separar características y target
        X_df, y_df = self._leer_dataset(ruta_absoluta, target)

        # Se divide el dataset en 3 folds normales
        kf = KFold(n_splits=self._N_SPLITS, shuffle=True, random_state=self._SEMILLA)

        mae_scores = []
        mse_scores = []
        rmse_scores = []
        r2_scores = []
        medae_scores = []
        ev_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_df, y_df), 1):
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_df.iloc[train_idx], y_df.iloc[val_idx]

            print(f"Fold {fold}")
            print(f"Tamaño del conjunto de entrenamiento: X={X_train.shape}, y={y_train.shape}")
            print(f"Tipos de datos del conjunto de entrenamiento:\n{X_train.dtypes}")
            print("="*100)

            X_seleccionado, y_train = self._preprocesar_datos(X_train, y_train)
            SecuenciaPreprocesamiento().guardar_secuencia()
            # ====================================================================================================
            # Hasta aquí se ha aplicado la secuencia de preprocesamiento al conjunto de entrenamiento. 
            # Ahora se eligirá el modelo de ML y sus hiperparámetros, para realizar el entrenamiento del modelo.
            # ====================================================================================================
            selector_modelo = SelectorModeloRegresion(self._SEMILLA)
            modelo_ml = selector_modelo.get_modelo_ml(X_seleccionado, y_train)

            modelo_ml.fit(X_seleccionado, y_train)

            print()
            print("Modelo entrenado. Evaluando en conjunto de validación...".upper())
            print()

            X_val, y_val = self._preprocesar_datos(X_val, y_val)
            predicciones = modelo_ml.predict(X_val)

            mae = mean_absolute_error(y_val, predicciones)
            mse = mean_squared_error(y_val, predicciones)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, predicciones)
            medae = median_absolute_error(y_val, predicciones)
            ev = explained_variance_score(y_val, predicciones)

            mae_scores.append(mae)
            mse_scores.append(mse)
            rmse_scores.append(rmse)
            r2_scores.append(r2)
            medae_scores.append(medae)
            ev_scores.append(ev)
            
            print(f"Fold {fold}")
            print(f"  MAE      : {mae:.4f}")
            print(f"  MSE      : {mse:.4f}")
            print(f"  RMSE     : {rmse:.4f}")
            print(f"  R2       : {r2:.4f}")
            print(f"  MedAE    : {medae:.4f}")
            print(f"  EV       : {ev:.4f}")
            print("=" * 100)

        print("Promedios finales:")
        print("MAE      :", np.mean(mae_scores))
        print("MSE      :", np.mean(mse_scores))
        print("RMSE     :", np.mean(rmse_scores))
        print("R2       :", np.mean(r2_scores))
        print("MedAE    :", np.mean(medae_scores))
        print("EV       :", np.mean(ev_scores))
        
        self._reiniciar_fases_pipeline()
        return None
    
    def _preprocesar_datos(self, X: pd.DataFrame, y: pd.Series):
        X_copy = X.copy()
        y_copy = y.copy()
        PERMITIR_NONE = False
        
        # El tratamiento que manipula Y debe hacerse fuera del pipeline
        tratar_duplicados = TratarDuplicados(PERMITIR_NONE, self._SEMILLA)
        tratar_faltantes_numericos = TratarFaltantesNumericos(PERMITIR_NONE, self._SEMILLA)
        tratar_faltantes_strings = TratarFaltantesStrings(PERMITIR_NONE, self._SEMILLA)
        
        tratar_duplicados.fit(X_copy, y_copy)
        X_copy, y_copy = tratar_duplicados.transform(X_copy, y_copy)
        tratar_faltantes_numericos.fit(X_copy, y_copy)
        X_copy, y_copy = tratar_faltantes_numericos.transform(X_copy, y_copy)
        tratar_faltantes_strings.fit(X_copy, y_copy)
        X_copy, y_copy = tratar_faltantes_strings.transform(X_copy, y_copy)

        pipeline = Pipeline([
            ("codificar_variables_binarias", CodificarVariablesBinarias(PERMITIR_NONE, self._SEMILLA)),
            ("codificar_variables_categoricas_rango_bajo", CodificarVariablesCategoricasRangoBajo(PERMITIR_NONE, self._SEMILLA)),
            ("codificar_variables_categoricas_rango_medio", CodificarVariablesCategoricasRangoMedio(PERMITIR_NONE, self._SEMILLA)),
            ("codificar_variables_categoricas_rango_alto", CodificarVariablesCategoricasRangoAlto(PERMITIR_NONE, self._SEMILLA)),
        ])
        X_preprocesado = pipeline.fit_transform(X_copy)

        tratar_outliers_numericos = TratarOutliersNumericos(PERMITIR_NONE, self._SEMILLA)
        tratar_outliers_numericos.fit(X_preprocesado, y_copy)
        X_preprocesado, y_copy = tratar_outliers_numericos.transform(X_preprocesado, y_copy)

        pipeline = Pipeline([
            ("escalar_datos_numericos", EscalarDatosNumericos(PERMITIR_NONE, self._SEMILLA)),
            ("normalizar_datos_numericos", NormalizarDatosNumericos(PERMITIR_NONE, self._SEMILLA)),
            ("crear_nueva_variable", CrearNuevaVariable(PERMITIR_NONE, self._SEMILLA)),
        ])
        
        X_preprocesado = pipeline.fit_transform(X_preprocesado, y_copy)
        
        cols_with_nan = X_preprocesado.columns[X_preprocesado.isnull().any()].tolist()
        print("PIPELINE COMPLETADO")
        print(f"Tamaño del conjunto de entrenamiento después del pipeline: {X_preprocesado.shape}, {y_copy.shape}")
        print(f"Tipos de datos del conjunto de entrenamiento después del pipeline: \n{X_preprocesado.dtypes}")
        print("Columnas con NaN:", cols_with_nan)
        print("="*100)

        seleccionar_variables = SeleccionarVariables(PERMITIR_NONE, "regresion", self._SEMILLA)
        seleccionar_variables.fit(X_preprocesado, y_copy)
        X_seleccionado = seleccionar_variables.transform(X_preprocesado, y_copy)

        print("SELECCIÓN DE VARIABLES COMPLETADA")
        print(f"Tamaño del conjunto de entrenamiento preprocesado: {X_seleccionado.shape}, {y_copy.shape}")
        print(f"Tipos de datos del conjunto de entrenamiento preprocesado: \n{X_seleccionado.dtypes}")
        print("="*100)

        return X_seleccionado, y_copy

    
    def _reiniciar_fases_pipeline(self):
        fases = {
            "tratar_duplicados": TratarDuplicados(),
            "tratar_faltantes_numericos": TratarFaltantesNumericos(),
            "tratar_faltantes_strings": TratarFaltantesStrings(),
            "codificar_variables_binarias": CodificarVariablesBinarias(),
            "codificar_variables_categoricas_rango_bajo": CodificarVariablesCategoricasRangoBajo(),
            "codificar_variables_categoricas_rango_medio": CodificarVariablesCategoricasRangoMedio(),
            "codificar_variables_categoricas_rango_alto": CodificarVariablesCategoricasRangoAlto(),
            "tratar_outliers_numericos": TratarOutliersNumericos(),
            "escalar_datos_numericos": EscalarDatosNumericos(),
            "normalizar_datos_numericos": NormalizarDatosNumericos(),
            "crear_nueva_variable": CrearNuevaVariable(),
            "seleccionar_variables": SeleccionarVariables(),
            "selector_modelo_regresion": SelectorModeloRegresion()
        }
        
        [fase.reiniciar() for fase in fases.values()]

        def imprimir_valores_por_defecto():
            for nombre, fase in fases.items():
                print(f"{nombre}: {fase.log_params}")    
        
        print("Valores por defecto después de reiniciar:")
        imprimir_valores_por_defecto()

    def imprimir_secuencia_preprocesamiento(self):
        secuencia = SecuenciaPreprocesamiento()
        secuencia.imprimir_secuencia()
        print("=" * 100)