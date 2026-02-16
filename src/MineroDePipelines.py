import pandas as pd
import numpy as np

from src.config.Configuracion import Configuracion
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    median_absolute_error, 
    explained_variance_score
)

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
        self._SEMILLA = None
        self._N_FOLDS = 3

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
        skf = StratifiedKFold(n_splits=self._N_FOLDS, shuffle=True, random_state=self._SEMILLA)

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
    
    def construir_pipeline_regresion(self, X_df: pd.DataFrame, y_df: pd.Series):
        # Se divide el dataset en 3 folds normales
        kf = KFold(n_splits=self._N_FOLDS, shuffle=True, random_state=self._SEMILLA)

        mae_scores = []
        mse_scores = []
        rmse_scores = []
        r2_scores = []
        medae_scores = []
        ev_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_df, y_df), 1):
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_df.iloc[train_idx], y_df.iloc[val_idx]

            print("="*100)
            print(f"Fold: {fold}")
            print("="*100)

            print("Preprocesando datos de entrenamiento...")
            X_train_preprocesado, y_train_preprocesado = self._preprocesar_datos(
                X_train.copy(), 
                y_train.copy(), 
                imprimir_resultados=False
            )

            if fold == 1:
                SecuenciaPreprocesamiento().guardar_secuencia()

            print("Seleccionando modelo de ML y configurando sus hiperparámetros...")
            selector_modelo = SelectorModeloRegresion(self._SEMILLA)
            selector_modelo.fit(X_train_preprocesado, y_train_preprocesado)

            print("Entrenando modelo de ML...")
            modelo_ml = selector_modelo.get_modelo_ml()
            modelo_ml.fit(X_train_preprocesado, y_train_preprocesado)

            print("Procesando datos de validación...")
            X_val, y_val = self._preprocesar_datos(
                X_val.copy(), 
                y_val.copy(), 
                imprimir_resultados=False
            )

            print("Evaluando modelo de ML en conjunto de validación...")
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

            print(f"Fold: {fold} procesado con éxito")
            
        print("="*100)
        print("Promedios finales".upper())
        print("="*100)
        print(f"{'MAE':<8}: {np.mean(mae_scores)}")
        print(f"{'MSE':<8}: {np.mean(mse_scores)}")
        print(f"{'RMSE':<8}: {np.mean(rmse_scores)}")
        print(f"{'R2':<8}: {np.mean(r2_scores)}")
        print(f"{'MedAE':<8}: {np.mean(medae_scores)}")
        print(f"{'EV':<8}: {np.mean(ev_scores)}")
        
        self._reiniciar_fases_pipeline()
        return None
    
    def _preprocesar_datos(self, X_copy: pd.DataFrame, y_copy: pd.Series, imprimir_resultados=False):
        configuracion = Configuracion()
        PERMITIR_NONE = configuracion.permitir_none
        PERMITIR_LLM = configuracion.permitir_llm
        SEMILLA = configuracion.semilla_aleatoria
        
        # Instanciamos cada fase del pipeline con los parámetros correspondientes.
        tratar_duplicados = TratarDuplicados(PERMITIR_NONE, SEMILLA)
        tratar_faltantes_numericos = TratarFaltantesNumericos(PERMITIR_NONE, SEMILLA)
        tratar_faltantes_strings = TratarFaltantesStrings(PERMITIR_NONE, SEMILLA)
        codificar_variables_binarias = CodificarVariablesBinarias(PERMITIR_NONE, SEMILLA)
        codificar_variables_categoricas_rango_bajo = CodificarVariablesCategoricasRangoBajo(PERMITIR_NONE, SEMILLA)
        codificar_variables_categoricas_rango_medio = CodificarVariablesCategoricasRangoMedio(PERMITIR_NONE, SEMILLA)
        codificar_variables_categoricas_rango_alto = CodificarVariablesCategoricasRangoAlto(PERMITIR_NONE, SEMILLA)
        tratar_outliers_numericos = TratarOutliersNumericos(PERMITIR_NONE, SEMILLA)
        escalar_datos_numericos = EscalarDatosNumericos(PERMITIR_NONE, SEMILLA)
        normalizar_datos_numericos = NormalizarDatosNumericos(PERMITIR_NONE, SEMILLA)
        crear_nueva_variable = CrearNuevaVariable(PERMITIR_NONE, SEMILLA)
        seleccionar_variables = SeleccionarVariables(PERMITIR_NONE, SEMILLA, "regresion")

        # El tratamiento que manipula Y debe hacerse fuera del pipeline
        tratar_duplicados.fit(X_copy, y_copy)
        X_copy, y_copy = tratar_duplicados.transform(X_copy, y_copy)

        tratar_faltantes_numericos.fit(X_copy, y_copy)
        X_copy, y_copy = tratar_faltantes_numericos.transform(X_copy, y_copy)

        tratar_faltantes_strings.fit(X_copy, y_copy)
        X_copy, y_copy = tratar_faltantes_strings.transform(X_copy, y_copy)

        pipeline = Pipeline([
            ("codificar_variables_binarias", codificar_variables_binarias),
            ("codificar_variables_categoricas_rango_bajo", codificar_variables_categoricas_rango_bajo),
            ("codificar_variables_categoricas_rango_medio", codificar_variables_categoricas_rango_medio),
            ("codificar_variables_categoricas_rango_alto", codificar_variables_categoricas_rango_alto),
        ])
        X_copy = pipeline.fit_transform(X_copy)

        tratar_outliers_numericos.fit(X_copy, y_copy)
        X_copy, y_preprocesado = tratar_outliers_numericos.transform(X_copy, y_copy)

        pipeline = Pipeline([
            ("escalar_datos_numericos", escalar_datos_numericos),
            ("normalizar_datos_numericos", normalizar_datos_numericos),
            ("crear_nueva_variable", crear_nueva_variable),
        ])
        X_copy = pipeline.fit_transform(X_copy, y_preprocesado)
        
        seleccionar_variables.fit(X_copy, y_preprocesado)
        X_preprocesado = seleccionar_variables.transform(X_copy, y_preprocesado)

        if imprimir_resultados:
            self._imprimir_resultado_pipeline(X_preprocesado, y_preprocesado)

        return X_preprocesado, y_preprocesado
    
    def _imprimir_resultado_pipeline(self, X_df, y_df):
        cols_with_nan = X_df.columns[X_df.isna().any()].tolist()

        print("\tPIPELINE COMPLETADO")

        # Forma y tamaño de X y y
        print(f"\tX: {X_df.shape}")
        print(f"\ty: {y_df.shape}\n")

        # Tipos de datos
        print("\tTipos de datos:")
        max_len = max(len(col) for col in X_df.columns)  # ancho dinámico
        for col, tipo in X_df.dtypes.items():
            print(f"\t{col:<{max_len}} : {tipo}")

        # Columnas con NaN
        print(f"\tColumnas con NaN: {cols_with_nan}")
        print("")

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
            print("Valores por defecto de las fases:")
            for nombre, fase in fases.items():
                print(f"{nombre}: {fase.log_params}")    
        
        print("="*100)
        print("Fases del pipeline reiniciadas a sus valores por defecto.")
        # imprimir_valores_por_defecto()
        print("="*100)

    def imprimir_secuencia_preprocesamiento(self):
        secuencia = SecuenciaPreprocesamiento()
        secuencia.imprimir_secuencia()
        print("=" * 100)