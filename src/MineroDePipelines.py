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
    explained_variance_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from .SecuenciaPreprocesamiento import SecuenciaPreprocesamiento
from .cash.SelectorModeloRegresion import SelectorModeloRegresion
from .cash.SelectorModeloClustering import SelectorModeloClustering
from .cash.SelectorModeloClasificacion import SelectorModeloClasificacion

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

    def construir_pipeline_clasificacion(self, X_df: pd.DataFrame, y_df: pd.Series) -> None:
        # Se divide el dataset en 3 folds normales
        skf = StratifiedKFold(n_splits=self._N_FOLDS, shuffle=True, random_state=self._SEMILLA)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y_df), 1):
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_df.iloc[train_idx], y_df.iloc[val_idx]

            print("="*100)
            print(f"Fold: {fold}")
            print("="*100)

            print("Preprocesando datos de entrenamiento...")
            X_train_preprocesado, y_train_preprocesado = self._preprocesar_datos(
                X_train.copy(), 
                y_train.copy(),
                tarea="clasificacion", 
                imprimir_resultados=False
            )

            if fold == 1:
                SecuenciaPreprocesamiento().guardar_secuencia()

            print("Seleccionando modelo de ML y configurando sus hiperparámetros...")
            selector_modelo = SelectorModeloClasificacion(self._SEMILLA)
            selector_modelo.fit(X_train_preprocesado, y_train_preprocesado)

            print("Entrenando modelo de ML...")
            modelo_ml = selector_modelo.get_modelo_ml()
            modelo_ml.fit(X_train_preprocesado, y_train_preprocesado)

            print("Procesando datos de validación...")
            X_val, y_val = self._preprocesar_datos(
                X_val.copy(), 
                y_val.copy(),
                tarea="clasificacion",
                imprimir_resultados=False
            )

            print("Evaluando modelo de ML en conjunto de validación...")
            print(f"Columnas con nulos: {X_val.columns[X_val.isna().any()].tolist()}")
            predicciones = modelo_ml.predict(X_val)

            accuracy = accuracy_score(y_val, predicciones)
            precision = precision_score(y_val, predicciones, average="weighted", zero_division=0)
            recall = recall_score(y_val, predicciones, average="weighted", zero_division=0)
            f1 = f1_score(y_val, predicciones, average="weighted", zero_division=0)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            print(f"Fold: {fold} procesado con éxito")
            
        print("="*100)
        print("Promedios finales".upper())
        print("="*100)
        print(f"{'Accuracy':<10}: {np.mean(accuracy_scores)}")
        print(f"{'Precision':<10}: {np.mean(precision_scores)}")
        print(f"{'Recall':<10}: {np.mean(recall_scores)}")
        print(f"{'F1':<10}: {np.mean(f1_scores)}")
        
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
                tarea="regresion",
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
                tarea="regresion",
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
    
    def construir_pipeline_clustering(self, X_df: pd.DataFrame):
        X_train = X_df.copy()

        print("Preprocesando datos de entrenamiento...")
        X_train_preprocesado, _ = self._preprocesar_datos(
            X_train.copy(), 
            None, 
            tarea="clustering",
            imprimir_resultados=False
        )


        print("Seleccionando modelo de ML y configurando sus hiperparámetros...")
        selector_modelo = SelectorModeloClustering(self._SEMILLA)
        selector_modelo.fit(X_train_preprocesado)

        print("Entrenando modelo de ML...")
        modelo_ml = selector_modelo.get_modelo_ml()
        SecuenciaPreprocesamiento().guardar_secuencia()
        
        etiquetas = modelo_ml.fit_predict(X_train_preprocesado)

        numero_etiquetas = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)

        silhouette = None
        # El Silhouette Score falla si solo hay 1 grupo o si cada punto es su propio grupo
        if 1 < numero_etiquetas < len(X_train_preprocesado):
            try:
                # Usar 'sample_size' para velocidad en Big Data
                # Si X tiene < 10k filas, usa todo. Si tiene más, usa una muestra de 10k.
                tamaño_muestra = 10_000 if X_train_preprocesado.shape[0] > 10_000 else None
                
                silhouette = silhouette_score(
                    X_train_preprocesado, 
                    etiquetas, 
                    metric='euclidean', 
                    sample_size=tamaño_muestra,  # ¡Esto salva tu CPU!
                    random_state=42
                )

            except Exception as e:
                silhouette = -1.0 # Error en cálculo
        else:
            # Castigo fuerte si el algoritmo colapsó todo en 1 solo grupo
            silhouette = -1.0
    
        calinski = None
        try:
            calinski = calinski_harabasz_score(X_train_preprocesado, etiquetas)
        except:
            calinski = 0.0

        davies = None
        try:
            davies = davies_bouldin_score(X_train_preprocesado, etiquetas)
        except:
            davies = 999.0 # Valor malo por defecto
        
        print("="*100)
        print("Resultados finales".upper())
        print("="*100)
        print(f"{'Silhouette':<8}: {silhouette}")
        print(f"{'Calinski-Harabasz':<8}: {calinski}")
        print(f"{'Davies-Bouldin':<8}: {davies}")
        
        self._reiniciar_fases_pipeline()
        return None
    
    def _preprocesar_datos(self, X_copy: pd.DataFrame, y_copy: pd.Series, tarea: str, imprimir_resultados=False):
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
        seleccionar_variables = SeleccionarVariables(PERMITIR_NONE, SEMILLA, tarea)

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