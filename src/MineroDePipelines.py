import time

import pandas as pd
import numpy as np

from src.config.Configuracion import Configuracion
from .PipelineLogger import PipelineLogger
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
        self._SEMILLA = Configuracion().semilla_aleatoria
        self._N_FOLDS = 3
        self._logger = PipelineLogger().get_logger()
        self._id_pipeline = 0
        self._id_ejecucion_modelo = 0
        self._selector_aleatorio = np.random.default_rng(self._SEMILLA)

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

        codificar_variables_binarias.fit(X_copy, y_copy)
        X_copy = codificar_variables_binarias.transform(X_copy)

        tratar_faltantes_numericos.fit(X_copy, y_copy)
        X_copy, y_copy = tratar_faltantes_numericos.transform(X_copy, y_copy)

        tratar_faltantes_strings.fit(X_copy, y_copy)
        X_copy, y_copy = tratar_faltantes_strings.transform(X_copy, y_copy)

        pipeline = Pipeline([
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
    
    def pipeline_supervisado(self, X_df: pd.DataFrame, y_df: pd.Series, tarea: str):

        def get_k_fold_coss_validation(n_folds: int = 3) -> StratifiedKFold | KFold:
            """
            Devuelve un objeto de validación cruzada K-Fold o Stratified K-Fold según la tarea.
            
            :param n_folds: Número de folds para la validación cruzada.
            :return: Un objeto de validación cruzada K-Fold o Stratified K-Fold configurado
             con el número de folds y la semilla aleatoria.
            :rtype: StratifiedKFold | KFold
            """
            if tarea == "clasificacion":
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self._SEMILLA)
            else:
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=self._SEMILLA)
            
            return kf
        
        def get_diccionario_metricas_inicializadas() -> dict[str, dict]:
            """
            Inicializa un diccionario de listas para almacenar las métricas de evaluación según la tarea.
            
            :return: Un diccionario donde las claves son los nombres de las métricas 
             y los valores son listas vacías para almacenar los resultados de cada fold.
            :rtype: dict[str, dict]
            """
            diccionario_metricas = None
            if tarea == "clasificacion":
                diccionario_metricas = {
                    "accuracy_scores": {},
                    "precision_scores": {},
                    "recall_scores": {},
                    "f1_scores": {},
                }
            else:
                diccionario_metricas = {
                    "mae_scores": {},
                    "mse_scores": {},
                    "rmse_scores": {},
                    "r2_scores": {},
                    "medae_scores": {},
                    "ev_scores": {},
                }

            return diccionario_metricas

        def actualizar_metricas_fold(
            y_true, 
            y_pred, 
            numero_ejecucion_modelo: int, 
            metricas: dict[str, dict]
        ) -> None:
            """
            Actualiza las listas de métricas con los resultados del fold actual.
            
            :param y_true: Las etiquetas verdaderas del conjunto de validación.
            :param y_pred: Las etiquetas predichas por el modelo para el conjunto de validación.
            :param numero_ejecucion_modelo: El numero de vez de la seleccion del modelo.
            :param metricas: Un diccionario de listas donde se almacenan las métricas de evaluación.
            """
            if tarea == "clasificacion":
                if y_true is None or y_pred is None:
                    # Si no se pudieron calcular las predicciones
                    accuracy = 0.0
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                
                else:
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

                if metricas.get("accuracy_scores").get(numero_ejecucion_modelo) is None:
                    metricas["accuracy_scores"][numero_ejecucion_modelo] = []
                    metricas["precision_scores"][numero_ejecucion_modelo] = []
                    metricas["recall_scores"][numero_ejecucion_modelo] = []
                    metricas["f1_scores"][numero_ejecucion_modelo] = []

                metricas["accuracy_scores"][numero_ejecucion_modelo].append(accuracy)
                metricas["precision_scores"][numero_ejecucion_modelo].append(precision)
                metricas["recall_scores"][numero_ejecucion_modelo].append(recall)
                metricas["f1_scores"][numero_ejecucion_modelo].append(f1)

            else:
                if y_true is None or y_pred is None:
                    # Si no se pudieron calcular las predicciones
                    mae = 999.0
                    mse = 999.0
                    rmse = 999.0
                    r2 = -1.0
                    medae = 999.0
                    ev = -1.0

                else:
                    mae = mean_absolute_error(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_true, y_pred)
                    medae = median_absolute_error(y_true, y_pred)
                    ev = explained_variance_score(y_true, y_pred)

                if metricas.get("mae_scores").get(numero_ejecucion_modelo) is None:
                    metricas["mae_scores"][numero_ejecucion_modelo] = []
                    metricas["mse_scores"][numero_ejecucion_modelo] = []
                    metricas["rmse_scores"][numero_ejecucion_modelo] = []
                    metricas["r2_scores"][numero_ejecucion_modelo] = []
                    metricas["medae_scores"][numero_ejecucion_modelo] = []
                    metricas["ev_scores"][numero_ejecucion_modelo] = []

                metricas["mae_scores"][numero_ejecucion_modelo].append(mae)
                metricas["mse_scores"][numero_ejecucion_modelo].append(mse)
                metricas["rmse_scores"][numero_ejecucion_modelo].append(rmse)
                metricas["r2_scores"][numero_ejecucion_modelo].append(r2)
                metricas["medae_scores"][numero_ejecucion_modelo].append(medae)
                metricas["ev_scores"][numero_ejecucion_modelo].append(ev)

            return None

        def get_selector_modelo() -> SelectorModeloClasificacion | SelectorModeloRegresion:
            """
            Devuelve un selector de modelo de ML configurado según la tarea.
            
            :return: Un objeto SelectorModeloClasificacion si la tarea es de
             clasificación, o un objeto SelectorModeloRegresion si la tarea es
             de regresión, ambos configurados con la semilla
            :rtype: SelectorModeloClasificacion | SelectorModeloRegresion
            """
            selector_modelo = None
            if tarea == "clasificacion":
                selector_modelo = SelectorModeloClasificacion()

            else:
                selector_modelo = SelectorModeloRegresion()

            return selector_modelo
            
        def procesar_datos_pipeline_por_cada_fold(
            X_df: pd.DataFrame, 
            y_df: pd.Series,
            pipeline_aleatorio: dict[str, str | None],
        ) -> tuple[pd.DataFrame, pd.Series]:
            """
            Procesa los datos de entrada utilizando el pipeline de preprocesamiento configurado.
            
            :param X_df: DataFrame con las características del dataset.
            :param y_df: Serie con las etiquetas del dataset.
            :param pipeline_aleatorio: Un diccionario con las fases del pipeline aleatorio y sus algoritmos seleccionados.
            :param entrenamiento: Indica si el procesamiento es para la fase de entrenamiento o validación.
             Esto puede afectar cómo se aplican ciertas transformaciones (por ejemplo, tratamiento de outliers).
            :return: Una tupla con el DataFrame de características procesado y la Serie de etiquetas procesada.
            :rtype: tuple[pd.DataFrame, pd.Series]
            """
            kf = get_k_fold_coss_validation()
            folds_procesados = {}

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_df, y_df), 1):
                X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
                y_train, y_val = y_df.iloc[train_idx], y_df.iloc[val_idx]

                X_train_procesado = X_train.copy()
                y_train_procesado = y_train.copy()
                X_val_procesado = X_val.copy()
                y_val_procesado = y_val.copy()

                fases_instancias_fold_n = self._crear_fases_instancias()
                self._configurar_instancias(fases_instancias_fold_n, pipeline_aleatorio, tarea)

                for fase, instancia in fases_instancias_fold_n.items():
                    X_train_copy = X_train_procesado.copy()
                    y_train_copy = y_train_procesado.copy()
                    X_val_copy = X_val_procesado.copy()
                    y_val_copy = y_val_procesado.copy()

                    instancia.fit(X_train_copy, y_train_copy)
                    X_train_procesado, y_train_procesado = instancia.transform(X_train_copy, y_train_copy)
                    X_val_procesado, y_val_procesado = instancia.transform(X_val_copy, y_val_copy)

                folds_procesados[fold] = {
                    "X_train": X_train_procesado,
                    "y_train": y_train_procesado,
                    "X_val": X_val_procesado,
                    "y_val": y_val_procesado
                }

            return folds_procesados
        
        
        self._id_pipeline += 1

        fases_instancias_un_solo_uso = self._crear_fases_instancias()
        pipeline_aleatorio = self._generar_pipeline_aleatorio(fases_instancias_un_solo_uso)

        self._logger.info(
            f"id_pipeline: {self._id_pipeline}",
            extra={
                "fase": "Minero",
                "pipeline": pipeline_aleatorio
            }
        )

        metricas = get_diccionario_metricas_inicializadas()

        tiempo_inicio_pipeline_folds = time.perf_counter()
        folds_procesados = procesar_datos_pipeline_por_cada_fold(X_df, y_df, pipeline_aleatorio)
        tiempo_final_pipeline_folds = time.perf_counter()

        tiempo_total_pipeline_folds = tiempo_final_pipeline_folds - tiempo_inicio_pipeline_folds
        self._logger.info(
            f"Tiempo total de preprocesamiento en folds: {tiempo_total_pipeline_folds:.2f} segundos",
            extra={
                "fase": "Minero",
            }
        )

        selector_modelo = get_selector_modelo()
        algoritmos_disponibles = selector_modelo.ALGORITMOS

        #! Cambiar en produccion para que se ejecute N veces y no solo 1
        for numero_ejecucion_modelo in range(1):
            algoritmo_seleccionado = self._selector_aleatorio.choice(algoritmos_disponibles).item()
            selector_modelo.log_algoritmo = algoritmo_seleccionado

            for fold, datos in folds_procesados.items():
                tiempo_inicio_entrenamiento = time.perf_counter()
                selector_modelo.calcular_hiper_parametros(datos["X_train"], datos["y_train"])
                result_modelo = selector_modelo.entrenar_modelo(datos["X_train"], datos["y_train"])

                tiempo_final_entrenamiento = time.perf_counter()
                tiempo_total_entrenamiento = tiempo_final_entrenamiento - tiempo_inicio_entrenamiento

                if result_modelo.is_failure:
                    self._logger.error(
                        f"Error en entrenamiento del modelo en fold {fold}: {result_modelo.get_error()}",
                        extra={
                            "fase": "Minero",
                            "fold": fold
                        }
                    )
                    
                    actualizar_metricas_fold(None, None, numero_ejecucion_modelo, metricas)
                    break
                
                modelo_ml = result_modelo.get_value()
                predicciones = modelo_ml.predict(datos["X_val"])
                actualizar_metricas_fold(datos["y_val"], predicciones, numero_ejecucion_modelo, metricas)

        
        return pipeline_aleatorio, metricas, tiempo_total_pipeline_folds + tiempo_total_entrenamiento

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

    def imprimir_secuencia_preprocesamiento(self):
        secuencia = SecuenciaPreprocesamiento()
        secuencia.imprimir_secuencia()
        print("=" * 100)

    def _crear_fases_instancias(self) -> dict[str, object]:
        """
        Devuelve un diccionario con instancias de cada fase del pipeline.
        Estas instancias son nuevas y no tienen ninguna configuración previa.

        :return: Un diccionario donde las claves son los nombres de las fases
         del pipeline y los valores son instancias nuevas de cada fase.
        :rtype: dict[str, object]
        """
        return {
            "tratar_duplicados": TratarDuplicados(),
            "codificar_variables_binarias": CodificarVariablesBinarias(),
            "tratar_faltantes_numericos": TratarFaltantesNumericos(),
            "tratar_faltantes_strings": TratarFaltantesStrings(),
            "codificar_variables_categoricas_rango_bajo": CodificarVariablesCategoricasRangoBajo(),
            "codificar_variables_categoricas_rango_medio": CodificarVariablesCategoricasRangoMedio(),
            "codificar_variables_categoricas_rango_alto": CodificarVariablesCategoricasRangoAlto(),
            "tratar_outliers_numericos": TratarOutliersNumericos(),
            "escalar_datos_numericos": EscalarDatosNumericos(),
            "normalizar_datos_numericos": NormalizarDatosNumericos(),
            "crear_nueva_variable": CrearNuevaVariable(),
            "seleccionar_variables": SeleccionarVariables()
        }
    
    def _generar_pipeline_aleatorio(self, fases: dict[str, object]) -> dict[str, object]:
        """
        Genera un pipeline de preprocesamiento con las fases en un orden aleatorio.

        :param fases: Un diccionario con instancias de cada fase del pipeline.
        :return: Un diccionario con el algoritmo seleccionado para cada fase.
        :rtype: dict[str, object]
        """
        config = Configuracion()
        permitir_none = config.permitir_none
        permitir_llm = config.permitir_llm

        pipeline_aleatorio = {}
        for fase, instancia in fases.items():
            algoritmos_disponibles = instancia.ALGORITMOS

            while True:
                algoritmo_seleccionado = self._selector_aleatorio.choice(algoritmos_disponibles)

                if not permitir_none and algoritmo_seleccionado is None:
                    continue  # Reintentar si None no está permitido

                if not permitir_llm and algoritmo_seleccionado.lower() == "llm":
                    continue  # Reintentar si LLM no está permitido

                pipeline_aleatorio[fase] = algoritmo_seleccionado
                break
        
        return pipeline_aleatorio
    
    def _configurar_instancias(
            self, 
            fases_instancias: dict[str, object], 
            pipeline_aleatorio: dict[str, str | None],
            tarea: str
    ) -> None:
        """
        Configura cada instancia de fase con el algoritmo seleccionado en el pipeline aleatorio.

        :param fases_instancias: Un diccionario con instancias de cada fase del pipeline.
        :param pipeline_aleatorio: Un diccionario con el algoritmo seleccionado para cada fase.
        :param tarea: El tipo de tarea (clasificación, regresión, clustering) para configurar las fases que lo requieran.
        """
        for fase, algoritmo in pipeline_aleatorio.items():
            instancia = fases_instancias[fase]
            instancia.log_algoritmo = algoritmo

        fases_instancias["seleccionar_variables"].tarea = tarea
