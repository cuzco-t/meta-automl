import time

import pandas as pd
import numpy as np

from src.config.Configuracion import Configuracion
from .PipelineLogger import PipelineLogger
from .Result import Result
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
    
    def pipeline_supervisado(
        self, 
        X_df: pd.DataFrame, 
        y_df: pd.Series, 
        tarea: str,
        descripcion: str | None = None
    ) -> tuple[dict[str, str | None], dict[str, dict], list[float]]:

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
                    # Si no se pudieron calcular las predicciones, se castiga
                    metricas["accuracy_scores"][numero_ejecucion_modelo] = [-1.0] * self._N_FOLDS
                    metricas["precision_scores"][numero_ejecucion_modelo] = [-1.0] * self._N_FOLDS
                    metricas["recall_scores"][numero_ejecucion_modelo] = [-1.0] * self._N_FOLDS
                    metricas["f1_scores"][numero_ejecucion_modelo] = [-1.0] * self._N_FOLDS
                    return None
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

                valores_metricas = {
                    "accuracy_scores": accuracy,
                    "precision_scores": precision,
                    "recall_scores": recall,
                    "f1_scores": f1,
                }

                for key, value in valores_metricas.items():
                    metricas[key].setdefault(numero_ejecucion_modelo, []).append(value)

            else:
                if y_true is None or y_pred is None:
                    # Si no se pudieron calcular las predicciones, se castiga
                    metricas["mae_scores"][numero_ejecucion_modelo] = [999.0] * self._N_FOLDS
                    metricas["mse_scores"][numero_ejecucion_modelo] = [999.0] * self._N_FOLDS
                    metricas["rmse_scores"][numero_ejecucion_modelo] = [999.0] * self._N_FOLDS
                    metricas["r2_scores"][numero_ejecucion_modelo] = [-1.0] * self._N_FOLDS
                    metricas["medae_scores"][numero_ejecucion_modelo] = [999.0] * self._N_FOLDS
                    metricas["ev_scores"][numero_ejecucion_modelo] = [-1.0] * self._N_FOLDS
                    return None

                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                medae = median_absolute_error(y_true, y_pred)
                ev = explained_variance_score(y_true, y_pred)

                valores_metricas = {
                    "mae_scores": mae,
                    "mse_scores": mse,
                    "rmse_scores": rmse,
                    "r2_scores": r2,
                    "medae_scores": medae,
                    "ev_scores": ev,
                }

                for key, value in valores_metricas.items():
                    metricas[key].setdefault(numero_ejecucion_modelo, []).append(value)

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
        ) -> Result[dict[int, dict[str, pd.DataFrame | pd.Series]], str]:
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

            if tarea == "clasificacion" and pd.api.types.is_numeric_dtype(y_df):
                # y_df = y_df.astype("object")
                unique_vals = np.unique(y_df)
                val_to_int = {v: i for i, v in enumerate(unique_vals)}
                y_discrete = np.array([val_to_int[v] for v in y_df])
                y_df = pd.Series(y_discrete, index=y_df.index)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_df, y_df), 1):
                X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
                y_train, y_val = y_df.iloc[train_idx], y_df.iloc[val_idx]

                X_train_procesado, y_train_procesado = X_train.copy(), y_train.copy()
                X_val_procesado, y_val_procesado = X_val.copy(), y_val.copy()

                fases_instancias_fold_n = self.crear_fases_instancias()
                self._configurar_instancias(fases_instancias_fold_n, pipeline_aleatorio, tarea)

                for fase, instancia in fases_instancias_fold_n.items():
                    X_train_copy, y_train_copy = X_train_procesado.copy(), y_train_procesado.copy()
                    X_val_copy, y_val_copy = X_val_procesado.copy(), y_val_procesado.copy()

                    # instancia.fit(X_train_copy, y_train_copy)
                    # X_train_procesado, y_train_procesado = instancia.transform(X_train_copy, y_train_copy)
                    # X_val_procesado, y_val_procesado = instancia.transform(X_val_copy, y_val_copy)

                    try:
                        if fase == "crear_nueva_variable":
                            instancia.tarea = tarea
                            instancia.descripcion = descripcion

                        instancia.fit(X_train_copy, y_train_copy)
                        X_train_procesado, y_train_procesado = instancia.transform(X_train_copy, y_train_copy)
                        X_val_procesado, y_val_procesado = instancia.transform(X_val_copy, y_val_copy)

                    except ValueError as e:
                        if fase == "seleccionar_variables" and "contains NaN" in str(e):
                            self._logger.error(
                                f"Pipeline mal configurado",
                                extra={
                                    "fase": fase,
                                    "funcion": "Minero - procesar_datos_pipeline_por_cada_fold",
                                    "clase_error": e.__class__.__name__,
                                    # "mensaje_error": str(e)
                                }
                            )

                            return Result.fail('Pipeline mal configurado')
                
                folds_procesados[fold] = {
                    "X_train": X_train_procesado,
                    "y_train": y_train_procesado,
                    "X_val": X_val_procesado,
                    "y_val": y_val_procesado
                }

            return Result.ok(folds_procesados)
        
        # ============================================================================================

        self._id_pipeline += 1
        logger = PipelineLogger().get_logger()

        fases_instancias_un_solo_uso = self.crear_fases_instancias()
        pipeline_aleatorio = self._generar_pipeline_aleatorio(fases_instancias_un_solo_uso)

        logger.info(
            "Pipeline generado aleatoriamente",
            extra={
                "pipeline": pipeline_aleatorio
            }
        )

        metricas = get_diccionario_metricas_inicializadas()

        logger.info(
            "Iniciando preprocesamiento en folds",
            extra={
                "pipeline": pipeline_aleatorio
            }
        )

        tiempo_inicio_pipeline_folds = time.perf_counter()
        result_folds_procesados = procesar_datos_pipeline_por_cada_fold(X_df, y_df, pipeline_aleatorio)
        tiempo_final_pipeline_folds = time.perf_counter()

        if result_folds_procesados.is_failure:
            return {
                "pipeline": pipeline_aleatorio,
                "metricas": None,
                "lista_modelos_ml": [
                    str(self._selector_aleatorio.choice(get_selector_modelo().ALGORITMOS))
                    for _ in range(10)
                ],
                "tiempos_pipeline_modelos": None
            }

        folds_procesados = result_folds_procesados.get_value()

        tiempo_total_pipeline_folds = tiempo_final_pipeline_folds - tiempo_inicio_pipeline_folds
        self._logger.info(
            f"Tiempo total de preprocesamiento en folds: {tiempo_total_pipeline_folds:.2f} segundos",
            extra={
                "fase": "Minero",
            }
        )

        selector_modelo = get_selector_modelo()
        algoritmos_disponibles = selector_modelo.ALGORITMOS

        tiempo_total_ejecuciones_modelos = []
        lista_modelos_ml = []
        #! Cambiar en produccion para que se ejecute N veces y no solo 1
        for numero_ejecucion_modelo in range(10):
            algoritmo_seleccionado = str(self._selector_aleatorio.choice(algoritmos_disponibles))
            selector_modelo.log_algoritmo = algoritmo_seleccionado
            lista_modelos_ml.append(algoritmo_seleccionado)

            #? Meta-features globales
            X_global = pd.concat([folds_procesados[1]["X_train"], folds_procesados[1]["X_val"]])
            y_global = pd.concat([folds_procesados[1]["y_train"], folds_procesados[1]["y_val"]])
            selector_modelo.calcular_hiper_parametros(X_global, y_global)


            tiempo_inicio_entrenamiento = time.perf_counter()
            for fold, datos in folds_procesados.items():
                result_modelo = selector_modelo.entrenar_modelo(datos["X_train"], datos["y_train"])

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
                try:
                    predicciones = modelo_ml.predict(datos["X_val"])
                except ValueError as e:
                    if "contains NaN" in str(e):
                        self._logger.error(
                            "Valores NaN en predicciones del modelo",
                            extra={
                                "fase": "Prediccion del modelo",
                                "modelo": selector_modelo.log_algoritmo,
                                "fold": fold,
                                "funcion": "Minero - pipeline_supervisado",
                            }
                        )
                        actualizar_metricas_fold(None, None, numero_ejecucion_modelo, metricas)
                        break

                    elif "The feature names should match those that were passed during fit" in str(e):
                        self._logger.error(
                            "Inconsistencia entre columnas de entrenamiento y validacion",
                            extra={
                                "fase": "Prediccion del modelo",
                                "modelo": selector_modelo.log_algoritmo,
                                "fold": fold,
                                "funcion": "Minero - pipeline_supervisado",
                            }
                        )
                        actualizar_metricas_fold(None, None, numero_ejecucion_modelo, metricas)
                        break

                actualizar_metricas_fold(datos["y_val"], predicciones, numero_ejecucion_modelo, metricas)

            tiempo_final_entrenamiento = time.perf_counter()
            tiempo_total_entrenamiento = tiempo_final_entrenamiento - tiempo_inicio_entrenamiento

            tiempo_total_ejecuciones_modelos.append(tiempo_total_entrenamiento)

        tiempos_totales_pipeline_modelos = np.array(tiempo_total_ejecuciones_modelos) + tiempo_total_pipeline_folds
        tiempos_totales_pipeline_modelos = tiempos_totales_pipeline_modelos.tolist()

        metricas_promediadas = {
            key: [float(np.mean(scores)) for num, scores in value.items()]
            for key, value in metricas.items()
        }

        datos_pipeline = {
            "pipeline": pipeline_aleatorio,
            "metricas": metricas_promediadas,
            "lista_modelos_ml": lista_modelos_ml,
            "tiempos_pipeline_modelos": tiempos_totales_pipeline_modelos
        }

        return datos_pipeline

    def crear_fases_instancias(self) -> dict[str, object]:
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

                if fase == "crear_nueva_variable":
                    print("")

                if not permitir_none and algoritmo_seleccionado is None:
                    continue  # Reintentar si None no está permitido
                
                algoritmo_seleccionado = str(algoritmo_seleccionado) if algoritmo_seleccionado is not None else None

                if not permitir_llm and algoritmo_seleccionado == "llm":
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

    def crear_selectores_modelos(self):
        return {
            "selector_modelo_clasificacion": SelectorModeloClasificacion(),
            "selector_modelo_regresion": SelectorModeloRegresion(),
            "selector_modelo_clustering": SelectorModeloClustering()
        }
