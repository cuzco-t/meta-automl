import time
import multiprocessing
import pandas as pd
import numpy as np

from src.Result import Result
from src.ExtractorMetaFeatures import ExtractorMetaFeatures
from src.cash.SelectorModeloClasificacion import SelectorModeloClasificacion

from src.minero.evaluador_modelos import EvaluadorModelos
from src.preprocesamiento.TratarDuplicados import TratarDuplicados
from src.preprocesamiento.CodificarVariablesBinarias import CodificarVariablesBinarias
from src.preprocesamiento.TratarFaltantesNumericos import TratarFaltantesNumericos
from src.preprocesamiento.TratarFaltantesStrings import TratarFaltantesStrings
from src.preprocesamiento.CodificarVariablesCategoricasRangoBajo import CodificarVariablesCategoricasRangoBajo
from src.preprocesamiento.CodificarVariablesCategoricasRangoMedio import CodificarVariablesCategoricasRangoMedio
from src.preprocesamiento.CodificarVariablesCategoricasRangoAlto import CodificarVariablesCategoricasRangoAlto
from src.preprocesamiento.TratarOutliersNumericos import TratarOutliersNumericos
from src.preprocesamiento.EscalarDatosNumericos import EscalarDatosNumericos
from src.preprocesamiento.NormalizarDatosNumericos import NormalizarDatosNumericos
from src.preprocesamiento.CrearNuevaVariable import CrearNuevaVariable
from src.preprocesamiento.SeleccionarVariables import SeleccionarVariables

TIME_LIMIT_SEGUNDOS = 30 * 60  # 30 minutos


class PipelineQTable:
    
    def crear_fases_instancias(self) -> dict[str, object]:
        """Retorna nuevas instancias de cada fase."""
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

    def ejecutar_pipeline_supervisado(
        self,
        folds: dict[int, dict[str, pd.DataFrame | pd.Series]],
        tarea: str,
        descripcion: str | None = None,
        guia_estado_acciones: dict[str, list[str]] = None,
    ) -> dict | None:
        """
        Aplica el pipeline supervisado usando Q-Table con consistencia entre folds
        y timeout global de 20 minutos.

        Args:
            folds: diccionario de folds con X_train, y_train, X_val, y_val.
            tarea: "clasificación" o "regresión".
            descripcion: descripción del dataset (para crear_nueva_variable y seleccionar_variables).
            guia_estado_acciones: diccionario estado -> lista de acciones ordenadas por Q.

        Returns:
            Diccionario con métricas de validación, o None si falla o excede el timeout.
        """
        # Lanzar la ejecución en un proceso aparte con timeout
        queue = multiprocessing.Queue()
        proceso = multiprocessing.Process(
            target=self._ejecutar_pipeline_supervisado_interno,
            args=(queue, folds, tarea, descripcion, guia_estado_acciones)
        )
        proceso.start()
        proceso.join(timeout=TIME_LIMIT_SEGUNDOS)

        if proceso.is_alive():
            proceso.terminate()
            proceso.join()
            print("[TIMEOUT] La ejecución del pipeline supervisado excedió los 20 minutos.")
            return None

        if queue.empty():
            return None

        resultado = queue.get()
        return resultado

    def _ejecutar_pipeline_supervisado_interno(
        self,
        queue: multiprocessing.Queue,
        folds: dict[int, dict[str, pd.DataFrame | pd.Series]],
        tarea: str,
        descripcion: str | None = None,
        guia_estado_acciones: dict[str, list[str]] = None,
    ):
        """Ejecuta la lógica completa del pipeline (sin timeout externo)."""
        try:
            resultado = self._preprocesar_y_evaluar(folds, tarea, descripcion, guia_estado_acciones)
            queue.put(resultado)
        except Exception as e:
            print(f"[ERROR] Excepción en pipeline supervisado: {e}")
            queue.put(None)

    def _preprocesar_y_evaluar(
        self,
        folds: dict[int, dict[str, pd.DataFrame | pd.Series]],
        tarea: str,
        descripcion: str | None = None,
        guia_estado_acciones: dict[str, list[str]] = None,
    ) -> dict | None:
        """Preprocesa con consistencia entre folds, luego entrena y evalúa."""

        # ------------------------------------------------------------------
        # PREPROCESAMIENTO CONSISTENTE ENTRE FOLDS
        # ------------------------------------------------------------------
        estados = list(guia_estado_acciones.keys())
        # Lista de listas de acciones para cada estado
        acciones_por_estado = [guia_estado_acciones[est][:] for est in estados]
        num_fases = 12  # 12 fases de preprocesamiento
        num_folds = len(folds)

        # Indices de acción actuales para cada estado (fase)
        indices_actuales = [0] * num_fases

        folds_procesados = None

        while True:
            # Intentar procesar todos los folds con los índices actuales
            exito = True
            folds_procesados_temp = {}

            for fold_num, fold_data in folds.items():
                X_train = fold_data["X_train"].copy()
                y_train = fold_data["y_train"].copy() if fold_data["y_train"] is not None else None
                X_val = fold_data["X_val"].copy()
                y_val = fold_data["y_val"].copy() if fold_data["y_val"] is not None else None

                fases_instancias = self.crear_fases_instancias()

                for i_fase in range(num_fases):
                    fase_actual = list(fases_instancias.keys())[i_fase]
                    idx_accion = indices_actuales[i_fase]
                    acciones_de_estado = acciones_por_estado[i_fase]

                    if idx_accion >= len(acciones_de_estado):
                        # Ya no quedan acciones para esta fase -> pipeline roto
                        print(f"[FATAL] No quedan acciones para la fase {fase_actual} (índice {idx_accion})")
                        return None

                    accion_str = acciones_de_estado[idx_accion]
                    accion_limpia = accion_str.replace(f"{fase_actual}_", "")
                    instancia = fases_instancias[fase_actual]
                    instancia.log_algoritmo = accion_limpia if accion_limpia != 'None' else None

                    try:
                        if fase_actual == "crear_nueva_variable" and descripcion:
                            instancia.descripcion = descripcion
                        elif fase_actual == "seleccionar_variables" and descripcion:
                            instancia.descripcion = descripcion

                        instancia.fit(X_train, y_train)
                        X_train, y_train = instancia.transform(X_train, y_train)
                        X_val, y_val = instancia.transform(X_val, y_val)

                        print(f"OK   - Fold: {fold_num:<3} | Fase: {fase_actual:<45} | Algoritmo: {instancia.log_algoritmo}")
                    
                    except Exception as e:
                        print(f"ERROR - Fold: {fold_num:<3} | Fase: {fase_actual:<45} | Algoritmo: {instancia.log_algoritmo}")
                        print(f"Exception: {e}")

                        # Falló la acción actual para este fold.
                        # Incrementar el índice de esta fase y reintentar desde fold 0
                        indices_actuales[i_fase] += 1
                        exito = False
                        break  # Sale del bucle interno de fases

                if not exito:
                    break  # Sale del bucle de folds para reiniciar

                # Si llegamos aquí, todas las fases para este fold funcionaron
                folds_procesados_temp[fold_num] = {
                    "X_train": X_train.copy(),
                    "y_train": y_train.copy() if y_train is not None else None,
                    "X_val": X_val.copy(),
                    "y_val": y_val.copy() if y_val is not None else None
                }

            if exito:
                # Todos los folds se procesaron correctamente
                folds_procesados = folds_procesados_temp
                break

            # Si no hubo éxito, ya se incrementó el índice en la fase que falló.
            # Se reinicia el bucle while para intentar de nuevo con los nuevos índices.
            print(f"[REINTENTO] Nuevos índices: {indices_actuales}")

        # ------------------------------------------------------------------
        # FIN PREPROCESAMIENTO CONSISTENTE
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # FASE 12 - SELECCION DE LLM
        # ------------------------------------------------------------------
        # El estado 12 (si existe) corresponde al LLM
        if len(estados) > 12:
            llm_estado = estados[12]
            acciones_llm = guia_estado_acciones[llm_estado]
            llm_seleccionado = acciones_llm[0] if acciones_llm else "ninguno"
            llm_seleccionado = llm_seleccionado if llm_seleccionado != "ninguno" else None
        else:
            llm_seleccionado = None
        print(f"LLM seleccionado: {llm_seleccionado}")

        # ------------------------------------------------------------------
        # FASE 13 - SELECCION DE MODELO
        # ------------------------------------------------------------------
        if tarea == "clasificación":
            # Extraer número de cluster del primer estado (ej: "cluster3_fase0")
            numero_cluster = estados[0].split("_")[0].replace("cluster", "")
            estado_modelo = f"cluster{numero_cluster}_fase13"
            acciones_modelo = guia_estado_acciones.get(estado_modelo, [])

            for accion_modelo in acciones_modelo:
                accion_limpia = accion_modelo.replace("clasificacion_", "")
                selector = SelectorModeloClasificacion()
                selector.log_algoritmo = accion_limpia
                selector.llm_seleccionado = llm_seleccionado

                # Calcular hiperparámetros con datos del primer fold
                primer_fold = folds_procesados[list(folds_procesados.keys())[0]]
                X_train_primer = primer_fold["X_train"]
                y_train_primer = primer_fold["y_train"]

                # Calcular meta-features globales
                X_total = pd.concat([primer_fold['X_train'], primer_fold['X_val']], ignore_index=True)
                y_total = None
                if primer_fold['y_train'] is not None and primer_fold['y_val'] is not None:
                    y_total = pd.concat([
                        pd.Series(primer_fold['y_train']),
                        pd.Series(primer_fold['y_val'])
                    ], ignore_index=True)
                elif primer_fold['y_train'] is not None:
                    y_total = pd.Series(primer_fold['y_train']).reset_index(drop=True)
                elif primer_fold['y_val'] is not None:
                    y_total = pd.Series(primer_fold['y_val']).reset_index(drop=True)

                extractor = ExtractorMetaFeatures()
                meta_features_globales, _ = extractor.extraer_desde_dataframe(X_total, y_total)
                meta_features_globales = extractor.eliminar_constantes_errores(meta_features_globales)
                meta_features_globales_formateadas = extractor.formatear_meta_features_globales(meta_features_globales)

                try:
                    selector.calcular_hiper_parametros(X_train_primer, y_train_primer, meta_features_globales_formateadas)
                except Exception as e:
                    print(f"ERROR - al calcular hiperparámetros para el modelo '{accion_limpia}'")
                    print(f"Exception: {e}")
                    continue

                # Entrenar modelo en cada fold
                modelos_folds = []
                for fold_num, fold_data in folds_procesados.items():
                    X_train = fold_data['X_train']
                    y_train = fold_data['y_train']
                    result_modelo = selector.entrenar_modelo(X_train, y_train)
                    if result_modelo.is_failure:
                        modelos_folds.append(Result.fail(result_modelo.get_error()))
                        break
                    modelos_folds.append(result_modelo)

                # Si algún fold falló, probar siguiente modelo
                if any(m.is_failure for m in modelos_folds):
                    print(f"[MODELO FALLÓ] {accion_limpia}, probando siguiente...")
                    continue

                # Evaluar
                evaluador = EvaluadorModelos()
                metricas = evaluador.evaluar_modelos([modelos_folds], folds_procesados, tarea)
                print("CLASIFICACIÓN FINALIZADA")
                return metricas

            # Si se agotaron todos los modelos
            print("[ERROR] No quedan modelos de clasificación disponibles")
            return None

        elif tarea == "regresión":
            # TODO: implementar regresión
            return None

        else:
            return None
