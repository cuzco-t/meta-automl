import time

import numpy as np
import pandas as pd

from ..Result import Result

from ..preprocesamiento.TratarDuplicados import TratarDuplicados
from ..preprocesamiento.CodificarVariablesBinarias import CodificarVariablesBinarias
from ..preprocesamiento.TratarFaltantesNumericos import TratarFaltantesNumericos
from ..preprocesamiento.TratarFaltantesStrings import TratarFaltantesStrings
from ..preprocesamiento.CodificarVariablesCategoricasRangoBajo import CodificarVariablesCategoricasRangoBajo
from ..preprocesamiento.CodificarVariablesCategoricasRangoMedio import CodificarVariablesCategoricasRangoMedio
from ..preprocesamiento.CodificarVariablesCategoricasRangoAlto import CodificarVariablesCategoricasRangoAlto
from ..preprocesamiento.TratarOutliersNumericos import TratarOutliersNumericos
from ..preprocesamiento.EscalarDatosNumericos import EscalarDatosNumericos
from ..preprocesamiento.NormalizarDatosNumericos import NormalizarDatosNumericos
from ..preprocesamiento.CrearNuevaVariable import CrearNuevaVariable
from ..preprocesamiento.SeleccionarVariables import SeleccionarVariables

class EjecutorPreprocesamiento:
    def __init__(self, logger):
        self.logger = logger

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

    def configurar_instancias(self, instancias: dict, pipeline: dict, tarea: str):
        """Asigna algoritmo a cada instancia y configura tarea si es necesario."""
        for fase, algoritmo in pipeline.items():
            instancias[fase].log_algoritmo = algoritmo
        instancias["seleccionar_variables"].tarea = tarea
        instancias["crear_nueva_variable"].tarea = tarea

    def ejecutar_pipeline(
        self, 
        folds: dict[int, dict[str, pd.DataFrame | pd.Series]],
        pipeline: dict, 
        tarea: str,
        descripcion: str | None = None
    ) -> Result[dict[int, dict[str, pd.DataFrame | pd.Series]], str]:
        """Aplica el pipeline completo sobre los datos de cada fold (entrenamiento y transformación)."""
        folds_procesados = {}
        
        tiempo_inicio_preprocesamiento = time.time()
        for fold_num, fold_data in folds.items():
            X_train = fold_data["X_train"].copy()
            y_train = fold_data["y_train"].copy() if fold_data["y_train"] is not None else None
            X_val = fold_data["X_val"].copy()
            y_val = fold_data["y_val"].copy() if fold_data["y_val"] is not None else None
            
            instancias = self.crear_fases_instancias()
            self.configurar_instancias(instancias, pipeline, tarea)
            
            try:
                for fase, instancia in instancias.items():
                    if fase == "crear_nueva_variable" and descripcion:
                        instancia.descripcion = descripcion
                    instancia.fit(X_train, y_train)
                    X_train, y_train = instancia.transform(X_train, y_train)
                    X_val, y_val = instancia.transform(X_val, y_val)
                    self.logger.info(f"Fold {fold_num}: Fase '{fase}', algoritmo '{instancia.log_algoritmo}' OK.")
                
                folds_procesados[fold_num] = {
                    "X_train": X_train.copy(),
                    "y_train": y_train.copy() if y_train is not None else None,
                    "X_val": X_val.copy(),
                    "y_val": y_val.copy() if y_val is not None else None
                }
            except Exception as e:
                self.logger.error(f"Pipeline mal configurado en fold {fold_num}: {e}")
                return Result.fail({
                    "error": str(e),
                    "pipeline": pipeline,
                    "fase": fase
                }), None 
        
        tiempo_fin_preprocesamiento = time.time()
        tiempo_total_preprocesamiento = tiempo_fin_preprocesamiento - tiempo_inicio_preprocesamiento
        
        return Result.ok(folds_procesados), tiempo_total_preprocesamiento
        
    def ejecutar_pipeline_clustering(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        pipeline: dict | None = None,
        descripcion: str | None = None
    ) -> Result[dict[str, pd.DataFrame | pd.Series], str]:
        """Aplica el pipeline completo sobre los datos para clustering."""
        tiempo_inicio_preprocesamiento = time.time()
        
        X_proc = X.copy()
        y_proc = y.copy() if y is not None else None
        
        instancias = self.crear_fases_instancias()
        self.configurar_instancias(instancias, pipeline or {}, "clustering")
        
        try:
            for fase, instancia in instancias.items():
                if fase == "crear_nueva_variable" and descripcion:
                    instancia.descripcion = descripcion
                instancia.fit(X_proc, y_proc)
                X_proc, y_proc = instancia.transform(X_proc, y_proc)
            
            tiempo_fin_preprocesamiento = time.time()
            tiempo_total = tiempo_fin_preprocesamiento - tiempo_inicio_preprocesamiento
            
            return Result.ok({"X_proc": X_proc, "y_proc": y_proc}), tiempo_total
            
        except Exception as e:
            self.logger.error(f"Error en pipeline clustering: {e}")
            return Result.fail({
                "error": str(e),
                "pipeline": pipeline,
                "fase": fase
            }), None
