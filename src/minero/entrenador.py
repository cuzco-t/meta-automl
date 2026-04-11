import time
import pandas as pd

from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass

from ..Result import Result
from ..ExtractorMetaFeatures import ExtractorMetaFeatures

from ..cash.SelectorModeloClasificacion import SelectorModeloClasificacion
from ..cash.SelectorModeloRegresion import SelectorModeloRegresion
from ..cash.SelectorModeloClustering import SelectorModeloClustering

print_original = print

def print(*args, **kwargs):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_original(f"{ahora} |", *args, **kwargs)

class Entrenador:
    """Clase para entrenar modelos de machine learning"""
    
    def entrenar(
        self,
        folds_data: Dict[int, Dict[str, pd.DataFrame | pd.Series]],
        nombres_modelos: List[str],
        tarea: str,
        llm_seleccionado: str | None = None
    ) -> tuple[List[List[Result]], List[float]]:
        """
        Entrena múltiples modelos sobre folds de datos.
        
        Args:
            folds_data: Diccionario con folds. Estructura: {fold_num: {'X_train': df, 'X_val': df, 'y_train': series, 'y_val': series}}
            nombres_modelos: Lista con nombres de los modelos a entrenar
            tarea: Tipo de tarea ('clasificacion', 'regresion', 'clustering')
        
        Returns:
            Tupla con:
                - Lista de objetos Result con los modelos entrenados
                - Lista con tiempos promedio de entrenamiento por modelo
        """
        
        modelos_entrenados_results = []
        tiempos_promedio = []
        
        # Juntar datos del primer fold
        primer_fold = folds_data[list(folds_data.keys())[0]]
        X = pd.concat([primer_fold['X_train'], primer_fold['X_val']], ignore_index=True)
        # Concatenar y, si existe
        if primer_fold['y_train'] is not None and primer_fold['y_val'] is not None:
            y = pd.concat([
                pd.Series(primer_fold['y_train']),
                pd.Series(primer_fold['y_val'])
            ], ignore_index=True)
        elif primer_fold['y_train'] is not None:
            y = pd.Series(primer_fold['y_train']).reset_index(drop=True)
        elif primer_fold['y_val'] is not None:
            y = pd.Series(primer_fold['y_val']).reset_index(drop=True)
        else:
            y = None

        extractor = ExtractorMetaFeatures()
        meta_features_globales_totales, _ = extractor.extraer_desde_dataframe(X.copy(), y.copy())
        meta_features_globales_limpias = extractor.eliminar_constantes_errores(meta_features_globales_totales)
        meta_features_globales_formateadas = extractor.formatear_meta_features_globales(meta_features_globales_limpias)

        for num_modelo, nombre_modelo in enumerate(nombres_modelos, 1):
            # Obtener selector según la tarea
            selector = self._obtener_selector(tarea)
            selector.log_algoritmo = nombre_modelo
            selector.llm_seleccionado = llm_seleccionado
            
            
            
            # Calcular hiperparámetros (una sola vez por modelo)
            try:
                selector.calcular_hiper_parametros(X, y, meta_features_globales_formateadas)
            
            except Exception as e:
                print(f"ERROR - al calcular hiperparámetros para el modelo '{nombre_modelo}'")
                print(f"Exception: {e}")
                modelos_entrenados_results.append([
                    Result.fail(f"Error al calcular hiperparámetros: {e}"),
                    Result.fail(f"Error al calcular hiperparámetros: {e}"),
                    Result.fail(f"Error al calcular hiperparámetros: {e}")
                ])
                tiempos_promedio.append(0.0)
                continue
            
            # Iterar sobre cada fold y entrenar
            modelos_folds = []
            tiempos_folds = []

            print(f"({num_modelo}/{len(nombres_modelos)}) Entrenando modelo '{nombre_modelo}'")

            for num_fold, fold_data in folds_data.items():
                X_train = fold_data['X_train']
                y_train = fold_data['y_train']
                
                print(f"({num_fold}/{len(folds_data)}) Iniciando fold")

                # Medir tiempo de entrenamiento
                tiempo_inicio = time.time()
                result_modelo_entrenado = selector.entrenar_modelo(X_train, y_train)
                tiempo_fin = time.time()
                
                tiempo_entrenamiento = tiempo_fin - tiempo_inicio
                print(f"({num_fold}/{len(folds_data)}) Fold terminado, tiempo: {tiempo_entrenamiento:.2f} segundos")
                
                if result_modelo_entrenado.is_failure:
                    while len(modelos_folds) < 3:
                        modelos_folds.append(Result.fail(result_modelo_entrenado.get_error()))
                        tiempos_folds.append(tiempo_entrenamiento)
                    break

                modelos_folds.append(result_modelo_entrenado)
                tiempos_folds.append(tiempo_entrenamiento)
            
            print(f"({num_modelo}/{len(nombres_modelos)}) Modelo entrenado '{nombre_modelo}'")
            print("")

            modelos_entrenados_results.append(modelos_folds)
            # Calcular promedio de tiempos para este modelo
            tiempo_promedio = sum(tiempos_folds) / len(tiempos_folds)
            tiempos_promedio.append(tiempo_promedio)
        
        return modelos_entrenados_results, tiempos_promedio
    
    def entrenar_clustering(
        self,
        X: pd.DataFrame,
        nombres_modelos: List[str],
        llm_seleccionado: str | None = None
    ) -> tuple[List[Result], List[float]]:
        """
        Entrena múltiples modelos de clustering sobre datos.
        
        Args:
            X: DataFrame con los datos de entrada
            nombres_modelos: Lista con nombres de los modelos a entrenar
        
        Returns:
            Tupla con:
                - Lista de objetos Result con las etiquetas predichas por cada modelo
                - Lista con tiempos de entrenamiento por modelo
        """
        modelos_entrenados_results = []
        tiempos_entrenamiento = []
        
        extractor = ExtractorMetaFeatures()
        meta_features_globales_totales, _ = extractor.extraer_desde_dataframe(X.copy(), None)
        meta_features_globales_limpias = extractor.eliminar_constantes_errores(meta_features_globales_totales)
        meta_features_globales_formateadas = extractor.formatear_meta_features_globales(meta_features_globales_limpias)

        for nombre_modelo in nombres_modelos:
            # Obtener selector para clustering
            selector = self._obtener_selector('clustering')
            selector.log_algoritmo = nombre_modelo
            selector.llm_seleccionado = llm_seleccionado
            
            # Calcular hiperparámetros
            try:
                selector.calcular_hiper_parametros(X, meta_features_globales_formateadas)
            except Exception as e:
                print(f"Error al calcular hiperparámetros para el modelo '{nombre_modelo}': {e}")
                modelos_entrenados_results.append(
                    Result.fail(f"{e}")
                )
                tiempos_entrenamiento.append(0.0)
                continue

            # Medir tiempo de entrenamiento
            tiempo_inicio = time.time()
            result_etiquetas = selector.entrenar_modelo(X)
            tiempo_fin = time.time()
            
            tiempo_entrenamiento = tiempo_fin - tiempo_inicio
            modelos_entrenados_results.append(result_etiquetas)
            tiempos_entrenamiento.append(tiempo_entrenamiento)
        
        return modelos_entrenados_results, tiempos_entrenamiento
    
    def _obtener_selector(self, tarea: str):
        """
        Obtiene el selector de modelo según la tarea.
        
        Args:
            tarea: Tipo de tarea ('clasificacion', 'regresion', 'clustering')
        
        Returns:
            Instancia del selector correspondiente
        """
        if tarea.lower() == 'clasificacion':
            return SelectorModeloClasificacion()
        elif tarea.lower() == 'regresion':
            return SelectorModeloRegresion()
        elif tarea.lower() == 'clustering':
            return SelectorModeloClustering()
        else:
            raise ValueError(f"Tarea no válida: {tarea}")