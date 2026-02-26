import time
import pandas as pd

from typing import Dict, List
from dataclasses import dataclass

from ..Result import Result

from ..cash.SelectorModeloClasificacion import SelectorModeloClasificacion
from ..cash.SelectorModeloRegresion import SelectorModeloRegresion
from ..cash.SelectorModeloClustering import SelectorModeloClustering


class Entrenador:
    """Clase para entrenar modelos de machine learning"""
    
    def entrenar(
        self,
        folds_data: Dict[int, Dict[str, pd.DataFrame | pd.Series]],
        nombres_modelos: List[str],
        tarea: str
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
        
        for nombre_modelo in nombres_modelos:
            # Obtener selector según la tarea
            selector = self._obtener_selector(tarea)
            selector.log_algoritmo = nombre_modelo
            
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
            
            # Calcular hiperparámetros (una sola vez por modelo)
            selector.calcular_hiper_parametros(X, y)
            hiperparametros = selector.log_params
            
            # Iterar sobre cada fold y entrenar
            modelos_folds = []
            tiempos_folds = []
            for num_fold, fold_data in folds_data.items():
                # Crear un nuevo selector para cada fold
                selector_fold = self._obtener_selector(tarea)
                selector_fold.log_algoritmo = nombre_modelo
                selector_fold.log_params = hiperparametros
                
                X_train = fold_data['X_train']
                y_train = fold_data['y_train']
                
                # Medir tiempo de entrenamiento
                tiempo_inicio = time.time()
                result_modelo_entrenado = selector_fold.entrenar_modelo(X_train, y_train)
                tiempo_fin = time.time()
                
                tiempo_entrenamiento = tiempo_fin - tiempo_inicio
                modelos_folds.append(result_modelo_entrenado)
                tiempos_folds.append(tiempo_entrenamiento)
            
            modelos_entrenados_results.append(modelos_folds)
            # Calcular promedio de tiempos para este modelo
            tiempo_promedio = sum(tiempos_folds) / len(tiempos_folds)
            tiempos_promedio.append(tiempo_promedio)
        
        return modelos_entrenados_results, tiempos_promedio
    
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