import numpy as np

from typing import Dict, List, Any, Tuple


class VectorizadorPipeline:
    """
    Construye representaciones one-hot de los pasos de un pipeline.
    El mapa de índices se crea una vez y se reutiliza.
    """

    def __init__(self, fases_instancias: Dict[str, object], modelos_por_tarea: Dict[str, List[str]]):
        """
        Args:
            fases_instancias: Diccionario con nombre_fase -> instancia (debe tener atributo ALGORITMOS).
            modelos_por_tarea: Diccionario con tarea -> lista de nombres de algoritmos ML.
        """
        self.fases = fases_instancias
        self.modelos_por_tarea = modelos_por_tarea
        self.mapa_indices, self.dimensiones = self._construir_mapa()

    def _construir_mapa(self) -> Tuple[Dict[str, Dict[Any, int]], int]:
        """Construye un índice numérico para cada posible valor (algoritmo) de cada fase y cada modelo."""
        contador = 0
        mapa = {}

        # Fases de preprocesamiento
        for fase, instancia in self.fases.items():
            mapa[fase] = {}
            for alg in instancia.ALGORITMOS:
                mapa[fase][alg] = contador
                contador += 1

        # Modelos ML por tarea
        for tarea, modelos in self.modelos_por_tarea.items():
            clave = f"modelos_{tarea}"
            mapa[clave] = {}
            for alg in modelos:
                mapa[clave][alg] = contador
                contador += 1

        return mapa, contador

    def vectorizar_pipeline(self, tarea: str, pipeline: Dict[str, str], modelo: str) -> List[List[float]]:
        """
        Genera la secuencia de vectores de estado para un pipeline dado.

        Args:
            tarea: 'clasificacion', 'regresion' o 'clustering'.
            pipeline: Diccionario fase -> algoritmo.
            modelo: Nombre del algoritmo ML final.

        Returns:
            Lista de vectores (listas de float) representando el estado después de cada paso.
        """
        vector = [0.0] * self.dimensiones
        historia = []

        # Activar fases una por una
        for fase, alg in pipeline.items():
            idx = self.mapa_indices[fase][alg]
            vector[idx] = 1.0
            historia.append(vector.copy())

        if modelo == "modelo_no_ejecutado":
            return historia

        # Activar modelo final
        idx_modelo = self.mapa_indices[f"modelos_{tarea}"][modelo]
        vector[idx_modelo] = 1.0
        historia.append(vector.copy())

        return historia