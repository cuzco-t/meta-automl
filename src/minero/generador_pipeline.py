import numpy as np


class GeneradorPipeline:
    def __init__(self, semilla: int, permitir_none: bool, permitir_llm: bool):
        self.rng = np.random.default_rng(semilla)
        self.permitir_none = permitir_none
        self.permitir_llm = permitir_llm

    def generar_pipeline_aleatorio(self, fases: dict[str, object]) -> dict[str, str | None]:
        """Retorna un dict {nombre_fase: algoritmo_seleccionado}."""
        pipeline = {}
        for fase, instancia in fases.items():
            algoritmos = instancia.ALGORITMOS
            while True:
                seleccion = self.rng.choice(algoritmos)
                if not self.permitir_none and seleccion is None:
                    continue
                if not self.permitir_llm and seleccion == "llm":
                    continue
                pipeline[fase] = str(seleccion) if seleccion is not None else None
                break
        
        return pipeline

    