import pandas as pd
import numpy as np

class QTable:
    def __init__(self):
        RUTA_Q_TABLE = "data/q-table/q_table.csv"
        self.q_table = pd.read_csv(RUTA_Q_TABLE)

        RUTA_CENTROIDES = "data/normalizacion/centroides_k10.csv"
        self.centroides = pd.read_csv(RUTA_CENTROIDES).to_numpy()

    def determinar_cluster(self, vector: list[float], numero_cluster: int = 1) -> str:
        """
        Dado un vector de meta-features, determina a qué cluster pertenece
        comparándolo con los centroides usando distancia euclidiana.

        Args:
            vector: vector de meta-features normalizado.
            numero_cluster: 1 devuelve el cluster más cercano, 2 el segundo
                más cercano, y así sucesivamente.
        """
        numero_cluster = max(1, numero_cluster)
        vector_np = np.array(vector)

        distancias = []
        for i, centroide in enumerate(self.centroides):
            distancia = np.sqrt(np.sum((vector_np - centroide[1:]) ** 2))
            distancias.append((distancia, f"cluster{i}"))

        distancias.sort(key=lambda elemento: elemento[0])
        indice = min(numero_cluster - 1, len(distancias) - 1)
        return distancias[indice][1]

    def obtener_acciones_ordenadas(self, nombre_estado: str) -> list[str]:
        """
        Dado el nombre del estado (ej: 'cluster6_fase3'), retorna la lista de acciones
        recomendadas ordenadas de mayor a menor valor Q.

        Args:
            nombre_estado: identificador completo del estado, ej. "cluster6_fase3".

        Returns:
            Lista de nombres de acciones ordenadas por Q-value descendente.
            Si el estado no existe en la tabla, retorna lista vacía.
        """
        filtro = self.q_table[self.q_table['estado'] == nombre_estado].copy()
        if filtro.empty:
            return []

        filtro_ordenado = filtro.sort_values('q_value', ascending=False)
        resultado = filtro_ordenado['accion'].tolist()
        return resultado

    