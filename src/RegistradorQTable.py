import psycopg2
from psycopg2 import sql

from src.config.Configuracion import Configuracion

from datetime import datetime

print_original = print


def print(*args, **kwargs):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_original(f"{ahora} |", *args, **kwargs)


class RegistradorQTable:
    """
    Clase independiente para guardar resultados del Q-Learning
    en la tabla 'modelo_q_learning'.

    - No reutiliza BaseDeDatos para evitar acoplamiento.
    - Verifica que la conexión esté activa antes de insertar.
    - Si se indica 'dataset_imposible', inserta -1111 en todas las métricas.
    """

    def __init__(self):
        config = Configuracion()
        self._config = config
        self._conn = None
        self._conectar()

    # ------------------------------------------------------------------
    # Gestión de conexión
    # ------------------------------------------------------------------
    def _conectar(self):
        """Establece una nueva conexión (o reconecta)."""
        try:
            if self._conn is None or self._conn.closed:
                self._conn = psycopg2.connect(
                    host=self._config.db_host,
                    dbname=self._config.db_name,
                    user=self._config.db_user,
                    password=self._config.db_password,
                    port=self._config.db_port,
                    prepare_threshold=None,  # Evita prepared statements
                )
                print("=" * 50)
                print("CONEXION A BD (RegistradorQTable) ESTABLECIDA")
                print("=" * 50)
        except Exception as e:
            print(f"[ERROR] Conexión fallida en RegistradorQTable: {e}")
            self._conn = None

    def _verificar_conexion(self):
        """
        Verifica que la conexión esté activa ejecutando un query simple.
        Si no lo está, reconecta.
        """
        try:
            if self._conn is None or self._conn.closed:
                self._conectar()
            else:
                # Ejecuta un SELECT 1 para comprobar que la conexión responde
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
        except Exception:
            print("[WARN] Conexión no válida, reconectando...")
            self._conectar()

    def cerrar(self):
        """Cierra la conexión si está abierta."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            print("Conexion RegistradorQTable cerrada.")

    # ------------------------------------------------------------------
    # Inserción en modelo_q_learning
    # ------------------------------------------------------------------
    def guardar_resultado(
        self,
        nombre_automl: str,
        task_id: int,
        nombre_dataset: str,
        fuente: str,
        cluster_nombre: str,
        tiempo: float,
        metricas: dict,
        formula_recompensa: str,
        dataset_imposible: bool = False,
    ) -> None:
        """
        Inserta un registro en la tabla modelo_q_learning.

        Args:
            nombre_automl: Identificador del experimento AutoML.
            task_id: ID de la tarea OpenML.
            nombre_dataset: Nombre del dataset.
            fuente: Origen de los datos (ej: "openml", "csv").
            cluster_nombre: Nombre del cluster asignado (ej: "cluster3").
            tiempo: Tiempo total de ejecución en segundos.
            metricas: Diccionario con las métricas (f1, accuracy, precision, etc.).
                       Si dataset_imposible es True, este argumento se ignora.
            formula_recompensa: Fórmula de recompensa utilizada.
            dataset_imposible: Si es True, inserta -1111 en todas las métricas.
        """
        self._verificar_conexion()
        if self._conn is None:
            print("[ERROR] No se pudo conectar a la BD. Registro NO insertado.")
            return

        # ---- Si el dataset es imposible, sobreescribir métricas con -1111 ----
        if dataset_imposible:
            metricas = {
                "f1": -1111,
                "accuracy": -1111,
                "precision": -1111,
                "recall": -1111,
                "mae": -1111,
                "mse": -1111,
                "rmse": -1111,
                "medae": -1111,
                "ev": -1111,
                "r2": -1111,
                "silhouette": -1111,
                "calinski": -1111,
                "davies": -1111,
            }
            print("[INFO] Dataset imposible de procesar -> métricas con -1111")

        # ---- Construir la sentencia INSERT ----
        insert_query = sql.SQL("""
            INSERT INTO public.modelo_q_learning (
                nombre_automl, task_id, nombre_dataset, fuente, cluster_nombre,
                tiempo,
                f1, accuracy, precision, recall,
                mae, mse, rmse, medae, ev, r2,
                silhouette, calinski, davies,
                formula_recompensa
            ) VALUES (
                %(nombre_automl)s, %(task_id)s, %(nombre_dataset)s, %(fuente)s,
                %(cluster_nombre)s, %(tiempo)s,
                %(f1)s, %(accuracy)s, %(precision)s, %(recall)s,
                %(mae)s, %(mse)s, %(rmse)s, %(medae)s, %(ev)s, %(r2)s,
                %(silhouette)s, %(calinski)s, %(davies)s,
                %(formula_recompensa)s
            )
        """)

        params = {
            "nombre_automl": nombre_automl,
            "task_id": task_id,
            "nombre_dataset": nombre_dataset,
            "fuente": fuente,
            "cluster_nombre": cluster_nombre,
            "tiempo": round(tiempo, 2),          # numeric(10,2)
            "f1": metricas.get("f1"),
            "accuracy": metricas.get("accuracy"),
            "precision": metricas.get("precision"),
            "recall": metricas.get("recall"),
            "mae": metricas.get("mae"),
            "mse": metricas.get("mse"),
            "rmse": metricas.get("rmse"),
            "medae": metricas.get("medae"),
            "ev": metricas.get("ev"),
            "r2": metricas.get("r2"),
            "silhouette": metricas.get("silhouette"),
            "calinski": metricas.get("calinski"),
            "davies": metricas.get("davies"),
            "formula_recompensa": formula_recompensa,
        }

        # ---- Ejecutar el INSERT ----
        try:
            with self._conn.cursor() as cur:
                cur.execute(insert_query, params)
            self._conn.commit()
            print("OK - Registro guardado en modelo_q_learning")
        except Exception as e:
            self._conn.rollback()
            print(f"[ERROR] Fallo al insertar en modelo_q_learning: {e}")