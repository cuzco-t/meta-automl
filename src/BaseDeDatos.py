import os
import psycopg

from src.config.Configuracion import Configuracion


class BaseDeDatos:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Evita que se reinicialice si ya se creó la instancia
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.conn = None
        self.conectar()
        

    def conectar(self):
        configuracion = Configuracion()

        if self.conn is None or self.conn.closed:
            self.conn = psycopg.connect(
                host=configuracion.db_host,
                dbname=configuracion.db_name,
                user=configuracion.db_user,
                password=configuracion.db_password,
                port=configuracion.db_port,
            )
            print("Conexión a la base de datos establecida.")
    
    def guardar_meta_features_globales(self, meta_features_json: str, meta_features_vectorizadas: list):
        query = """
        INSERT INTO demo (meta_features_json_humano, meta_features_json_binario, meta_features_json_vector)
        VALUES (%s, %s, %s)
        """
        params = (meta_features_json, meta_features_json, meta_features_vectorizadas)
        self.insertar(query, params)


    def ejecutar_script_sql(self, ruta_sql):
        """
        Ejecuta un archivo .sql completo (schema, extensiones, etc.)
        """
        conn = self.conectar()

        with open(ruta_sql, "r", encoding="utf-8") as f:
            sql = f.read()

        with conn.cursor() as cur:
            cur.execute(sql)

        conn.commit()
        print(f"Script {ruta_sql} ejecutado con éxito.")


    def insertar(self, query, params):
        """
        Ejecuta un INSERT parametrizado
        """
        conn = self.conn
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()
        print("Inserción realizada con éxito.")


    def cerrar(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            print("Conexión a la base de datos cerrada.")
