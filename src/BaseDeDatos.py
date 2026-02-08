import os
from dotenv import load_dotenv
import psycopg

load_dotenv()


class BaseDeDatos:
    def __init__(self):
        self.conn = None

    def conectar(self):
        if self.conn is None or self.conn.closed:
            self.conn = psycopg.connect(
                host=os.getenv("DB_HOST"),
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                port=os.getenv("DB_PORT", 5432),
            )
            print("Conexión a la base de datos establecida.")
        return self.conn


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
        conn = self.conectar()
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()
        print("Inserción realizada con éxito.")


    def cerrar(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            print("Conexión a la base de datos cerrada.")
