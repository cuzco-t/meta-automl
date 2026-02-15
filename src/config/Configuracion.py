import os

from dotenv import load_dotenv


class Configuracion:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Ojo: __init__ se ejecuta cada vez que llamas a la clase
        if not hasattr(self, "_initialized"):
            self._cargar_configuracion()
            self._initialized = True

    def _cargar_configuracion(self):
        load_dotenv()
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        
        self.semilla_aleatoria = int(os.getenv('SEMILLA_ALEATORIA', 42))
        self.etiqueta_error = os.getenv('ETIQUETA_ERROR', 'ERROR')
        self.permitir_none = os.getenv('PERMITIR_NONE').lower() == 'true'
        self.permitir_llm = os.getenv('PERMITIR_LLM').lower() == 'true'

        