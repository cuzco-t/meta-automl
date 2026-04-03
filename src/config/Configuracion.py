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
            

    def _cargar_configuracion(self):
        load_dotenv()
        # Base de datos
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        
        # Semilla para reproducibilidad
        self.semilla_aleatoria = int(os.getenv('SEMILLA_ALEATORIA', 42))
        
        # Configuración para silenciar warnings de pymfe
        self.silenciar_pymfe_warnings = os.getenv('SILENCIAR_PYMFE_WARNINGS').lower() == 'true'

        # Variables de entorno para el logger
        self.loki_url = os.getenv('LOKI_URL')
        self.loki_username = os.getenv('LOKI_USERNAME')
        self.loki_api_key = os.getenv('LOKI_API_KEY')

        # Configuracion LLM
        self.llm_host = os.getenv('LLM_HOST', 'http://localhost:11434')
        self.llm_timeout = int(os.getenv('LLM_TIMEOUT', 10_800))  # 3 horas por defecto
        self.llm_modelo = os.getenv('LLM_MODELO', 'deepseek-r1:8b')
        self.llm_num_ctx = int(os.getenv('LLM_NUM_CTX', 20_000))
        self.permitir_llm = os.getenv('PERMITIR_LLM').lower() == 'true'

        # Configuraciones generales
        self.etiqueta_error = os.getenv('ETIQUETA_ERROR', 'ERROR')
        self.permitir_none = os.getenv('PERMITIR_NONE').lower() == 'true'
        self.num_pipelines_por_dataset = int(os.getenv('NUM_PIPELINES_POR_DATASET', 10))
        self.num_modelos_por_pipeline = int(os.getenv('NUM_MODELOS_POR_PIPELINE', 10))
        self.max_segundos_entrenamiento = int(os.getenv('MAX_SEGUNDOS_ENTRENAMIENTO', 300))  # 5 minutos por defecto

        