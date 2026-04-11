import logging

from logging_loki import LokiHandler
from pythonjsonlogger import jsonlogger

from src.config.Configuracion import Configuracion


class PipelineLogger:
    _instance = None  # Singleton

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = logging.getLogger("mi_pipeline")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # evita duplicados en root logger

        self._configure()
        self._initialized = True

    def _configure(self):
        config = Configuracion()

        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )

        # Evitar duplicar handlers si ya existen
        if self.logger.handlers:
            return

        # Loki configurado
        if all([config.loki_url, config.loki_username, config.loki_api_key]):
            handler = LokiHandler(
                url=config.loki_url,
                tags={"app": "mi_app_python", "env": "prod"},
                auth=(config.loki_username, config.loki_api_key),
                version="1",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        else:
            # fallback a consola
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            self.logger.warning(
                "Loki no configurado. Se usarán logs por consola.",
                extra={"fase": "configuración"}
            )

    def get_logger(self):
        return self.logger