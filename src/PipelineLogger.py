import logging

from logging_loki import LokiHandler
from pythonjsonlogger import jsonlogger

from src.config.Configuracion import Configuracion


class PipelineLogger:
    def __init__(self):
        self.logger = logging.getLogger("mi_pipeline")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            self._configure()

    def _configure(self):
        config = Configuracion()

        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(fase)s %(message)s'
        )

        # Verificamos que Loki esté correctamente configurado
        if all([config.loki_url, config.loki_username, config.loki_api_key]):
            handler = LokiHandler(
                url=config.loki_url,
                tags={"app": "meta-automl"},
                auth=(config.loki_username, config.loki_api_key),
                version="1",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        else:
            # Si faltan variables, caemos a consola para no perder logs
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.warning(
                "Loki no configurado. Se usarán logs por consola.",
                extra={"fase": "configuración"}
            )
            
    def get_logger(self):
        return self.logger
