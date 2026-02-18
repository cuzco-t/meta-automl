import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_object_dtype, is_categorical_dtype

from ..RegistroTecnica import RegistroTecnica

class CodificarVariablesBinarias(BaseEstimator, TransformerMixin, RegistroTecnica):
    _instance = None  # Atributo de clase para la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CodificarVariablesBinarias, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        semilla: para reproducibilidad
        """
        # Evitamos re-inicializar si la instancia ya existía
        if not hasattr(self, "_initialized"):
            RegistroTecnica.__init__(self, log_fase="codificar_variables_binarias")
            self.log_fase = "codificar_variables_binarias"
            self.permitir_none = permitir_none
            self.semilla = semilla
            self.config_test = config_test
            self.reiniciar()
            self._initialized = True

    def reiniciar(self):
        """
        Reinicia valores de logs de selección de técnica y parámetros para la próxima ejecución del pipeline.
        Esto es necesario porque esta clase es un singleton y se reutiliza en cada fold del pipeline
        """
        self.log_algoritmo = None
        self.log_params = {}

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Decide aleatoriamente la técnica a aplicar y la guarda en self.log_algoritmo
        """
        if self.log_algoritmo is not None:
            return self
        
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

        else:
            generador_aleatorio = np.random.default_rng()
            TECNICAS = self._permitir_none([
                None, 
                "label_encoding"
            ])
            self.log_algoritmo = generador_aleatorio.choice(TECNICAS)
            
            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X)

        self.registrar_algoritmo(self.log_algoritmo)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Aplica la codificación seleccionada a las variables binarias (2 categorías)
        """
        match self.log_algoritmo:
            case None:
                return X
            
            case "label_encoding":
                X_codificado = self._codificar_label_encoding(X.copy())
                return X_codificado

            case _:
                raise ValueError(f"Técnica de codificación desconocida: {self.log_algoritmo}")

    def _calcular_parametros(self, X: pd.DataFrame):
        """
        Calcula y guarda en self.log_params los parámetros necesarios para la técnica seleccionada
        """
        for col in X.columns:
            if not ((is_object_dtype(X[col]) or is_categorical_dtype(X[col])) and X[col].nunique() == 2):
                continue

            if self.log_algoritmo == "label_encoding":
                categorias = sorted(X[col].dropna().unique())
                mapeo = {cat: i for i, cat in enumerate(categorias)}
                self.log_params[col] = mapeo

            self.registrar_parametros(self.log_params)

    def _codificar_label_encoding(self, X_copy: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica label encoding a las columnas binarias codificadas en self.log_params
        
        :return: DataFrame con las columnas binarias codificadas
        :rtype: DataFrame
        """
        for col, mapa in self.log_params.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapa).astype(float)
        
        return X_copy