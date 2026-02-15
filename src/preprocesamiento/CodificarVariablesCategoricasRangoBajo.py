import numpy as np
import pandas as pd

from ..RegistroTecnica import RegistroTecnica
from sklearn.base import BaseEstimator, TransformerMixin

class CodificarVariablesCategoricasRangoBajo(BaseEstimator, TransformerMixin, RegistroTecnica):
    _instance = None  # Atributo de clase para almacenar la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CodificarVariablesCategoricasRangoBajo, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        semilla: para reproducibilidad
        """
        # Evitamos re-inicializar si la instancia ya existe
        if not hasattr(self, "_initialized"):
            RegistroTecnica.__init__(self, log_fase="codificar_variables_categoricas_rango_bajo")
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
                "one_hot_encoding", 
                "label_encoding"
            ])
            self.log_algoritmo = generador_aleatorio.choice(TECNICAS)
            
            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X)

        self.registrar_algoritmo(self.log_algoritmo)
        return self
    
    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Aplica la técnica de codificación seleccionada a las columnas categóricas
        de bajo rango (<=5% categorías únicas)
        """
        match self.log_algoritmo:
            case None:
                return X
            
            case "one_hot_encoding":
                X_codificado = self._codificar_one_hot_encoding(X.copy())
                return X_codificado
            
            case "label_encoding":
                X_codificado = self._codificar_label_encoding(X.copy())
                return X_codificado
            
            case _:
                raise ValueError(f"Técnica de codificación no soportada: {self.log_algoritmo}")
        
    def _calcular_parametros(self, X: pd.DataFrame) -> None:
        """
        Calcula y guarda en self.log_params los parámetros necesarios para la técnica seleccionada
        """
        for col in X.columns:
            if not pd.api.types.is_object_dtype(X[col]):
                continue

            ratio_unicos = X[col].nunique() / len(X)

            if ratio_unicos <= 0.05:
                if self.log_algoritmo == "one_hot_encoding":
                    self.log_params[col] = True
                    continue

                elif self.log_algoritmo == "label_encoding":
                    categorias = sorted(X[col].dropna().unique())
                    mapa = {cat: i for i, cat in enumerate(categorias)}
                    self.log_params[col] = mapa

            self.registrar_parametros(self.log_params)

    def _codificar_one_hot_encoding(self, X_copy: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica one hot encoding a las columnas categóricas de bajo rango codificadas en self.log_params
        
        :return: DataFrame con las columnas categóricas de bajo rango codificadas
        :rtype: DataFrame
        """
        for col in list(self.log_params.keys()):
            dummies = pd.get_dummies(X_copy[col], prefix=col, dtype=int)
            X_copy = pd.concat([X_copy.drop(columns=[col]), dummies], axis=1)

        return X_copy
    
    def _codificar_label_encoding(self, X_copy: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica label encoding a las columnas categóricas de bajo rango codificadas en self.log_params
        
        :return: DataFrame con las columnas categóricas de bajo rango codificadas
        :rtype: DataFrame
        """
        for col, mapa in self.log_params.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapa)

        return X_copy
