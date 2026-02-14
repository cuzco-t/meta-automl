import numpy as np
import pandas as pd

from ..RegistroTecnica import RegistroTecnica
from sklearn.base import BaseEstimator, TransformerMixin

class CodificarVariablesCategoricasRangoMedio(BaseEstimator, TransformerMixin, RegistroTecnica):
    _instance = None  # Atributo de clase para la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CodificarVariablesCategoricasRangoMedio, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, random_state=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        random_state: para reproducibilidad
        """
        # Evitamos re-inicializar si la instancia ya existe
        if not hasattr(self, "_initialized"):
            self.nombre_fase = "codificar_variables_categoricas_rango_medio"
            self.permitir_none = permitir_none
            self.random_state = random_state
            self.tecnica_seleccionada_ = None
            self.parametro_tecnica_ = {}
            self._initialized = True

    def reiniciar(self):
        """
        Reinicia valores de logs de selección de técnica y parámetros para la próxima ejecución del pipeline.
        Esto es necesario porque esta clase es un singleton y se reutiliza en cada fold del pipeline
        """
        self.tecnica_seleccionada_ = None
        self.parametro_tecnica_ = {}

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X, y=None):
        """
        Selecciona aleatoriamente la técnica a aplicar a las variables categóricas
        de rango medio y la guarda en self.tecnica_seleccionada_
        """
        if self.tecnica_seleccionada_ is not None:
            return self
        
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "frequency-encoding", "eliminar"]
        TECNICAS = self._permitir_none(TECNICAS)

        self.tecnica_seleccionada_ = generador_aleatorio.choice(TECNICAS)
        return self

    def transform(self, X, y=None):
        """
        Aplica la codificación seleccionada a las variables categóricas de rango medio
        (ratio de valores únicos entre 0.05 y 0.90)
        """
        if self.tecnica_seleccionada_ is None:
            self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, None)
            return X if y is None else (X, y)

        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        columnas_a_eliminar = []
        for col in X_df.columns:
            if pd.api.types.is_object_dtype(X_df[col]):
                ratio_unicos = X_df[col].nunique() / len(X_df)

                # Dataset de entrenamiento
                if col not in self.parametro_tecnica_:
                    if 0.05 < ratio_unicos < 0.90:
                        if self.tecnica_seleccionada_ == "frequency-encoding":
                            if self.parametro_tecnica_.get(col, None) is None:
                                self.parametro_tecnica_[col] = X_df[col].value_counts(normalize=True)

                            X_df[col] = X_df[col].map(self.parametro_tecnica_[col])
                        
                        elif self.tecnica_seleccionada_ == "eliminar":
                            columnas_a_eliminar.append(col)
                
                # Dataset de validacion
                else:
                    if self.tecnica_seleccionada_ == "frequency-encoding":
                        X_df[col] = X_df[col].map(self.parametro_tecnica_[col])
                    
                    elif self.tecnica_seleccionada_ == "eliminar":
                        pass  # La columna ya se eliminará al final del transform

        if columnas_a_eliminar:
            X_df = X_df.drop(columns=columnas_a_eliminar)
            self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, "eliminar_columnas")
        else:
            self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

        if y is None:
            return X_df
        else:
            return X_df, y
