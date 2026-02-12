import numpy as np
import pandas as pd

from ..RegistroTecnica import RegistroTecnica
from sklearn.base import BaseEstimator, TransformerMixin

class CodificarVariablesBinarias(BaseEstimator, TransformerMixin, RegistroTecnica):
    _instance = None  # Atributo de clase para la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CodificarVariablesBinarias, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, random_state=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        random_state: para reproducibilidad
        """
        # Evitamos re-inicializar si la instancia ya existía
        if not hasattr(self, "_initialized"):
            self.nombre_fase = "codificar_variables_binarias"
            self.permitir_none = permitir_none
            self.random_state = random_state
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
        Selecciona aleatoriamente la técnica a aplicar en variables binarias
        y la guarda en self.tecnica_seleccionada_
        """
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "label-encoding"]
        TECNICAS = self._permitir_none(TECNICAS)

        self.tecnica_seleccionada_ = generador_aleatorio.choice(TECNICAS)
        return self

    def transform(self, X, y=None):
        """
        Aplica la codificación seleccionada a las variables binarias (2 categorías)
        """
        if self.tecnica_seleccionada_ is None:
            self.registrar_tecnica(self.nombre_fase, None, None)
            return X if y is None else (X, y)

        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        for col in X_df.columns:
            if pd.api.types.is_object_dtype(X_df[col]) and X_df[col].nunique() == 2:
                if self.tecnica_seleccionada_ == "label-encoding":
                    if col not in self.parametro_tecnica_:
                        # Entrenamiento crea el mapa de categorías a números
                        categorias = sorted(X_df[col].dropna().unique())
                        mapa = {cat: i for i, cat in enumerate(categorias)}
                        self.parametro_tecnica_[col] = mapa
                        
                        X_df[col] = X_df[col].map(self.parametro_tecnica_[col])
                        self.registrar_tecnica(self.nombre_fase, self.tecnica_seleccionada_, self.parametro_tecnica_)

                    else:
                        # En validación se usa el mapa creado en entrenamiento
                        X_df[col] = X_df[col].map(self.parametro_tecnica_[col])

        if y is None:
            return X_df
        else:
            return X_df, y
