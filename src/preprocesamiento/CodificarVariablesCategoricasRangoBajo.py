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

    def __init__(self, permitir_none=True, random_state=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        random_state: para reproducibilidad
        """
        # Evitamos re-inicializar si la instancia ya existe
        if not hasattr(self, "_initialized"):
            self.log_fase = "codificar_variables_categoricas_rango_bajo"
            self.permitir_none = permitir_none
            self.random_state = random_state
            self.log_algoritmo = None
            self.log_params = {}
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

    def fit(self, X, y=None):
        """
        Selecciona aleatoriamente la técnica a aplicar a las variables categóricas
        de bajo rango y la guarda en self.log_algoritmo
        """
        if self.log_algoritmo is not None:
            return self
        
        generador_aleatorio = np.random.default_rng(self.random_state)
        TECNICAS = [None, "one-hot-encoding", "label-encoding"]
        TECNICAS = self._permitir_none(TECNICAS)

        self.log_algoritmo = generador_aleatorio.choice(TECNICAS)
        return self

    def transform(self, X, y=None):
        """
        Aplica la codificación seleccionada a las variables categóricas de bajo rango.
        También elimina columnas con ratio de valores únicos >= 0.90
        """
        if self.log_algoritmo is None:
            self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
            return X if y is None else (X, y)

        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        columnas_one_hot = []
        for col in X_df.columns:
            if pd.api.types.is_object_dtype(X_df[col]):
                ratio_unicos = X_df[col].nunique() / len(X_df)

                # Dataset de entrenamiento
                if col not in self.log_params:
                    # Columnas de bajo rango
                    if ratio_unicos <= 0.05:
                        if self.log_algoritmo == "one-hot-encoding":
                            dummies = pd.get_dummies(X_df[col], prefix=col, dtype=int)
                            X_df = pd.concat([X_df.drop(columns=[col]), dummies], axis=1)
                            columnas_one_hot.append(col)

                        elif self.log_algoritmo == "label-encoding":
                            # Crea el mapa de categorías a números
                            categorias = sorted(X_df[col].dropna().unique())
                            mapa = {cat: i for i, cat in enumerate(categorias)}
                            self.log_params[col] = mapa
                            
                            self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)
                            X_df[col] = X_df[col].map(self.log_params[col])

                # Dataset de validacion
                else:
                    if self.log_algoritmo == "one-hot-encoding":
                        dummies = pd.get_dummies(X_df[col], prefix=col, dtype=int)
                        X_df = pd.concat([X_df.drop(columns=[col]), dummies], axis=1)

                    elif self.log_algoritmo == "label-encoding":
                        X_df[col] = X_df[col].map(self.log_params[col])

        if self.log_algoritmo == "one-hot-encoding":
            self.log_params = columnas_one_hot
            self.registrar_tecnica(self.log_fase, self.log_algoritmo, self.log_params)


        if y is None:
            return X_df
        else:
            return X_df, y
