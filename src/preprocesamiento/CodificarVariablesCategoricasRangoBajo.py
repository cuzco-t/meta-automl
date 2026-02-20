import numpy as np
import pandas as pd

from pandas.api.types import is_object_dtype, is_categorical_dtype

from ..RegistroTecnica import RegistroTecnica

class CodificarVariablesCategoricasRangoBajo(RegistroTecnica):
    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        semilla: para reproducibilidad
        """
        RegistroTecnica.__init__(self, log_fase="codificar_variables_categoricas_rango_bajo")
        self.permitir_none = permitir_none
        self.semilla = semilla
        self.config_test = config_test
        
        self.ALGORITMOS = [
            None, 
            "one_hot_encoding", 
            "label_encoding"
        ]

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Decide aleatoriamente la técnica a aplicar y la guarda en self.log_algoritmo
        """
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

        else:
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
                return X, y
            
            case "one_hot_encoding":
                X_codificado = self._codificar_one_hot_encoding(X.copy())
                return X_codificado, y
            
            case "label_encoding":
                X_codificado = self._codificar_label_encoding(X.copy())
                return X_codificado, y
            
            case _:
                raise ValueError(f"Técnica de codificación no soportada: {self.log_algoritmo}")
        
    def _calcular_parametros(self, X: pd.DataFrame) -> None:
        """
        Calcula y guarda en self.log_params los parámetros necesarios para la técnica seleccionada
        """
        for col in X.columns:
            if not (is_object_dtype(X[col]) or is_categorical_dtype(X[col])):
                continue

            ratio_unicos = X[col].nunique() / len(X)

            if ratio_unicos <= 0.05:
                if self.log_algoritmo == "one_hot_encoding":
                    self.log_params[col] = X[col].dropna().unique().tolist()

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
        for col, valores in self.log_params.items():
            # Crear dummies para los valores presentes en X_copy
            dummies = pd.get_dummies(X_copy[col], prefix=col, dtype=int)
            
            # Reindexar para asegurar que existan todas las columnas de entrenamiento
            columnas_esperadas = [f"{col}_{v}" for v in valores]
            dummies = dummies.reindex(columns=columnas_esperadas, fill_value=0)
            
            # Reemplazar columna original con dummies
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
                X_copy[col] = X_copy[col].map(mapa).astype(float)

        return X_copy
