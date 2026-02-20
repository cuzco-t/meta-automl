import numpy as np
import pandas as pd

from ..RegistroTecnica import RegistroTecnica

class TratarFaltantesStrings(RegistroTecnica):
    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        semilla: para reproducibilidad
        """
        RegistroTecnica.__init__(self, log_fase="tratar_faltantes_strings")
        self.permitir_none = permitir_none
        self.semilla = semilla
        self.config_test = config_test
        self.ALGORITMOS = [
            None, 
            "moda", 
            "aleatorio", 
            "eliminar", 
            "etiqueta_desconocido"
        ]

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas
    

    def fit(self, X, y=None):
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

    def transform(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Aplica la técnica seleccionada a los valores faltantes de tipo string
        """

        match self.log_algoritmo:
            case None:
                return X, y
            
            case "moda":
                X_imputado = self._imputar_con_parametros(X.copy())
                return X_imputado, y
            
            case "aleatorio":
                X_imputado = self._imputar_aleatorio(X.copy())
                return X_imputado, y
            
            case "eliminar":
                X_clean, y_clean = self._eliminar(X.copy(), y.copy() if y is not None else None)
                return X_clean, y_clean
            
            case "etiqueta_desconocido":
                X_imputado = self._imputar_etiqueta_desconocido(X.copy())
                return X_imputado, y
            
            case _:
                raise ValueError(f"Técnica desconocida: {self.log_algoritmo}")

    def _calcular_parametros(self, X_df: pd.DataFrame) -> None:
        """
        Calcula y guarda en self.log_params los parámetros necesarios para la técnica seleccionada
        """
        for col in X_df.columns:
            es_texto = (
                X_df[col].dtype == "object"
                or pd.api.types.is_string_dtype(X_df[col])
            )

            if not (es_texto and X_df[col].isna().any()):
                continue

            if self.log_algoritmo == "moda":
                moda = X_df[col].mode()
                self.log_params[col] = moda.iloc[0] if not moda.empty else None
            
            elif self.log_algoritmo == "aleatorio":
                valores_validos = X_df[col].dropna().unique()
                self.log_params[col] = list(valores_validos) if len(valores_validos) > 0 else None

            elif self.log_algoritmo == "etiqueta_desconocido":
                self.log_params[col] = "DESCONOCIDO"

            self.registrar_parametros(self.log_params)
        
    def _imputar_con_parametros(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputa los valores faltantes usando los parámetros calculados previamente y guardados en self.log_params
        
        :return: DataFrame con los valores faltantes imputados
        :rtype: DataFrame
        """
        for col, valor in self.log_params.items():
            X_df[col] = X_df[col].fillna(valor)
        return X_df
    
    def _imputar_aleatorio(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputa los valores faltantes de forma aleatoria usando los valores válidos de cada columna
        
        :return: DataFrame con los valores faltantes imputados
        :rtype: DataFrame
        """

        rng = np.random.default_rng(self.semilla)

        for col in X_df.columns:
            es_texto = (
                X_df[col].dtype == "object"
                or pd.api.types.is_string_dtype(X_df[col])
            )

            if not (es_texto and X_df[col].isna().any()):
                continue

            valores_validos = self.log_params.get(col)
            if valores_validos is None:
                continue

            cantidad_faltantes = X_df[col].isna().sum()
            X_df.loc[X_df[col].isna(), col] = rng.choice(valores_validos, cantidad_faltantes)

        return X_df
    
    def _eliminar(self, X_df: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Elimina filas con valores faltantes en columnas de tipo string en X_df y las correspondientes en y_df
        
        :return: X_df e y_df sin filas con valores faltantes
        :rtype: tuple[DataFrame, Series]
        """
        # Seleccionar solo columnas de tipo string
        cols_string = X_df.select_dtypes(include="object").columns

        if len(cols_string) == 0:
            # No hay columnas de tipo string, no eliminamos nada
            return X_df, y

        # Mascara: True si la fila NO tiene NaN en ninguna columna de tipo string
        mask = ~X_df[cols_string].isna().any(axis=1)

        # Filtrar X_df y y_df según la máscara
        X_clean = X_df.loc[mask]
        y_clean = y.loc[mask] if y is not None else None

        return X_clean, y_clean
    
    def _imputar_etiqueta_desconocido(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputa los valores faltantes con la etiqueta "DESCONOCIDO"
        
        :return: DataFrame con los valores faltantes imputados
        :rtype: DataFrame
        """
        for col in X_df.columns:
            es_texto = (
                X_df[col].dtype == "object"
                or pd.api.types.is_string_dtype(X_df[col])
            )

            if not (es_texto and X_df[col].isna().any()):
                continue

            valor_desconocido = self.log_params.get(col, "DESCONOCIDO")
            X_df[col] = X_df[col].fillna(valor_desconocido)

        return X_df
    