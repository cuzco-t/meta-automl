import numpy as np
import pandas as pd
from scipy import stats

from ..RegistroTecnica import RegistroTecnica

class NormalizarDatosNumericos(RegistroTecnica):
    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        semilla: para reproducibilidad
        """
        RegistroTecnica.__init__(self, log_fase="normalizar_datos_numericos")
        self.permitir_none = permitir_none
        self.semilla = semilla
        self.config_test = config_test
        self.ALGORITMOS = [
            "box_cox", 
            "cuadrado"
            "inverso",
            "ln", 
            None, 
            "sqrt", 
            "z_score", 
        ]

    def _permitir_none(self, tecnicas):
        if not self.permitir_none:
            return [t for t in tecnicas if t is not None]
        return tecnicas

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Selecciona aleatoriamente la técnica de normalización
        """        
        if self.config_test is not None:
            self.log_algoritmo = self.config_test.get("algoritmo")
            self.log_params = self.config_test.get("params")

        else:
            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X)

        self.registrar_algoritmo(self.log_algoritmo)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Aplica la técnica seleccionada a las columnas numéricas
        """
        match self.log_algoritmo:
            case None:
                return X, y
            
            case "z_score" | "box_cox" | "sqrt" | "ln":
                X_normalizado = self._normalizar_con_parametros(X.copy())
                return X_normalizado, y
            
            case "inverso":
                X_normalizado = self._normalizar_con_inverso(X.copy())
                return X_normalizado, y
            
            case "cuadrado":
                X_normalizado = self._normalizar_con_cuadrado(X.copy())
                return X_normalizado, y
            
            case _:
                raise ValueError(f"Técnica de normalización desconocida: {self.log_algoritmo}")

    def _calcular_parametros(self, X_df: pd.DataFrame) -> None:
        """
        Calcula los parámetros necesarios para las técnicas matemáticas:
            None, z_score, box_cox, cuadrado, sqrt, ln, inverso
        y los guarda en self.log_params para poder aplicarlos luego.
        """
        cols_numericas = X_df.select_dtypes(include=np.number).columns

        if self.log_algoritmo is None:
            self.log_params = {}
        
        elif self.log_algoritmo == "cuadrado":
            self.log_params = {}

        elif self.log_algoritmo == "z_score":
            self.log_params = {
                "mean": {col: float(X_df[col].mean()) for col in cols_numericas},
                "std": {col: float(X_df[col].std(ddof=0)) for col in cols_numericas}
            }

        elif self.log_algoritmo == "box_cox":
            self.log_params["lambda"] = {}
            self.log_params["desplazamiento"] = {}

            for col in cols_numericas:
                col_data = X_df[col].copy()
                desplazamiento = 0

                # box_cox requiere datos positivos
                if (col_data <= 0).any():
                    desplazamiento = abs(col_data.min()) + 1e-3
                    col_data = col_data + desplazamiento

                _, lam = stats.boxcox(col_data)
                self.log_params["lambda"][col] = float(lam)
                self.log_params["desplazamiento"][col] = float(desplazamiento)

        elif self.log_algoritmo in ["sqrt", "ln"]:
            self.log_params["desplazamiento"] = {}
            for col in cols_numericas:
                desplazamiento = 0

                # Solo sqrt, ln e inverso necesitan desplazamiento para evitar <=0
                if self.log_algoritmo in ["sqrt", "ln"] and (X_df[col] <= 0).any():
                    desplazamiento = abs(X_df[col].min()) + 1e-3
                    
                self.log_params["desplazamiento"][col] = float(desplazamiento)
        
        elif self.log_algoritmo == "inverso":
            self.log_params = {}

        self.registrar_parametros(self.log_params)
   
    def _normalizar_con_parametros(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza usando la técnica seleccionada y los parámetros calculados en fit()
         - z_score: (X - mean) / std
         - box_cox: stats.boxcox(X + desplazamiento, lambda)
         - sqrt: sqrt(X + desplazamiento)
         - ln: log1p(X + desplazamiento)
        """
        columnas_numericas = X_df.select_dtypes(include=np.number).columns

        for col in columnas_numericas:
            if self.log_algoritmo == "z_score":
                X_df[col] = (X_df[col] - self.log_params["mean"][col]) / self.log_params["std"][col]

            elif self.log_algoritmo == "box_cox":
                desplazamiento = self.log_params["desplazamiento"][col]
                lam = self.log_params["lambda"][col]
                X_df[col] = stats.boxcox(X_df[col] + desplazamiento, lam)

            elif self.log_algoritmo == "sqrt":
                desplazamiento = self.log_params["desplazamiento"][col]
                X_df[col] = np.sqrt(X_df[col] + desplazamiento)

            elif self.log_algoritmo == "ln":
                desplazamiento = self.log_params["desplazamiento"][col]
                X_df[col] = np.log1p(X_df[col] + desplazamiento)

        return X_df
    
    def _normalizar_con_cuadrado(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza usando la técnica de cuadrado: X^2
        """
        columnas_numericas = X_df.select_dtypes(include=np.number).columns

        for col in columnas_numericas:
            X_df[col] = X_df[col] ** 2

        return X_df
    
    def _normalizar_con_inverso(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza usando la técnica de inverso: 1 / X
        """
        columnas_numericas = X_df.select_dtypes(include=np.number).columns

        for col in columnas_numericas:
            X_df[col] = 1 / X_df[col]

        return X_df