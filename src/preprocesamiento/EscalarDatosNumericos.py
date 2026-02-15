import numpy as np
import pandas as pd

from ..RegistroTecnica import RegistroTecnica
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

class EscalarDatosNumericos(BaseEstimator, TransformerMixin, RegistroTecnica):
    _instance = None  # Atributo de clase para almacenar la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EscalarDatosNumericos, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        semilla: para reproducibilidad
        """
        # Evitamos re-inicializar si la instancia ya existe
        if not hasattr(self, "_initialized"):
            RegistroTecnica.__init__(self, log_fase="escalar_datos_numericos")
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

    def fit(self, X, y=None):
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
                "min_max", 
                "max_abs_scaler", 
                "standard_scaler", 
                "robust_scaler"
            ])
            self.log_algoritmo = generador_aleatorio.choice(TECNICAS)

            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X)

        self.registrar_algoritmo(self.log_algoritmo)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Aplica el escalado seleccionado a las columnas numéricas
        """
        match self.log_algoritmo:
            case None:
                return X
            
            case "min_max" | "standard_scaler" | "robust_scaler" | "max_abs_scaler":
                X_escalado = self._escalar_con_parametros(X.copy())
                return X_escalado
            
            case _:
                raise ValueError(f"Técnica de escalado desconocida: {self.log_algoritmo}")
    
    def _get_instancia_scaler(self):
        if self.log_algoritmo == "min_max":
            return MinMaxScaler()
        elif self.log_algoritmo == "standard_scaler":
            return StandardScaler()
        elif self.log_algoritmo == "robust_scaler":
            return RobustScaler()
        elif self.log_algoritmo == "max_abs_scaler":
            return MaxAbsScaler()
        else:
            return None

    def _calcular_parametros(self, X_df: pd.DataFrame) -> None:
        """
        Calcula los parámetros necesarios para la técnica seleccionada y los guarda en self.log_params
        """
        cols_numericas = X_df.select_dtypes(include=np.number).columns
        self.log_params["columnas"] = cols_numericas.tolist()

        escalador = self._get_instancia_scaler()
        escalador.fit(X_df[cols_numericas])

        if self.log_algoritmo == "min_max": 
            self.log_params["params"] = {
                "feature_range": escalador.feature_range,
                "data_min_": escalador.data_min_.tolist(),
                "data_max_": escalador.data_max_.tolist(),
                "data_range_": escalador.data_range_.tolist(),
                "scale_": escalador.scale_.tolist(),
                "min_": escalador.min_.tolist(),
                "n_features_in_": escalador.n_features_in_
            }

        elif self.log_algoritmo == "standard_scaler":
            self.log_params["params"] = {
                "with_mean": escalador.with_mean,
                "with_std": escalador.with_std,
                "mean_": escalador.mean_.tolist() if escalador.with_mean else None,
                "var_": escalador.var_.tolist() if escalador.with_std else None,
                "scale_": escalador.scale_.tolist() if escalador.with_std else None,
                "n_features_in_": escalador.n_features_in_
            }

        elif self.log_algoritmo == "robust_scaler":
            self.log_params["params"] = {
                "with_centering": escalador.with_centering,
                "with_scaling": escalador.with_scaling,
                "quantile_range": escalador.quantile_range,
                "center_": escalador.center_.tolist() if escalador.with_centering else None,
                "scale_": escalador.scale_.tolist() if escalador.with_scaling else None,
                "n_features_in_": escalador.n_features_in_
            }

        elif self.log_algoritmo == "max_abs_scaler":
            self.log_params["params"] = {
                "max_abs_": escalador.max_abs_.tolist(),
                "scale_": escalador.scale_.tolist(),
                "n_features_in_": escalador.n_features_in_
            }

        self.registrar_parametros(self.log_params)

    def _escalar_con_parametros(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el scaler configurado en self.log_algoritmo
        utilizando los parámetros guardados en self.log_params
        """

        escalador = self._get_instancia_scaler()

        # Asignar automáticamente todos los parámetros
        for key, value in self.log_params["params"].items():

            # Convertir listas a numpy cuando corresponda
            if isinstance(value, list):
                value = np.array(value)

            setattr(escalador, key, value)

        cols_numericas = X_df.select_dtypes(include=np.number).columns

        X_scaled = escalador.transform(X_df[cols_numericas])
        X_df[cols_numericas] = np.round(X_scaled, 3).astype(np.float32)

        return X_df
