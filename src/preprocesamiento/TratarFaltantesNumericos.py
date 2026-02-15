import numpy as np
import pandas as pd

from scipy import stats

from ..RegistroTecnica import RegistroTecnica

class TratarFaltantesNumericos(RegistroTecnica):
    _instance = None  # Atributo de clase para almacenar la instancia única

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TratarFaltantesNumericos, cls).__new__(cls)
        return cls._instance

    def __init__(self, permitir_none=True, semilla=None, config_test=None):
        """
        permitir_none: si True, permite que no se aplique ninguna técnica
        semilla: para reproducibilidad
        """
        # Evitamos re-inicializar si ya existe la instancia
        if not hasattr(self, "_initialized"):
            RegistroTecnica.__init__(self, log_fase="tratar_faltantes_numericos")
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

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'TratarFaltantesNumericos':
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
                "media", 
                "mediana", 
                "moda", 
                "media_geometrica", 
                "aleatorio", 
                "eliminar"
            ])
            self.log_algoritmo = generador_aleatorio.choice(TECNICAS)
            
            self.registrar_algoritmo(self.log_algoritmo)
            self._calcular_parametros(X)

        self.registrar_algoritmo(self.log_algoritmo)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Aplica la técnica seleccionada a los valores faltantes numéricos
        """

        match self.log_algoritmo:
            case None:
                return X, y

            case "media" | "mediana" | "moda" | "media_geometrica":
                X_imputado = self._imputar_con_parametros(X.copy())
                return X_imputado, y
            
            case "aleatorio":
                X_imputado = self._imputar_aleatorio(X.copy())
                return X_imputado, y

            case "eliminar":
                X_clean, y_clean = self._eliminar(X.copy(), y.copy())
                return X_clean, y_clean
            
            case _:
                raise ValueError(f"Técnica desconocida: {self.log_algoritmo}")

    def _calcular_parametros(self, X_df: pd.DataFrame) -> None:
        """
        Calcula y guarda en self.log_params los parámetros necesarios para la técnica seleccionada
        """
        for col in X_df.columns:
            if not (pd.api.types.is_numeric_dtype(X_df[col]) and X_df[col].isna().any()):
                continue

            if self.log_algoritmo == "media":
                self.log_params[col] = float(np.round(X_df[col].mean(), 2))

            elif self.log_algoritmo == "mediana":
                self.log_params[col] = float(np.round(X_df[col].median(), 2))

            elif self.log_algoritmo == "moda":
                moda = X_df[col].mode()
                self.log_params[col] = moda.iloc[0] if not moda.empty else None

            elif self.log_algoritmo == "media_geometrica":
                valores = X_df[col].dropna()
                valores_pos = valores[valores > 0]

                if not valores_pos.empty:
                    self.log_params[col] = float(np.round(stats.gmean(valores_pos), 2))
                else:
                    self.log_params[col] = None

            elif self.log_algoritmo == "aleatorio":
                valores_validos = X_df[col].dropna().unique()
                self.log_params[col] = list(valores_validos) if len(valores_validos) > 0 else None

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
            if not (pd.api.types.is_numeric_dtype(X_df[col]) and X_df[col].isna().any()):
                continue

            valores_validos = self.log_params.get(col)
            if valores_validos is None:
                continue

            cantidad_faltantes = X_df[col].isna().sum()
            X_df.loc[X_df[col].isna(), col] = rng.choice(valores_validos, cantidad_faltantes)

        return X_df
    
    def _eliminar(self, X_df: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Elimina filas con valores faltantes en X_df y las correspondientes en y
        
        :return: X_df e y sin filas con valores faltantes
        :rtype: tuple[DataFrame, Series]
        """
        # Seleccionar solo columnas numéricas
        cols_numericas = X_df.select_dtypes(include=np.number).columns

        if len(cols_numericas) == 0:
            # No hay columnas numéricas, no eliminamos nada
            return X_df, y

        # Mascara: True si la fila NO tiene NaN en ninguna columna numérica
        mask = ~X_df[cols_numericas].isna().any(axis=1)

        # Filtrar X_df y y según la máscara
        X_clean = X_df.loc[mask]
        y_clean = y.loc[mask]

        return X_clean, y_clean
    