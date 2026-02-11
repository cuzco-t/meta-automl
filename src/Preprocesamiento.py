import os
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from scipy import stats
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

class Preprocesamiento:
    def __init__(self):
        self.secuencia_preprocesamiento = {}
        self._SEMILLA = int(os.getenv("SEMILLA_ALEATORIA", "42"))
        self._ETIQUETA_ERROR = os.getenv("ETIQUETA_ERROR", "ERROR")
        self._RANGOS_RATIOS_UNICOS = {
            "bajo": (0, 0.05),
            "medio": (0.05, 0.9),
            "alto": (0.9, 1.0)
        }

        # random.seed(self._SEMILLA)
        # np.random.seed(self._SEMILLA)
    
    def reiniciar_secuencia(self):
        self.secuencia_preprocesamiento = {}

    def imprimir_tipos_datos(self, X, y, texto):
        print(f"{texto.upper()}")
        print(f"Tipos de datos en X:")
        for i in range(X.shape[1]):
            print(f"Columna {i}: dtype = {X.iloc[:, i].dtype}")
        print(f"Tipo de dato en y: dtype = {y.dtype}")

    def preprocesar_datos(self, X: np.ndarray, y: np.ndarray, tarea: str) -> tuple[np.ndarray, np.ndarray, list]:
        self._tarea = tarea
        self.reiniciar_secuencia()
        
        # try: 

        X, y = self._balancear_clases(X, y)
        X, y = self._tratar_duplicados(X, y)
        X = self._tratar_faltantes_numericos(X)
        X = self._tratar_faltantes_strings(X)

        X = self._codificar_variables_binarias(X)
        X = self._codificar_variables_categoricas_rango_bajo(X)
        X = self._codificar_variables_categoricas_rango_medio(X)
        self.imprimir_tipos_datos(X, y, "Después de codificar variables categóricas")
        print("-" * 50)
        pd.set_option('display.max_columns', None)
        print(X.head())

        X = self._tratar_outliers_numericos(X)
        X = self._escalar_datos_numericos(X)
        X = self._normalizar_datos_numericos(X)
        X = self._crear_nueva_variable(X)

        # except Exception as e:
        #     print(f"Error durante el preprocesamiento: {e}")
        #     return None, None, self.secuencia_preprocesamiento

        return X, y, self.secuencia_preprocesamiento

    def _seleccionar_opcion_aleatoria(self, opciones):
        return random.choice(opciones)

    # TODO Clasificación
    def _balancear_clases(self, X, y):
        if self._tarea == "regresion":
            self.secuencia_preprocesamiento["balancear_clases"] = None
            return X, y
        
        TECNICAS = [None, "smote", "undersampling", "oversampling", "Borderline-SMOTE"]
        
        tecnica_seleccionada = self._seleccionar_opcion_aleatoria(TECNICAS)

        try:
            if tecnica_seleccionada is None:
                pass

            elif tecnica_seleccionada == "smote":
                smote = SMOTE(random_state=self._SEMILLA)
                X, y = smote.fit_resample(X, y)

            elif tecnica_seleccionada == "undersampling":
                undersampler = RandomUnderSampler(random_state=self._SEMILLA)
                X, y = undersampler.fit_resample(X, y)

            elif tecnica_seleccionada == "oversampling":
                oversampler = RandomOverSampler(random_state=self._SEMILLA)
                X, y = oversampler.fit_resample(X, y)

            elif tecnica_seleccionada == "Borderline-SMOTE":
                borderline_smote = BorderlineSMOTE(random_state=self._SEMILLA)
                X, y = borderline_smote.fit_resample(X, y)

            self.secuencia_preprocesamiento.append(tecnica_seleccionada)

        except:
            self.secuencia_preprocesamiento.append(self._ETIQUETA_ERROR)

        return X, y


    def _tratar_duplicados(self, X: pd.DataFrame, y: pd.Series):
        TECNICAS = ["eliminar"]

        tecnica_seleccionada = self._seleccionar_opcion_aleatoria(TECNICAS)
        self.secuencia_preprocesamiento["tratar_duplicados"] = tecnica_seleccionada

        if tecnica_seleccionada != "eliminar":
            return X, y

        # Asegurar DataFrame para detectar duplicados
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Máscara de filas únicas
        mask = ~X_df.duplicated()

        # Aplicar máscara preservando tipos
        if isinstance(X, pd.DataFrame):
            X_clean = X.loc[mask]
        else:
            X_clean = X_df.loc[mask].to_numpy()

        if isinstance(y, pd.Series):
            y_clean = y.loc[mask]
        else:
            y_arr = np.asarray(y)
            y_clean = y_arr[mask.values]

        return X_clean, y_clean


    def _tratar_faltantes_numericos(self, X: pd.DataFrame):
        TECNICAS = [None, "media", "mediana", "moda", "aleatorio", "media_geometrica", "eliminar"]

        tecnica_seleccionada = self._seleccionar_opcion_aleatoria(TECNICAS)
        self.secuencia_preprocesamiento["tratar_faltantes_numericos"] = tecnica_seleccionada

        if tecnica_seleccionada is None:
            return X

        # Asegurar DataFrame
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        filas_a_eliminar = None  # para "eliminar"

        for col in X_df.columns:
            if not (pd.api.types.is_numeric_dtype(X_df[col]) and X_df[col].isna().any()):
                continue

            if tecnica_seleccionada == "media":
                X_df[col] = X_df[col].fillna(X_df[col].mean())

            elif tecnica_seleccionada == "mediana":
                X_df[col] = X_df[col].fillna(X_df[col].median())

            elif tecnica_seleccionada == "moda":
                moda = X_df[col].mode()
                if not moda.empty:
                    X_df[col] = X_df[col].fillna(moda.iloc[0])

            elif tecnica_seleccionada == "aleatorio":
                valores_validos = X_df[col].dropna().values
                if len(valores_validos) > 0:
                    X_df.loc[X_df[col].isna(), col] = np.random.choice(
                        valores_validos, X_df[col].isna().sum()
                    )

            elif tecnica_seleccionada == "media_geometrica":
                valores = X_df[col].dropna()
                valores_pos = valores[valores > 0]
                if not valores_pos.empty:
                    X_df[col] = X_df[col].fillna(stats.gmean(valores_pos))

            elif tecnica_seleccionada == "eliminar":
                mask = X_df[col].notna()
                filas_a_eliminar = mask if filas_a_eliminar is None else filas_a_eliminar & mask

        if tecnica_seleccionada == "eliminar" and filas_a_eliminar is not None:
            X_df = X_df.loc[filas_a_eliminar]

        return X_df

  
    def _tratar_faltantes_strings(self, X: pd.DataFrame):
        TECNICAS = [None, "moda", "aleatorio", "eliminar", "etiqueta_desconocido"]

        tecnica_seleccionada = self._seleccionar_opcion_aleatoria(TECNICAS)
        self.secuencia_preprocesamiento["tratar_faltantes_strings"] = tecnica_seleccionada

        if tecnica_seleccionada is None:
            return X

        # Asegurar DataFrame
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        filas_a_eliminar = None

        for col in X_df.columns:
            es_texto = (
                X_df[col].dtype == "object"
                or pd.api.types.is_string_dtype(X_df[col])
            )

            if not (es_texto and X_df[col].isna().any()):
                continue

            if tecnica_seleccionada == "moda":
                moda = X_df[col].mode()
                if not moda.empty:
                    X_df[col] = X_df[col].fillna(moda.iloc[0])

            elif tecnica_seleccionada == "aleatorio":
                valores_validos = X_df[col].dropna().values
                if len(valores_validos) > 0:
                    X_df.loc[X_df[col].isna(), col] = np.random.choice(
                        valores_validos, X_df[col].isna().sum()
                    )

            elif tecnica_seleccionada == "etiqueta_desconocido":
                X_df[col] = X_df[col].fillna("DESCONOCIDO")

            elif tecnica_seleccionada == "eliminar":
                mask = X_df[col].notna()
                filas_a_eliminar = mask if filas_a_eliminar is None else filas_a_eliminar & mask

        if tecnica_seleccionada == "eliminar" and filas_a_eliminar is not None:
            X_df = X_df.loc[filas_a_eliminar]

        return X_df


    def _codificar_variables_binarias(self, X: pd.DataFrame):
        TECNICAS = [None, "label-encoding"]

        tecnica_seleccionada = self._seleccionar_opcion_aleatoria(TECNICAS)
        self.secuencia_preprocesamiento["codificar_variables_binarias"] = tecnica_seleccionada

        if tecnica_seleccionada is None:
            return X
        # Asegurar DataFrame
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in X_df.columns:
            if pd.api.types.is_object_dtype(X_df[col]) and X_df[col].nunique() == 2:
                if tecnica_seleccionada == "label-encoding":
                    X_df[col] = X_df[col].astype('category').cat.codes

        return X_df


    def _codificar_variables_categoricas_rango_bajo(self, X: pd.DataFrame):
        TECNICAS = [None, "one-hot-encoding", "label-encoding"]

        tecnica_seleccionada = self._seleccionar_opcion_aleatoria(TECNICAS)
        self.secuencia_preprocesamiento["codificar_variables_categoricas_rango_bajo"] = tecnica_seleccionada

        if tecnica_seleccionada is None:
            return X

        # Asegurar DataFrame
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        for col in X_df.columns:
            if pd.api.types.is_object_dtype(X_df[col]):
                ratio_unicos = X_df[col].nunique() / len(X_df)
                if ratio_unicos <= 0.05:
                    if tecnica_seleccionada == "one-hot-encoding":
                        dummies = pd.get_dummies(X_df[col], prefix=col)
                        X_df = pd.concat([X_df.drop(columns=[col]), dummies], axis=1)

                    elif tecnica_seleccionada == "label-encoding":
                        X_df[col] = X_df[col].astype('category').cat.codes
                
                if ratio_unicos >= 0.90:
                    X_df = X_df.drop(columns=[col])

        return X_df
    

    def _codificar_variables_categoricas_rango_medio(self, X: pd.DataFrame):
        TECNICAS = [None, "frequency-encoding", "eliminar"]

        tecnica_seleccionada = self._seleccionar_opcion_aleatoria(TECNICAS)
        self.secuencia_preprocesamiento["codificar_variables_categoricas_rango_medio"] = tecnica_seleccionada

        if tecnica_seleccionada is None:
            return X

        # Asegurar DataFrame
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        columnas_a_eliminar = []

        for col in X_df.columns:
            if pd.api.types.is_object_dtype(X_df[col]):
                ratio_unicos = X_df[col].nunique() / len(X_df)
                if 0.05 < ratio_unicos < 0.90:
                    if tecnica_seleccionada == "frequency-encoding":
                        freq = X_df[col].value_counts(normalize=True)
                        X_df[col] = X_df[col].map(freq)
                    elif tecnica_seleccionada == "eliminar":
                        columnas_a_eliminar.append(col)

        if columnas_a_eliminar:
            X_df = X_df.drop(columns=columnas_a_eliminar)

        return X_df


    def _tratar_outliers_numericos(self, X: pd.DataFrame):
        TECNICAS = [None, "media", "mediana", "moda",
                    "aleatorio", "media_geometrica", "eliminar"]

        tecnica_seleccionada = self._seleccionar_opcion_aleatoria(TECNICAS)
        self.secuencia_preprocesamiento["tratar_outliers_numericos"] = tecnica_seleccionada

        if tecnica_seleccionada is None:
            return X

        # Asegurar DataFrame
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        filas_a_eliminar = None  # máscara global

        for col in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                continue

            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1

            filas_outliers = (
                (X_df[col] < (Q1 - 1.5 * IQR)) |
                (X_df[col] > (Q3 + 1.5 * IQR))
            )

            if not filas_outliers.any():
                continue

            if tecnica_seleccionada == "media":
                X_df.loc[filas_outliers, col] = X_df[col].mean()

            elif tecnica_seleccionada == "mediana":
                X_df.loc[filas_outliers, col] = X_df[col].median()

            elif tecnica_seleccionada == "moda":
                moda = X_df[col].mode()
                if not moda.empty:
                    X_df.loc[filas_outliers, col] = moda.iloc[0]

            elif tecnica_seleccionada == "aleatorio":
                valores_validos = X_df.loc[~filas_outliers, col].values
                if len(valores_validos) > 0:
                    X_df.loc[filas_outliers, col] = np.random.choice(
                        valores_validos, filas_outliers.sum()
                    )

            elif tecnica_seleccionada == "media_geometrica":
                valores = X_df.loc[~filas_outliers, col]
                valores_pos = valores[valores > 0]
                if not valores_pos.empty:
                    X_df.loc[filas_outliers, col] = stats.gmean(valores_pos)

            elif tecnica_seleccionada == "eliminar":
                mask = ~filas_outliers
                filas_a_eliminar = mask if filas_a_eliminar is None else filas_a_eliminar & mask

        if tecnica_seleccionada == "eliminar" and filas_a_eliminar is not None:
            X_df = X_df.loc[filas_a_eliminar]

        return X_df


    def _escalar_datos_numericos(self, X: pd.DataFrame):
        TECNICAS = [None, "min-max", "max-abs-scaler", "standard-scaler", "robust-scaler"]
        
        tecnica_seleccionada = self._seleccionar_opcion_aleatoria(TECNICAS)
        self.secuencia_preprocesamiento["escalar_datos_numericos"] = tecnica_seleccionada  # registramos técnica seleccionada

        # Saber si el input era numpy
        is_numpy = isinstance(X, np.ndarray)
        
        # Convertir a DataFrame temporal si es numpy
        if is_numpy:
            X_df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)

        if tecnica_seleccionada is None:
            # No se aplica ninguna técnica
            return X if is_numpy else X_df

        # Escalar solo columnas numéricas
        cols_numericas = X_df.select_dtypes(include=np.number).columns
        X_num = X_df[cols_numericas]

        if tecnica_seleccionada == "min-max":
            scaler = MinMaxScaler()
        elif tecnica_seleccionada == "standard-scaler":
            scaler = StandardScaler()
        elif tecnica_seleccionada == "robust-scaler":
            scaler = RobustScaler()
        elif tecnica_seleccionada == "max-abs-scaler":
            scaler = MaxAbsScaler()
        else:
            # Si algo falla, devolvemos original
            return X if is_numpy else X_df

        # Aplicar escalamiento
        X_scaled = scaler.fit_transform(X_num)

        # Reconstruir DataFrame sin perder dtypes de columnas no numéricas
        X_df_scaled = X_df.copy()
        X_df_scaled[cols_numericas] = X_scaled

        return X_df_scaled.values if is_numpy else X_df_scaled


    def _normalizar_datos_numericos(self, X: pd.DataFrame):
        TECNICAS = [None, "z-score", "box-cox", "cuadrado", "sqrt", "ln", "inverso"]
        tecnica_seleccionada = self._seleccionar_opcion_aleatoria(TECNICAS)
        self.secuencia_preprocesamiento["normalizar_datos_numericos"] = tecnica_seleccionada

        # Convertir a DataFrame si es necesario
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
            is_numpy = True
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
            is_numpy = False
        else:
            X_df = pd.DataFrame(X)
            is_numpy = False

        if tecnica_seleccionada is None:
            return X_df.values if is_numpy else X_df

        # Iterar solo sobre columnas numéricas
        for col in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                continue

            if tecnica_seleccionada == "z-score":
                X_df[col] = (X_df[col] - X_df[col].mean()) / X_df[col].std()

            elif tecnica_seleccionada == "box-cox":
                # Solo si todos los valores son positivos
                if (X_df[col] > 0).all():
                    X_df[col], _ = stats.boxcox(X_df[col])
                else:
                    # Si hay valores <=0, se omite la transformación
                    pass

            elif tecnica_seleccionada == "cuadrado":
                X_df[col] = X_df[col] ** 2

            elif tecnica_seleccionada == "sqrt":
                X_df[col] = np.sqrt(np.abs(X_df[col]))

            elif tecnica_seleccionada == "ln":
                # Log natural, sumamos 1 para evitar log(0)
                X_df[col] = np.log1p(np.abs(X_df[col]))

            elif tecnica_seleccionada == "inverso":
                X_df[col] = 1 / (1 + np.abs(X_df[col]))

        return X_df.values if is_numpy else X_df

    def _crear_nueva_variable(self, X: pd.DataFrame):
        TECNICAS = [None, "suma", "resta", "multiplicacion", "ratio", "pca"]
        tecnica_seleccionada = self._seleccionar_opcion_aleatoria(TECNICAS)
        self.secuencia_preprocesamiento["crear_nueva_variable"] = tecnica_seleccionada

        # Convertir a DataFrame si es necesario
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
            is_numpy = True
        else:
            X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            is_numpy = False

        if tecnica_seleccionada is None:
            return X_df.values if is_numpy else X_df

        # Filtrar solo columnas numéricas
        numeric_cols = [col for col in X_df.columns if pd.api.types.is_numeric_dtype(X_df[col])]

        if tecnica_seleccionada in ["suma", "resta", "multiplicacion", "ratio"]:
            if len(numeric_cols) >= 2:
                # Seleccionamos dos columnas numéricas al azar
                col1, col2 = random.sample(numeric_cols, 2)

                if tecnica_seleccionada == "suma":
                    X_df['suma'] = X_df[col1] + X_df[col2]
                elif tecnica_seleccionada == "resta":
                    X_df['resta'] = X_df[col1] - X_df[col2]
                elif tecnica_seleccionada == "multiplicacion":
                    X_df['multiplicacion'] = X_df[col1] * X_df[col2]
                elif tecnica_seleccionada == "ratio":
                    X_df['ratio'] = X_df[col1] / (X_df[col2] + 1e-10)  # evitar división por cero

        elif tecnica_seleccionada == "pca":
            if len(numeric_cols) >= 2:
                n_components = min(2, len(numeric_cols))
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(X_df[numeric_cols])
                for i in range(n_components):
                    X_df[f'pca_{i}'] = pca_result[:, i]

        return X_df.values if is_numpy else X_df

    