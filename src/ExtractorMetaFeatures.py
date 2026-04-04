from datetime import datetime
import os
import time
import warnings
from multiprocessing import Process, Queue

import math
import numpy as np
import pandas as pd

from pymfe.pymfe.mfe import MFE
from toon_format import encode
from collections.abc import Mapping

from contextlib import contextmanager
from src.config.Configuracion import Configuracion

print_original = print

def print(*args, **kwargs):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_original(f"{ahora} |", *args, **kwargs)


class MetaFeatureTimeoutError(TimeoutError):
    pass


def _ejecutar_mfe_fit_extract_en_proceso(queue, mfe_kwargs, X, y, silenciar_warnings):
    try:
        mfe = MFE(**mfe_kwargs)

        if silenciar_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mfe.fit(X, y)
                nombres, valores = mfe.extract()
        else:
            mfe.fit(X, y)
            nombres, valores = mfe.extract()

        queue.put(("ok", nombres, valores))
    except Exception as e:
        queue.put(("fail", repr(e)))


class ExtractorMetaFeatures:
    _GRUPOS_META_FEATURES = [
        "landmarking",
        "general",
        "statistical",
        "model-based",
        "info-theory",
        "relative",
        "clustering",
        "complexity",
        "itemset",
        "concept"
    ]
    # Alias para compatibilidad con referencias antiguas.
    _GUPOS_META_FEATURES = _GRUPOS_META_FEATURES
    _CONSTANTE_ERROR = -1111.0
    _CONSTANTE_INFINITO = 2222.0
    _RUTA_DATASET_TEMPORAL = "temp_dataset.csv"
    _TIMEOUT_MFE_FIT_SEGUNDOS = 180   # 3 minutos para toda la extracción

    def __init__(self):
        self.df = None
        self.warnings_pymfe = Configuracion().silenciar_pymfe_warnings
        # Mapeo de cada meta-feature a su grupo (construido una sola vez)
        self._feature_to_group = self._construir_mapeo_feature_a_grupo()

    def _construir_mapeo_feature_a_grupo(self):
        """Construye un diccionario {nombre_meta_feature: nombre_grupo} a partir de los grupos conocidos."""
        mapeo = {}
        for grupo in self._GRUPOS_META_FEATURES:
            for feature in self._setear_variables_grupo(grupo, 0).keys():
                mapeo[feature] = grupo
        return mapeo

    def extraer(self, ruta_absoluta, target):
        X, y = self._leer_dataset(ruta_absoluta, target)
        meta_features = self._extraer_meta_features_por_grupos(X, y)

        meta_features = self._mapear_meta_features(meta_features)
        meta_features = self._agregar_meta_features_personalizadas(ruta_absoluta, X, meta_features)
        meta_features_vectorizadas = self._vectorizar_meta_features(meta_features)

        print("OK - Extracción de meta-features completada para el dataset:", ruta_absoluta)

        return meta_features, meta_features_vectorizadas

    def extraer_desde_dataframe(self, X_df: pd.DataFrame, y_df: pd.Series, vectorizar=False):
        X = X_df.copy().to_numpy()
        y = y_df.copy().to_numpy() if y_df is not None else None
        meta_features = self._extraer_meta_features_por_grupos(X, y)

        meta_features = self._mapear_meta_features(meta_features)

        ruta_temporal = self._guardar_dataset_temporal(X_df, y_df)
        try:
            meta_features = self._agregar_meta_features_personalizadas(ruta_temporal, X, meta_features)
        finally:
            if os.path.exists(ruta_temporal):
                os.remove(ruta_temporal)

        meta_features_vectorizadas = None
        if vectorizar:
            meta_features_vectorizadas = self._vectorizar_meta_features(meta_features)

        return meta_features, meta_features_vectorizadas

    def _extraer_meta_features_por_grupos(self, X: np.ndarray, y: np.ndarray | None) -> dict:
        """
        Extrae grupos secuencialmente con timeout individual (30s por grupo) y
        tiempo global máximo de 3 minutos. Los grupos completados se conservan;
        los que fallen o excedan el tiempo se rellenan con _CONSTANTE_ERROR.
        """
        print("=" * 50)
        print("EXTRAYENDO META-FEATURES POR GRUPOS (con timeout individual y global)")
        print("=" * 50)

        TIEMPO_MAX_GLOBAL = self._TIMEOUT_MFE_FIT_SEGUNDOS  # 180 segundos
        TIMEOUT_POR_GRUPO = 30  # segundos, ajustable

        meta_features = {}
        tiempo_inicio_global = time.time()

        for i, grupo in enumerate(self._GRUPOS_META_FEATURES):
            # Verificar tiempo restante
            tiempo_transcurrido = time.time() - tiempo_inicio_global
            if tiempo_transcurrido >= TIEMPO_MAX_GLOBAL:
                print(f"Tiempo global agotado ({TIEMPO_MAX_GLOBAL}s). Se rellenan los {len(self._GRUPOS_META_FEATURES)-i} grupos restantes con error.")
                for grupo_restante in self._GRUPOS_META_FEATURES[i:]:
                    meta_features[grupo_restante] = self._setear_variables_grupo(grupo_restante, self._CONSTANTE_ERROR)
                break

            # Tiempo disponible para este grupo (lo que quede hasta el límite global, pero sin exceder TIMEOUT_POR_GRUPO)
            tiempo_restante = TIEMPO_MAX_GLOBAL - tiempo_transcurrido
            timeout_usar = min(TIMEOUT_POR_GRUPO, tiempo_restante)

            print(f"Grupo '{grupo}' - timeout: {timeout_usar:.1f}s (restante global: {tiempo_restante:.1f}s)")
            tiempo_inicio_grupo = time.time()

            try:
                nombres, valores = self._ejecutar_mfe_con_timeout(
                    mfe_kwargs={"groups": [grupo]},
                    X=X,
                    y=y,
                    timeout_segundos=timeout_usar
                )
                meta_features_grupo = dict(zip(nombres, valores))

                # Rellenar features faltantes dentro del grupo (si pymfe no devolvió alguna)
                grupo_completo = self._setear_variables_grupo(grupo, self._CONSTANTE_ERROR)
                grupo_completo.update(meta_features_grupo)
                meta_features[grupo] = grupo_completo

                duracion = time.time() - tiempo_inicio_grupo
                print(f"  Completado en {duracion:.2f}s - {len(meta_features_grupo)} features extraídas")

            except MetaFeatureTimeoutError:
                print(f"  TIMEOUT en grupo '{grupo}' después de {timeout_usar}s -> se rellena con error")
                meta_features[grupo] = self._setear_variables_grupo(grupo, self._CONSTANTE_ERROR)
            except Exception as e:
                print(f"  ERROR en grupo '{grupo}': {e} -> se rellena con error")
                meta_features[grupo] = self._setear_variables_grupo(grupo, self._CONSTANTE_ERROR)

        print(f"Extracción finalizada en {time.time() - tiempo_inicio_global:.2f} segundos globales")
        return meta_features

    def _guardar_dataset_temporal(self, X_df: pd.DataFrame, y_df: pd.Series | None) -> str:
        ruta_temporal = self._RUTA_DATASET_TEMPORAL

        if y_df is not None:
            dataset_temporal = pd.concat([X_df, y_df], axis=1)
        else:
            dataset_temporal = X_df.copy()

        dataset_temporal.to_csv(ruta_temporal, index=False)
        print("OK - Dataset temporal guardado en:", ruta_temporal)
        return ruta_temporal

    def _ejecutar_mfe_con_timeout(self, mfe_kwargs: dict, X: np.ndarray, y: np.ndarray | None,
                                  timeout_segundos: int | None = None):
        timeout = self._TIMEOUT_MFE_FIT_SEGUNDOS if timeout_segundos is None else timeout_segundos
        queue = Queue()
        proceso = Process(
            target=_ejecutar_mfe_fit_extract_en_proceso,
            args=(queue, mfe_kwargs, X, y, self.warnings_pymfe),
        )

        proceso.start()
        proceso.join(timeout=timeout)

        if proceso.is_alive():
            proceso.terminate()
            proceso.join()
            raise MetaFeatureTimeoutError(
                f"Timeout: mfe.fit excedió el límite de {timeout} segundos"
            )

        try:
            status, *payload = queue.get(timeout=1)
        except Exception as e:
            raise RuntimeError("No fue posible obtener el resultado del proceso de meta-features") from e

        if status == "ok":
            nombres, valores = payload
            return nombres, valores

        raise RuntimeError(f"Error en proceso de meta-features: {payload[0]}")

    def _leer_dataset(self, ruta_absoluta, target):
        self.df = pd.read_csv(ruta_absoluta, encoding="utf-8")
        X = self.df.drop(columns=[target]).to_numpy()
        y = self.df[target].to_numpy()
        return X, y

    def _setear_variables_grupo(self, grupo, valor):
        grupos_meta_features = {
            "landmarking": {
                "best_node.mean": valor,
                "best_node.sd": valor,
                "elite_nn.mean": valor,
                "elite_nn.sd": valor,
                "linear_discr.mean": valor,
                "linear_discr.sd": valor,
                "naive_bayes.mean": valor,
                "naive_bayes.sd": valor,
                "one_nn.mean": valor,
                "one_nn.sd": valor,
                "random_node.mean": valor,
                "random_node.sd": valor,
                "worst_node.mean": valor,
                "worst_node.sd": valor
            },
            "general": {
                "attr_to_inst": valor,
                "cat_to_num": valor,
                "freq_class.mean": valor,
                "freq_class.sd": valor,
                "inst_to_attr": valor,
                "nr_attr": valor,
                "nr_bin": valor,
                "nr_cat": valor,
                "nr_class": valor,
                "nr_inst": valor,
                "nr_num": valor,
                "num_to_cat": valor
            },
            "statistical": {
                "can_cor.mean": valor,
                "can_cor.sd": valor,
                "cor.mean": valor,
                "cor.sd": valor,
                "cov.mean": valor,
                "cov.sd": valor,
                "eigenvalues.mean": valor,
                "eigenvalues.sd": valor,
                "g_mean.mean": valor,
                "g_mean.sd": valor,
                "gravity": valor,
                "h_mean.mean": valor,
                "h_mean.sd": valor,
                "iq_range.mean": valor,
                "iq_range.sd": valor,
                "kurtosis.mean": valor,
                "kurtosis.sd": valor,
                "lh_trace": valor,
                "mad.mean": valor,
                "mad.sd": valor,
                "max.mean": valor,
                "max.sd": valor,
                "mean.mean": valor,
                "mean.sd": valor,
                "median.mean": valor,
                "median.sd": valor,
                "min.mean": valor,
                "min.sd": valor,
                "nr_cor_attr": valor,
                "nr_disc": valor,
                "nr_norm": valor,
                "nr_outliers": valor,
                "p_trace": valor,
                "range.mean": valor,
                "range.sd": valor,
                "roy_root": valor,
                "sd.mean": valor,
                "sd.sd": valor,
                "sd_ratio": valor,
                "skewness.mean": valor,
                "skewness.sd": valor,
                "sparsity.mean": valor,
                "sparsity.sd": valor,
                "t_mean.mean": valor,
                "t_mean.sd": valor,
                "var.mean": valor,
                "var.sd": valor,
                "w_lambda": valor
            },
            "model-based": {
                "leaves": valor,
                "leaves_branch.mean": valor,
                "leaves_branch.sd": valor,
                "leaves_corrob.mean": valor,
                "leaves_corrob.sd": valor,
                "leaves_homo.mean": valor,
                "leaves_homo.sd": valor,
                "leaves_per_class.mean": valor,
                "leaves_per_class.sd": valor,
                "nodes": valor,
                "nodes_per_attr": valor,
                "nodes_per_inst": valor,
                "nodes_per_level.mean": valor,
                "nodes_per_level.sd": valor,
                "nodes_repeated.mean": valor,
                "nodes_repeated.sd": valor,
                "tree_depth.mean": valor,
                "tree_depth.sd": valor,
                "tree_imbalance.mean": valor,
                "tree_imbalance.sd": valor,
                "tree_shape.mean": valor,
                "tree_shape.sd": valor,
                "var_importance.mean": valor,
                "var_importance.sd": valor
            },
            "info-theory": {
                "attr_conc.mean": valor,
                "attr_conc.sd": valor,
                "attr_ent.mean": valor,
                "attr_ent.sd": valor,
                "class_conc.mean": valor,
                "class_conc.sd": valor,
                "class_ent": valor,
                "eq_num_attr": valor,
                "joint_ent.mean": valor,
                "joint_ent.sd": valor,
                "mut_inf.mean": valor,
                "mut_inf.sd": valor,
                "ns_ratio": valor
            },
            "relative": {
                "best_node.mean.relative": valor,
                "best_node.sd.relative": valor,
                "elite_nn.mean.relative": valor,
                "elite_nn.sd.relative": valor,
                "linear_discr.mean.relative": valor,
                "linear_discr.sd.relative": valor,
                "naive_bayes.mean.relative": valor,
                "naive_bayes.sd.relative": valor,
                "one_nn.mean.relative": valor,
                "one_nn.sd.relative": valor,
                "random_node.mean.relative": valor,
                "random_node.sd.relative": valor,
                "worst_node.mean.relative": valor,
                "worst_node.sd.relative": valor
            },
            "clustering": {
                "ch": valor,
                "int": valor,
                "nre": valor,
                "pb": valor,
                "sc": valor,
                "sil": valor,
                "vdb": valor,
                "vdu": valor
            },
            "complexity": {
                "c1": valor,
                "c2": valor,
                "cls_coef": valor,
                "density": valor,
                "f1.mean": valor,
                "f1.sd": valor,
                "f1v.mean": valor,
                "f1v.sd": valor,
                "f2.mean": valor,
                "f2.sd": valor,
                "f3.mean": valor,
                "f3.sd": valor,
                "f4.mean": valor,
                "f4.sd": valor,
                "hubs.mean": valor,
                "hubs.sd": valor,
                "l1.mean": valor,
                "l1.sd": valor,
                "l2.mean": valor,
                "l2.sd": valor,
                "l3.mean": valor,
                "l3.sd": valor,
                "lsc": valor,
                "n1": valor,
                "n2.mean": valor,
                "n2.sd": valor,
                "n3.mean": valor,
                "n3.sd": valor,
                "n4.mean": valor,
                "n4.sd": valor,
                "t1": valor,
                "t2": valor,
                "t3": valor,
                "t4": valor
            },
            "itemset": {
                "one_itemset.mean": valor,
                "one_itemset.sd": valor,
                "two_itemset.mean": valor,
                "two_itemset.sd": valor
            },
            "concept": {
                "cohesiveness.mean": valor,
                "cohesiveness.sd": valor,
                "conceptvar.mean": valor,
                "conceptvar.sd": valor,
                "impconceptvar.mean": valor,
                "impconceptvar.sd": valor,
                "wg_dist.mean": valor,
                "wg_dist.sd": valor
            }
        }

        return grupos_meta_features.get(grupo, {})

    def _mapear_meta_features(self, meta_features):
        def _transformar_valor(valor):
            if valor is None:
                return self._CONSTANTE_ERROR

            if isinstance(valor, (float, np.floating)):
                if math.isnan(valor):
                    return self._CONSTANTE_ERROR
                if math.isinf(valor):
                    return self._CONSTANTE_INFINITO
                return round(float(valor), 2)

            if isinstance(valor, (int, np.integer)):
                return round(float(valor), 2)

            return valor

        def _mapear_recursivo(obj):
            if isinstance(obj, Mapping):
                return {k: _mapear_recursivo(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_mapear_recursivo(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(_mapear_recursivo(v) for v in obj)
            else:
                return _transformar_valor(obj)

        print("OK - Meta-features mapeadas a tipos nativos y con valores de error/infinito reemplazados")
        return _mapear_recursivo(meta_features)

    def _vectorizar_meta_features(self, meta_features: dict):
        meta_features_vectorizadas = [valor for subdict in meta_features.values() for valor in subdict.values()]
        print("OK - Meta-features vectorizadas en una lista con longitud:", len(meta_features_vectorizadas))
        return meta_features_vectorizadas

    def _agregar_meta_features_personalizadas(self, ruta_absoluta, X, meta_features: dict):
        def calcular_peso_en_KB(ruta_absoluta: str) -> float:
            peso_bytes = os.path.getsize(ruta_absoluta)
            return peso_bytes / 1024

        def calcular_numero_de_columnas_con_valores_faltantes(X: np.ndarray) -> float:
            STRINGS_FALTANTES = {"", "na", "n/a", "null", "none"}
            df = pd.DataFrame(X)
            faltantes_std = df.isnull()
            faltantes_str = df.apply(
                lambda col: col.astype(str).str.strip().str.lower().isin(STRINGS_FALTANTES)
                if col.dtype == object else False
            )
            faltantes = faltantes_std | faltantes_str
            return float(faltantes.any(axis=0).sum())

        def calcular_numero_de_registros_duplicados(X: np.ndarray) -> float:
            df = pd.DataFrame(X)
            return float(df.duplicated().sum())

        meta_features['personalizadas'] = {
            "peso_kb": calcular_peso_en_KB(ruta_absoluta),
            "num_columnas_con_valores_faltantes": calcular_numero_de_columnas_con_valores_faltantes(X),
            "num_registros_duplicados": calcular_numero_de_registros_duplicados(X)
        }

        print("OK - Meta-features personalizadas calculadas y agregadas al diccionario")
        return meta_features

    def eliminar_constantes_errores(self, meta_features):
        valores_a_eliminar = {self._CONSTANTE_ERROR, self._CONSTANTE_INFINITO}
        for col, subdict in meta_features.items():
            meta_features[col] = {k: v for k, v in subdict.items() if v not in valores_a_eliminar}
        return meta_features

    def extraer_meta_features_por_columna(self, X: pd.DataFrame, y: pd.Series, meta_feature_variables=None):
        meta_features_por_columna = {}
        if meta_feature_variables is None:
            meta_feature_variables = list(self._setear_variables_grupo("statistical", 0).keys())

        for col in X.columns:
            X_col = X[[col]].to_numpy()
            try:
                ft = self._ejecutar_mfe_con_timeout(
                    mfe_kwargs={"features": meta_feature_variables},
                    X=X_col,
                    y=np.array(y) if y is not None else None,
                )
                meta_features_por_columna[col] = dict(zip(ft[0], ft[1]))
            except MetaFeatureTimeoutError:
                print(f"TIMEOUT al extraer meta-features para la columna: {col}")
            except Exception:
                print(f"ERROR al extraer meta-features para la columna: {col}")

        meta_features_por_columna = self._mapear_meta_features(meta_features_por_columna)
        meta_features_por_columna = self.eliminar_constantes_errores(meta_features_por_columna)
        return meta_features_por_columna

    def formatear_meta_features_por_columna(self, meta_features):
        df = pd.DataFrame(meta_features).T
        valores_meta_features_diccionario = df.to_dict(orient="records")
        valores_tton = encode(valores_meta_features_diccionario)

        lista_columnas = list(meta_features.keys())
        columnas_toon = encode(lista_columnas)

        texto_formateado = f"Columnas:\n{columnas_toon}\nMeta-features por columna:\n{valores_tton}"
        return texto_formateado

    def formatear_meta_features_globales(self, meta_features):
        texto_formateado = ""
        for grupo, features in meta_features.items():
            texto_formateado += f"Grupo: {grupo}\n"
            for feature, valor in features.items():
                texto_formateado += f"{feature}: {valor}\n"
        return texto_formateado

    @contextmanager
    def silenciar_warnings_pymfe(self):
        if self.warnings_pymfe:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
        else:
            yield