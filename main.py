import os
import json
import logging
import warnings

from src.MineroDePipelines import MineroDePipelines
from src.ExtractorMetaFeatures import ExtractorMetaFeatures
from src.BaseDeDatos import BaseDeDatos

def silenciar_warnings():
    warnings.filterwarnings("ignore")

def silenciar_logs():
    logging.getLogger("pymfe").setLevel(logging.ERROR)

def obtener_ruta_absoluta(ruta_relativa):
    return os.path.abspath(ruta_relativa)

def extraer_meta_features(ruta_absoluta, target):
    extractor = ExtractorMetaFeatures()
    return extractor.extraer(ruta_absoluta, target)

def guardar_meta_features_en_db(meta_features, meta_features_vectorizadas):
    db = BaseDeDatos()
    query = """
    INSERT INTO demo (meta_features_json_humano, meta_features_json_binario, meta_features_json_vector)
    VALUES (%s, %s, %s)
    """
    params = (json.dumps(meta_features), json.dumps(meta_features), meta_features_vectorizadas)
    db.insertar(query, params)
    db.cerrar()

if __name__ == "__main__":
    target = "purchase_amount_usd"
    ruta_relativa = "./data/descriptive_practice_dataset_large.csv"
    tarea = "regresion"
    
    silenciar_logs()
    silenciar_warnings()

    # Parte 1: Extraer meta-features
    ruta_absoluta = obtener_ruta_absoluta(ruta_relativa)
    # meta_features, meta_features_vectorizadas = extraer_meta_features(ruta_absoluta, target)

    # Parte 2: Minar pipelines
    minero = MineroDePipelines()
    # minero.construir_pipeline_clasificacion(ruta_absoluta, target)
    minero.construir_pipeline_regresion(ruta_absoluta, target)

    # Guardar meta-features en la base de datos
    # guardar_meta_features_en_db(meta_features, meta_features_vectorizadas)