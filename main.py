import os
import json
import logging
import warnings
import math
import numpy as np
import pandas as pd

from src.ExtractorMetaFeatures import ExtractorMetaFeatures
from src.BaseDeDatos import BaseDeDatos

if __name__ == "__main__":
    target = "purchase_amount_usd"
    ruta_relativa = "./data/descriptive_practice_dataset_large.csv"
    
    ruta_absoluta = os.path.abspath(ruta_relativa)

    logging.getLogger("pymfe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    extractor = ExtractorMetaFeatures()
    meta_features, meta_features_vectorizadas = extractor.extraer(ruta_absoluta, target)

    # Guardar meta-features en la base de datos
    db = BaseDeDatos()
    query = """
    INSERT INTO demo (meta_features_json_humano, meta_features_json_binario, meta_features_json_vector)
    VALUES (%s, %s, %s)
    """
    params = (json.dumps(meta_features), json.dumps(meta_features), meta_features_vectorizadas)
    db.insertar(query, params)
    db.cerrar()