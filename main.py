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
    df = pd.read_csv("./data/descriptive_practice_dataset_large.csv")

    X = df.drop(columns=[target]).to_numpy()
    y = df[target].to_numpy()

    logging.getLogger("pymfe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    extractor = ExtractorMetaFeatures()
    meta_features, meta_features_vectorizadas = extractor.extraer(X, y)

    # Guardar meta-features en la base de datos
    db = BaseDeDatos()
    query = """
    INSERT INTO demo (meta_features_json_humano, meta_features_json_binario, meta_features_json_vector)
    VALUES (%s, %s, %s)
    """
    params = (json.dumps(meta_features), json.dumps(meta_features), meta_features_vectorizadas)
    db.insertar(query, params)
    db.cerrar()