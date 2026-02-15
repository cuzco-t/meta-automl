import json
import logging
import warnings
import pandas as pd

from pathlib import Path

from src.MineroDePipelines import MineroDePipelines
from src.ExtractorMetaFeatures import ExtractorMetaFeatures
from src.BaseDeDatos import BaseDeDatos

def silenciar_warnings():
    warnings.filterwarnings("ignore")

def silenciar_logs():
    logging.getLogger("pymfe").setLevel(logging.ERROR)

def generador_csv(carpeta_datasets):
    carpeta = Path(carpeta_datasets)
    for archivo in carpeta.glob("*.csv"):
        yield archivo.resolve()

def main():
    PRIMERA_EJECUCION = True
    carpeta_datasets = "./data"
    target = "purchase_amount_usd"
    tarea = "regresion"

    silenciar_logs()
    silenciar_warnings()

    extractor = ExtractorMetaFeatures()
    minero = MineroDePipelines()
    db = BaseDeDatos()

    for ruta_absoluta_csv in generador_csv(carpeta_datasets):
        print(f"Procesando dataset: {ruta_absoluta_csv}...")
        df_main = pd.read_csv(ruta_absoluta_csv)

        X_df = df_main.drop(columns=[target])
        y_df = df_main[target]

        # meta_features, meta_features_vectorizadas = extractor.extraer_desde_dataframe(
        #     X_df.copy(), 
        #     y_df.copy(), 
        #     vectorizar=True
        # )

        # db.guardar_meta_features_globales(json.dumps(meta_features), meta_features_vectorizadas)
    
        minero.construir_pipeline_regresion(X_df, y_df)

        if PRIMERA_EJECUCION:
            break

    db.cerrar()


if __name__ == "__main__":
    main()