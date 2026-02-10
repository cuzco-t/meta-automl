import os
import logging
import warnings
import pandas as pd
from src.ExtractorMetaFeatures import ExtractorMetaFeatures

target = "purchase_amount_usd"
df = pd.read_csv("./data/descriptive_practice_dataset_large.csv")

X = df.drop(columns=[target]).to_numpy()
y = df[target].to_numpy()

logging.getLogger("pymfe").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

extractor = ExtractorMetaFeatures()
meta_features, meta_features_aplanadas = extractor.extraer(X, y)


print(f"Meta-features extraídas: {len(meta_features)} dimensiones.")
print(meta_features)

print(f"\nMeta-features aplanadas extraídas: {len(meta_features_aplanadas)} dimensiones.")
print(meta_features_aplanadas)