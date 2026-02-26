import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


class Segmentador:
    """
    Clase para segmentar datos usando K-Fold o Stratified K-Fold
    dependiendo del tipo de problema.
    """
    
    def __init__(self, n_splits=3, random_state=42):
        """
        Inicializa el segmentador.
        
        Args:
            n_splits: Número de folds
            random_state: Semilla para reproducibilidad
        """
        self.n_splits = n_splits
        self.random_state = random_state
    
    def segmentar(self, X: pd.DataFrame, y: pd.Series, tipo_problema="clasificacion"):
        """
        Segmenta los datos según el tipo de problema.
        
        Args:
            X: Características (DataFrame)
            y: Target (Series)
            tipo_problema: "clasificacion" o "regresion"
        
        Returns:
            Diccionario con formato {fold_num: {x_train, y_train, x_val, y_val}}
        """
        y_categorizada = self._categorizar_y(y)

        X_array = X.values
        y_array = y_categorizada.values
        
        folds = {}
        
        # Clasificación: Stratified K-Fold
        if tipo_problema.lower() == "clasificacion":
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )
            split_method = splitter.split(X_array, y_array)
        
        # Regresión: K-Fold
        elif tipo_problema.lower() == "regresion":
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )
            split_method = splitter.split(X_array)
        
        else:
            raise ValueError(f"Tipo de problema no soportado: {tipo_problema}")
        
        # Generar los folds
        for fold_num, (train_idx, val_idx) in enumerate(split_method, 1):
            folds[fold_num] = {
                "X_train": X.iloc[train_idx],
                "y_train": y_categorizada.iloc[train_idx],
                "X_val": X.iloc[val_idx],
                "y_val": y_categorizada.iloc[val_idx]
            }
        
        return folds

    def _categorizar_y(self, y: pd.Series) -> pd.Series:
        """Convierte y a formato adecuado para clasificación o regresión."""

        if not pd.api.types.is_numeric_dtype(y):
            return y
        
        y_copy = y.copy()
        unique_vals = np.unique(y_copy)
        val_to_int = {v: i for i, v in enumerate(unique_vals)}
        y_discrete = np.array([val_to_int[v] for v in y_copy])
        y_copy = pd.Series(y_discrete, index=y_copy.index)

        return y_copy
