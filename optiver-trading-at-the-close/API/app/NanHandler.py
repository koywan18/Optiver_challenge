import numpy as np
import pandas as pd

class NanHandlerTransformer():
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        for column in X_copy.columns:
            if X_copy[column].isnull().all():
                # Si toutes les valeurs de la colonne sont NaN, remplir avec 0
                X_copy[column].fillna(1, inplace=True)
            elif X_copy[column].isnull().any():
                # Si certaines valeurs sont NaN, remplir avec la médiane
                X_copy[column].fillna(X_copy[column].median(), inplace=True)
        return X_copy
