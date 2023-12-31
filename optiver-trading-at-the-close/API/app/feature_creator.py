
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Cette fonction doit être présente pour respecter l'API de Scikit-Learn.
        # Dans ce cas, il n'y a pas de logique de fitting, donc elle ne fait rien.
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        # Créez une copie de X pour éviter de modifier le DataFrame original
        data = X.copy()

        # Time-based Features
        data['intraday_momentum'] = data['wap'].diff()  # Change in WAP between consecutive data points
        data['time_decay'] = data['seconds_in_bucket'] / (data['seconds_in_bucket'].max() + 1)

        # Price and Volume Imbalance Features
        data['bid_ask_spread'] = data['ask_price'] - data['bid_price']
        data['imbalance_ratio'] = data['imbalance_size'] / (data['matched_size'] + 1e-9)

        # Statistical Features
        data['wap_mean'] = data['wap'].rolling(window=5, min_periods=1).mean()
        data['wap_std'] = data['wap'].rolling(window=5, min_periods =1).std()

        # Relative Price Features
        # data['price_vs_ma'] = data['wap'] / data['wap_mean']  # WAP relative to moving average
        data['price_vs_ma'] = data['wap'] / data['wap_mean'].where(data['wap_mean'] != 0, np.nan)

        # Auction Imbalance Indicators
        # Assuming imbalance_buy_sell_flag is already encoded appropriately

        # Lagged Features
        data['wap_lag_1'] = data['wap'].shift(1)

        # Non-linear Transformations
        data.loc[data['bid_size'] <= 0, 'bid_size'] = np.nan
        data.loc[data['ask_size'] <= 0, 'ask_size'] = np.nan

        data['log_bid_size'] = np.log1p(data['bid_size'])
        data['log_ask_size'] = np.log1p(data['ask_size'])

        # Remove any infinite values created by feature engineering
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        for column in data.columns:
            if data[column].isnull().all():
                # Si toutes les valeurs de la colonne sont NaN, remplir avec 0
                data[column].fillna(1, inplace=True)
            elif data[column].isnull().any():
                # Si certaines valeurs sont NaN, remplir avec la médiane
                data[column].fillna(data[column].median(), inplace=True)

        return data
