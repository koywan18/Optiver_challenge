from flask import Flask, request, jsonify
from NanHandler import NanHandlerTransformer
from feature_creator import FeatureCreator
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import pandas as pd
from joblib import load, dump
import os
import numpy as np

app = Flask(__name__)

# Load your CatBoost model
MODEL_FILEPATH = os.path.join('models', 'usable_model.joblib')
model = load(MODEL_FILEPATH)

data_cleaner =  NanHandlerTransformer()
feature_creator = FeatureCreator()
scaler = StandardScaler()
features = ['stock_id', 'date_id', 'seconds_in_bucket', 'imbalance_size',
       'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
       'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price',
       'ask_size', 'wap']

def convert_to_num_or_nan(cell):
    try:
        # Essayer de convertir en float
        return float(cell)
    except ValueError:
        # Si la conversion échoue, retourner NaN
        return np.nan

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Felix Flask app."

@app.route('/predict', methods=['POST'])
def predict():
    content_type = request.content_type

    if 'csv' in content_type:
        # If the content type is CSV, read the file using Pandas
        return "Inputs must me sent under json format"
    elif 'json' in content_type:
        # If the content type is JSON, read the JSON data using Pandas
        try:
            df_input = pd.DataFrame(request.json)
        except ValueError:
            try : 
                df_input = pd.DataFrame([request.json])
            except:
                return "Content-Type not supported!"
    else:
        return 'Content-Type not supported!'
    
    df_input = df_input.applymap(convert_to_num_or_nan)

    df = df_input[features]    
    #clean the data
    df_clean = data_cleaner.transform(df)

    #feature engineering and normalisation
    df_clean_scaled = feature_creator.transform(df_clean)
    
    #return df_clean_scaled.to_json()
    # # # Make predictions
    predictions = model.predict(df_clean_scaled)
    # # # Return the predictions as JSON
    return jsonify({'prediction' : predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
