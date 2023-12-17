from flask import Flask, request, jsonify
from NanHandler import NanHandlerTransformer
from feature_creator import FeatureCreator
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import pandas as pd
from joblib import load, dump
import os
import numpy as np
import warnings



app = Flask(__name__)


# Load your CatBoost model
MODEL_FILEPATH = os.path.join('models', 'usable_model.joblib')
model = load(MODEL_FILEPATH)

data_cleaner =  NanHandlerTransformer()
feature_creator = FeatureCreator()
scaler = StandardScaler()
class InvalidInputError(Exception):
    """ Exception personnalisée pour une entrée invalide. """
    pass

features = ['stock_id', 'date_id', 'seconds_in_bucket', 'imbalance_size',
       'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
       'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price',
       'ask_size', 'wap']


def convert_to_num_or_nan(cell):
    # Vérifier si la cellule est déjà un float
    if isinstance(cell, float):
        return cell

    try:
        return float(cell)
    except :
        # Lever une exception si la conversion échoue
        raise InvalidInputError("Invalid input format")
    

def validate_json_input(json_data, features):
    # Iterate over each feature and check if the key exists and is not empty
    for feature in features:
        if feature not in json_data:
            return False, f"Missing value for feature: {feature}"
        elif json_data[feature] is None or json_data[feature] == '':
            return False, f" empty value for feature: {feature}"
    return True, "Valid input"

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Felix Flask app."

@app.route('/predict', methods=['POST'])
def predict():
    content_type = request.content_type
    if 'json' not in content_type:
        # If the content type is CSV, read the file using Pandas
        return jsonify({'error' :'invalid input'})
    else:
        # If the content type is JSON, read the JSON data using Pandas
        try:
            json_data = request.json
        except:
            return jsonify({'error' :'invalid input'})
        is_valid, message = validate_json_input(json_data, features)
        if not is_valid:
            return jsonify({'error': message})

        df_input = pd.DataFrame([request.json])

    
    df = df_input[features]
    try:
        df_input = df_input.applymap(convert_to_num_or_nan)
    except InvalidInputError as e:
        return jsonify({'error' :'invalid input'})

  
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
