from flask import Flask, request, jsonify
from NanHandler import NanHandlerTransformer
from feature_creator import FeatureCreator
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import pandas as pd
from joblib import load, dump
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the F Flask app."


# Load your CatBoost model
MODEL_FILEPATH = os.path.join('models', 'usable_model.joblib')
model = load(MODEL_FILEPATH)

data_cleaner =  NanHandlerTransformer()
feature_creator = FeatureCreator()
scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    content_type = request.content_type

    if 'csv' in content_type:
        # If the content type is CSV, read the file using Pandas
        df = pd.read_csv(request.files.get('file'))
    elif 'json' in content_type:
        # If the content type is JSON, read the JSON data using Pandas
        df = pd.read_json(request.data)
    else:
        return 'Content-Type not supported!'

    #clean the data
    df_clean = data_cleaner.transform(df)
    #feature engineering and normalisation
    df_clean_scaled = scaler.fit_transform(feature_creator.transform(df_clean))
    # Make predictions
    predictions = model.predict(df_clean_scaled)

    # Return the predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
