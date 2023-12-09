from flask import Flask, request, jsonify
from NanHandler import NanHandlerTransformer
import pandas as pd
from joblib import load, dump
import os

app = Flask(__name__)

# Load your CatBoost model
MODEL_FILEPATH = os.path.join('models', 'usable_model.joblib')
model = load(MODEL_FILEPATH)

data_cleaner =  NanHandlerTransformer()


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
    # Make predictions
    predictions = model.predict(data_cleaner)

    # Return the predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
