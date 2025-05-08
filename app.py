from flask import Flask, request, jsonify
import pandas as pd
from automl_trainer.data_preprocessing import load_and_preprocess
from automl_trainer.model_search import run_optuna
from automl_trainer.save_model import save_model

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({'error': 'CSV file is missing'}), 400


    try:
        file = request.files['file']    
        df = pd.read_csv(file)
        preprocessor, X, y = load_and_preprocess(df)
        X_transformed = preprocessor.fit_transform(X)

        best_model, best_trial, _ = run_optuna(X_transformed, y)
        save_model(best_model, "best_model.pkl")

        return jsonify({
            "model": best_trial.params["model"],
            "accuracy": best_trial.value,
            "hyperparameters": best_trial.params
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
