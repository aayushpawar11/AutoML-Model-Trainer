from automl_trainer.data_preprocessing import load_and_preprocess
from automl_trainer.model_search import run_optuna
from automl_trainer.save_model import save_model

def main(csv_path):
    print("[INFO] Loading and preprocessing data...")
    preprocessor, X, y = load_and_preprocess(csv_path)
    X_transformed = preprocessor.fit_transform(X)

    print("[INFO] Running model search with Optuna...")
    best_model, best_trial, study = run_optuna(X_transformed, y)

    print("[INFO] Saving the best model...")
    save_model(best_model, "best_model.pkl")

    print("[DONE] Best model:", best_trial.params)
    print("Accuracy:", best_trial.value)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to dataset CSV")
    args = parser.parse_args()
    main(args.csv)
