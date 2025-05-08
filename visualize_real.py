from automl_trainer.data_preprocessing import load_and_preprocess
from automl_trainer.model_search import run_optuna
import optuna.visualization as vis

def main():
    # Load your dataset (same one used in training)
    preprocessor, X, y = load_and_preprocess("sample_data/sample.csv")
    X_transformed = preprocessor.fit_transform(X)

    # Run AutoML and get the real study
    _, _, study = run_optuna(X_transformed, y)

    # Visualization: Trial Accuracy
    vis.plot_optimization_history(study).show()

    # Visualization: Hyperparameter Importance
    vis.plot_param_importances(study).show()

if __name__ == "__main__":
    main()
