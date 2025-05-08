import optuna
import optuna.visualization as vis

def main():
    study = optuna.load_study(
        study_name="automl_study",
        storage="sqlite:///optuna_study.db"
    )

    vis.plot_optimization_history(study).show()
    vis.plot_param_importances(study).show()

if __name__ == "__main__":
    main()
