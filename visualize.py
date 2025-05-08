import optuna
import optuna.visualization as vis

def load_study():
    # Use the same logic you used in `run_optuna()`
    # We create a dummy study to visualize (re-run optimization if needed)
    def dummy_objective(trial):
        return trial.suggest_float("x", -10, 10) ** 2

    study = optuna.create_study(direction="minimize")
    study.optimize(dummy_objective, n_trials=10)
    return study

def main():
    study = load_study()  # Replace this later with your real study

    # Plot optimization history
    fig1 = vis.plot_optimization_history(study)
    fig1.show()

    # Plot parameter importance
    fig2 = vis.plot_param_importances(study)
    fig2.show()

if __name__ == "__main__":
    main()
