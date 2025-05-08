import csv
import os
import optuna
from optuna.storages import RDBStorage
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def objective(trial, X, y):
    model_type = trial.suggest_categorical("model", ["random_forest", "logistic_regression", "svc"])

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 2, 20)
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 1e2, log=True),
            solver="liblinear"
        )
    else:
        model = SVC(
            C=trial.suggest_float("C", 1e-3, 1e2, log=True),
            kernel=trial.suggest_categorical("kernel", ["linear", "rbf"])
        )

    score = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()
    # Do NOT set the model itself as an attribute (it's not serializable)
    # We'll return the model separately later
    pass

    return score

def run_optuna(X, y, study_name="automl_study", storage_path="sqlite:///optuna_study.db"):
    from optuna.storages import RDBStorage
    storage = RDBStorage(url=storage_path)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    best_trial = study.best_trial
    best_params = best_trial.params

    # Recreate the best model
    if best_params["model"] == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        best_model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"]
        )
    elif best_params["model"] == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        best_model = LogisticRegression(C=best_params["C"], solver="liblinear")
    else:
        from sklearn.svm import SVC
        best_model = SVC(C=best_params["C"], kernel=best_params["kernel"])

    best_model.fit(X, y)

    # ✅ Log all trials to CSV
    save_trials_to_csv(study)

    # ✅ Return everything needed
    return best_model, best_trial, study

def save_trials_to_csv(study, path="results/trial_results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)

        # Collect all parameter keys from trials
        param_keys = set().union(*(t.params.keys() for t in study.trials if t.state.name == "COMPLETE"))

        # Write header
        writer.writerow(["trial_number", "model", "accuracy", *param_keys])

        # Write each trial
        for trial in study.trials:
            if trial.state.name != "COMPLETE":
                continue
            row = [
                trial.number,
                trial.params.get("model", "N/A"),
                trial.value,
                *[trial.params.get(k, "") for k in param_keys]
            ]
            writer.writerow(row)
