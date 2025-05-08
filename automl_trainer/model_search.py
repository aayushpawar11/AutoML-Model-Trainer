import optuna
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
    trial.set_user_attr("model_object", model)
    return score

def run_optuna(X, y):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
    
    best_model = study.best_trial.user_attrs["model_object"]
    best_model.fit(X, y)

    return best_model, study.best_trial
