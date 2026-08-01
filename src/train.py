import yaml
import mlflow
import mlflow.xgboost
import optuna
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from preprocess import load_and_clean_data, save_feature_columns
import shutil
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

X_train, X_test, y_train, y_test = load_and_clean_data(
    data_path="../" + config["data_path"],
    target_column=config["target_column"],
    test_size=config["test_size"],
    random_state=config["random_state"],
)
save_feature_columns(X_train)

# Balance the training set only. NEVER apply SMOTE to X_test — the test
# set must stay "real" data, otherwise your evaluation metrics become
# meaningless (you'd be testing partly on fake, synthetic examples).
smote = SMOTE(random_state=config["random_state"])
X_train, y_train = smote.fit_resample(X_train, y_train)

mlflow.set_experiment("churn-prediction")


def objective(trial):
    # Optuna calls this function once per trial, each time picking new
    # values from the ranges below based on what worked well before.
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)

    with mlflow.start_run(nested=True):
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=config["random_state"],
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        })
        mlflow.log_metric("roc_auc", auc)

    return auc


with mlflow.start_run(run_name="optuna-search"):
    study = optuna.create_study(direction=config["optuna"]["direction"])
    study.optimize(objective, n_trials=config["optuna"]["n_trials"])

    print("Best params:", study.best_params)
    print("Best AUC:", study.best_value)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_auc", study.best_value)

    # Retrain ONE final model using the winning params from the search.
    # We do this as its own nested run so it's clearly tagged as
    # "the model we actually chose," separate from the 20 search trials.
    with mlflow.start_run(run_name="best-model", nested=True):
        best_params = study.best_params
        final_model = XGBClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            random_state=config["random_state"],
            eval_metric="logloss",
        )
        final_model.fit(X_train, y_train)

        predictions = final_model.predict(X_test)
        probabilities = final_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.xgboost.log_model(final_model, "model")

        # Save a fixed copy for the API to load — FIXED: use final_model,
        # not "model" (which only exists inside objective()).
        save_path = "../models/churn_model"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        mlflow.xgboost.save_model(final_model, save_path)

        print(f"Final model — Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")