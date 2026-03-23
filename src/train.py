import argparse
import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Logging setup  (think of this like syslog for your ML pipeline)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
def load_and_preprocess(data_path: str) -> tuple:
    """
    Load EasyVisa.csv, clean and encode features.
    Returns X (features), y (target).
    """
    log.info("Loading data from %s", data_path)
    visa = pd.read_csv(data_path)
    data = visa.copy()

    # Fix negative employee counts (33 records)
    data["no_of_employees"] = abs(data["no_of_employees"])

    # Drop non-predictive ID column
    data.drop("case_id", axis=1, inplace=True)

    # Encode target: Certified=1, Denied=0
    data["case_status"] = data["case_status"].apply(
        lambda x: 1 if x == "Certified" else 0
    )

    X = data.drop(["case_status"], axis=1)
    y = data["case_status"]

    # One-hot encode all categoricals (same as exam3.py)
    X = pd.get_dummies(X, drop_first=True)

    log.info("Dataset shape after preprocessing: %s", X.shape)
    log.info("Class distribution:\n%s", y.value_counts(normalize=True).to_string())

    return X, y


# ─────────────────────────────────────────────
# 2. TRAIN / VAL / TEST SPLIT
#    Same 70/27/3 split as exam3.py
# ─────────────────────────────────────────────
def split_data(X, y):
    # 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )
    # 27% val, 3% test  (90/10 of the 30% temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=1, stratify=y_temp
    )
    log.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# 3. RESAMPLING  (handles class imbalance)
# ─────────────────────────────────────────────
def apply_sampling(X_train, y_train, strategy: str = "over"):
    """
    strategy: 'over'  → SMOTE
              'under' → RandomUnderSampler
              'original' → no resampling
    """
    if strategy == "over":
        log.info("Applying SMOTE oversampling …")
        X_clean = X_train.dropna()
        y_clean = y_train.loc[X_clean.index]
        sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
        X_res, y_res = sm.fit_resample(X_clean, y_clean)
        log.info("After SMOTE: %s", dict(y_res.value_counts()))

    elif strategy == "under":
        log.info("Applying RandomUnderSampler …")
        rus = RandomUnderSampler(random_state=1, sampling_strategy=1)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        log.info("After undersampling: %s", dict(y_res.value_counts()))

    else:
        log.info("No resampling — using original data")
        X_res = X_train.dropna()
        y_res = y_train.loc[X_res.index]

    return X_res, y_res


# ─────────────────────────────────────────────
# 4. MODEL FACTORY
# ─────────────────────────────────────────────
def build_model(model_name: str, tune: bool = True,
                X_train=None, y_train=None):
    """
    Return a fitted model.  Optionally run RandomizedSearchCV (tune=True).
    Final model choices mirror exam3.py tuning section.
    """
    scorer = metrics.make_scorer(metrics.f1_score)

    if model_name == "gbm":
        base = GradientBoostingClassifier(random_state=1)
        param_grid = {
            "n_estimators": [100, 125, 150, 175, 200],
            "learning_rate": [0.1, 0.05, 0.01, 0.005],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "max_features": ["sqrt", "log2", 0.3, 0.5],
        }

    elif model_name == "rf":
        base = RandomForestClassifier(random_state=1)
        param_grid = {
            "n_estimators": [50, 75, 100, 125, 150],
            "min_samples_leaf": [1, 2, 4, 5, 10],
            "max_features": ["sqrt", "log2", 0.3, 0.5, None],
            "max_samples": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }

    elif model_name == "ada":
        base = AdaBoostClassifier(random_state=1)
        param_grid = {
            "n_estimators": [50, 75, 100, 125, 150],
            "learning_rate": [1.0, 0.5, 0.1, 0.01],
            "estimator": [
                DecisionTreeClassifier(max_depth=1, random_state=1),
                DecisionTreeClassifier(max_depth=2, random_state=1),
                DecisionTreeClassifier(max_depth=3, random_state=1),
            ],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose gbm | rf | ada")

    if tune and X_train is not None:
        log.info("Running RandomizedSearchCV for %s …", model_name)
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_grid,
            n_iter=50,
            n_jobs=-1,
            scoring=scorer,
            cv=5,
            random_state=1,
        )
        search.fit(X_train, y_train)
        log.info("Best params: %s | CV F1: %.4f",
                 search.best_params_, search.best_score_)
        model = search.best_estimator_
        best_params = search.best_params_
        cv_score = search.best_score_
    else:
        model = base
        best_params = {}
        cv_score = None

    model.fit(X_train, y_train)
    return model, best_params, cv_score


# ─────────────────────────────────────────────
# 5. METRICS HELPER
# ─────────────────────────────────────────────
def compute_metrics(model, X, y, prefix: str = "") -> dict:
    pred = model.predict(X)
    return {
        f"{prefix}accuracy":  accuracy_score(y, pred),
        f"{prefix}recall":    recall_score(y, pred),
        f"{prefix}precision": precision_score(y, pred),
        f"{prefix}f1":        f1_score(y, pred),
    }


# ─────────────────────────────────────────────
# 6. MAIN TRAINING PIPELINE  (MLflow experiment)
# ─────────────────────────────────────────────
def train(
    data_path: str = "EasyVisa.csv",
    model_name: str = "gbm",
    sampling: str = "over",
    tune: bool = True,
    experiment_name: str = "EasyVisa_Visa_Prediction",
):
    # ── Set / create MLflow experiment ──────────────────────────────────────
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_name}_{sampling}") as run:
        log.info("MLflow run id: %s", run.info.run_id)

        # ── Log pipeline config as params ───────────────────────────────────
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("sampling_strategy", sampling)
        mlflow.log_param("tuning_enabled", tune)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("random_state", 1)

        # ── 1. Load & preprocess ────────────────────────────────────────────
        X, y = load_and_preprocess(data_path)

        # ── 2. Split ────────────────────────────────────────────────────────
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size",   len(X_val))
        mlflow.log_param("test_size",  len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # ── 3. Resample ─────────────────────────────────────────────────────
        X_res, y_res = apply_sampling(X_train, y_train, strategy=sampling)

        mlflow.log_param("resampled_train_size", len(X_res))

        # ── 4. Build / tune model ───────────────────────────────────────────
        model, best_params, cv_score = build_model(
            model_name, tune=tune, X_train=X_res, y_train=y_res
        )

        # Log best hyperparameters found by RandomizedSearchCV
        for k, v in best_params.items():
            # Convert complex objects (e.g. DecisionTreeClassifier) to string
            mlflow.log_param(f"best_{k}", str(v))

        if cv_score is not None:
            mlflow.log_metric("cv_best_f1", cv_score)

        # ── 5. Evaluate on all splits ───────────────────────────────────────
        train_metrics = compute_metrics(model, X_res,   y_res,   prefix="train_")
        val_metrics   = compute_metrics(model, X_val,   y_val,   prefix="val_")
        test_metrics  = compute_metrics(model, X_test,  y_test,  prefix="test_")

        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)

        log.info("─── Validation Metrics ───────────────────────────")
        for k, v in val_metrics.items():
            log.info("  %-25s %.4f", k, v)

        log.info("─── Test Metrics ─────────────────────────────────")
        for k, v in test_metrics.items():
            log.info("  %-25s %.4f", k, v)

        # ── 6. Feature importance (tree-based models) ───────────────────────
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature":    X_train.columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)

            importance_path = "/tmp/feature_importances.csv"
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path, artifact_path="reports")
            log.info("Top 5 features:\n%s", importance_df.head().to_string(index=False))

        # ── 7. Log model with input example & signature ─────────────────────
        input_example = X_val.head(5)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=f"easyvisa_{model_name}",
        )

        log.info("Model logged to MLflow artifact store.")
        log.info("Run complete → mlflow ui  to inspect results.")

    return model


# ─────────────────────────────────────────────
# 7. CLI ENTRY POINT
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="EasyVisa Visa Approval — MLflow Training Pipeline"
    )
    parser.add_argument(
        "--data-path", default="EasyVisa.csv",
        help="Path to EasyVisa.csv (default: ./EasyVisa.csv)"
    )
    parser.add_argument(
        "--model", default="gbm",
        choices=["gbm", "rf", "ada"],
        help="Model to train: gbm | rf | ada  (default: gbm)"
    )
    parser.add_argument(
        "--sampling", default="over",
        choices=["over", "under", "original"],
        help="Resampling strategy: over | under | original  (default: over)"
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip RandomizedSearchCV hyperparameter tuning"
    )
    parser.add_argument(
        "--experiment", default="EasyVisa_Visa_Prediction",
        help="MLflow experiment name"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data_path,
        model_name=args.model,
        sampling=args.sampling,
        tune=not args.no_tune,
        experiment_name=args.experiment,
    )