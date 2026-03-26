"""
train.py — EasyVisa Visa Approval Prediction
MLflow Pipeline-Ready Training Script
*** ANNOTATED VERSION — Every line explained ***

Think of this file like a Linux init script for your ML pipeline:
  - It boots up in a defined order
  - Each section has one job
  - It's designed to be run from the command line OR called by an orchestrator

Usage:
    python train.py                          # default GBM + SMOTE oversampling
    python train.py --model rf --sampling under
    python train.py --no-tune                # skip hyperparameter search (faster)
    mlflow ui                                # view results at http://localhost:5000
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# Like loading kernel modules before the system starts.
# Each import brings in a specific toolkit.
# ══════════════════════════════════════════════════════════════════════════════

import argparse          # Parses command-line flags (--model, --sampling, etc.)
import logging           # Structured log output — like syslog for your pipeline
import os                # OS utilities (file paths, env vars)
import warnings          # Suppresses noisy sklearn/imblearn deprecation warnings

import mlflow            # Core MLflow: experiment tracking, run management
import mlflow.sklearn    # MLflow's sklearn flavor: log_model(), load_model()
import numpy as np       # Numerical arrays — the "math engine" under sklearn
import pandas as pd      # DataFrames — how we load and manipulate tabular data

# imbalanced-learn: handles the 66/34 class imbalance in EasyVisa data
from imblearn.over_sampling import SMOTE           # Creates synthetic minority samples
from imblearn.under_sampling import RandomUnderSampler  # Shrinks the majority class

from sklearn import metrics  # General metrics module (used to build custom scorer)

# Ensemble classifiers — our three candidate models from exam3.py
from sklearn.ensemble import (
    AdaBoostClassifier,          # Adaptive Boosting: builds weak trees sequentially
    GradientBoostingClassifier,  # GBM: minimizes loss via gradient descent on trees
    RandomForestClassifier,      # Bagging of deep trees with random feature subsets
)

# Individual metric functions — each returns a float score
from sklearn.metrics import (
    accuracy_score,   # (TP+TN) / total — can be misleading on imbalanced data
    f1_score,         # Harmonic mean of precision & recall — our primary metric
    precision_score,  # TP / (TP+FP) — "of all predicted Certified, how many were right?"
    recall_score,     # TP / (TP+FN) — "of all actual Certified, how many did we catch?"
)

# Model selection tools
from sklearn.model_selection import (
    RandomizedSearchCV,  # Tries random combos of hyperparams (faster than GridSearch)
    StratifiedKFold,     # K-fold that preserves class ratio in each fold
    cross_val_score,     # Runs cross-validation and returns score array
    train_test_split,    # Splits arrays into random train/test subsets
)

from sklearn.tree import DecisionTreeClassifier  # Used as base estimator in AdaBoost

warnings.filterwarnings("ignore")  # Mutes sklearn FutureWarnings — keep logs clean


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOGGING SETUP
# Like configuring /etc/syslog.conf — sets the format for all log messages.
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,                          # Show INFO and above (not DEBUG)
    format="%(asctime)s  %(levelname)s  %(message)s",  # timestamp + level + message
)
log = logging.getLogger(__name__)
# __name__ is the module name ("train" or "__main__")
# log.info("...") prints: 2026-03-22 12:00:00  INFO  Loading data from EasyVisa.csv


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADING & PREPROCESSING
# Loads raw CSV, cleans it, encodes it — produces X (features) and y (target).
# Think: mounting and formatting a disk before using it.
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(data_path: str) -> tuple:
    """
    Loads EasyVisa.csv, cleans and encodes it.
    Returns:
        X : pd.DataFrame — feature matrix (all inputs)
        y : pd.Series    — binary target (1=Certified, 0=Denied)
    """
    log.info("Loading data from %s", data_path)

    visa = pd.read_csv(data_path)   # Read CSV into a DataFrame (like a spreadsheet in memory)
    data = visa.copy()              # Work on a copy — never mutate the original (defensive programming)

    # Fix data quality issue: 33 records had negative employee counts (impossible)
    # abs() takes the absolute value — e.g. -500 → 500
    data["no_of_employees"] = abs(data["no_of_employees"])

    # Drop case_id — it's a unique row identifier with zero predictive value
    # axis=1 means "drop a column" (axis=0 would drop a row)
    # inplace=True modifies data directly instead of returning a new DataFrame
    data.drop("case_id", axis=1, inplace=True)

    # Binary encode the target column: "Certified" → 1, anything else → 0
    # apply() runs a function on every value in the column
    # lambda x: ... is an anonymous (inline) function
    data["case_status"] = data["case_status"].apply(
        lambda x: 1 if x == "Certified" else 0
    )

    # Separate features (X) from target (y)
    X = data.drop(["case_status"], axis=1)  # Everything except the target
    y = data["case_status"]                  # Just the target column

    # One-hot encode all categorical columns (e.g. "continent" → continent_Asia, continent_Europe...)
    # drop_first=True removes one dummy per category to avoid multicollinearity
    # (same approach used in exam3.py)
    X = pd.get_dummies(X, drop_first=True)

    log.info("Dataset shape after preprocessing: %s", X.shape)  # e.g. (25480, 23)
    log.info("Class distribution:\n%s", y.value_counts(normalize=True).to_string())
    # normalize=True shows proportions: 0.668 Certified, 0.332 Denied

    return X, y  # Return both as a tuple


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAIN / VAL / TEST SPLIT
# Divides data into 3 non-overlapping buckets.
# Same 70/27/3 split logic as exam3.py — preserves class ratio via stratify.
# ══════════════════════════════════════════════════════════════════════════════

def split_data(X, y):
    # Step 1: 70% train, 30% temp pool
    # random_state=1 → reproducible split (same result every run)
    # stratify=y → each split has same 66/34 class ratio as the full dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )

    # Step 2: Split the 30% temp pool into 27% val + 3% test
    # test_size=0.1 of 30% = 3% of total
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=1, stratify=y_temp
    )

    log.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    # Returns 6 objects: features + labels for each split
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RESAMPLING
# Addresses the 66/34 class imbalance BEFORE training.
# Only applied to training data — val and test stay untouched (real-world ratio).
# Think: tuning the input signal before feeding it to the model.
# ══════════════════════════════════════════════════════════════════════════════

def apply_sampling(X_train, y_train, strategy: str = "over"):
    """
    strategy options:
      'over'     → SMOTE: synthesizes new minority-class samples (Denied cases)
      'under'    → RandomUnderSampler: removes majority-class samples (Certified)
      'original' → no resampling; just drops NaN rows
    """
    if strategy == "over":
        log.info("Applying SMOTE oversampling …")

        # SMOTE requires no NaN values — drop them first
        X_clean = X_train.dropna()                      # Remove rows with missing values
        y_clean = y_train.loc[X_clean.index]            # Keep only matching labels

        # SMOTE: Synthetic Minority Oversampling TEchnique
        # sampling_strategy=1 → make minority class same size as majority (1:1 ratio)
        # k_neighbors=5 → each synthetic point is interpolated from 5 nearest real points
        sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
        X_res, y_res = sm.fit_resample(X_clean, y_clean)
        # fit_resample() both fits the SMOTE model and generates the balanced dataset

        log.info("After SMOTE: %s", dict(y_res.value_counts()))  # Should show equal counts

    elif strategy == "under":
        log.info("Applying RandomUnderSampler …")

        # Randomly removes majority-class rows until classes are balanced
        # Simpler than SMOTE but loses information
        rus = RandomUnderSampler(random_state=1, sampling_strategy=1)
        X_res, y_res = rus.fit_resample(X_train, y_train)

        log.info("After undersampling: %s", dict(y_res.value_counts()))

    else:
        # No resampling — use the data as-is (just clean NaNs)
        log.info("No resampling — using original data")
        X_res = X_train.dropna()
        y_res = y_train.loc[X_res.index]

    return X_res, y_res  # Return the (possibly rebalanced) training data


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL FACTORY
# Selects model + hyperparameter search space, runs RandomizedSearchCV,
# and returns the best-fitted model.
# Think: a configurable package installer — you pick the name, it handles the rest.
# ══════════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, tune: bool = True,
                X_train=None, y_train=None):
    """
    Returns: (fitted model, best_params dict, cv_score float)
    model_name: 'gbm' | 'rf' | 'ada'
    tune: if True, runs RandomizedSearchCV; if False, uses defaults
    """

    # F1 scorer — used as the optimization objective during hyperparameter search
    # make_scorer() wraps a metric function into sklearn's cross-validation interface
    scorer = metrics.make_scorer(metrics.f1_score)

    # ── GBM: Gradient Boosting Machine ────────────────────────────────────────
    if model_name == "gbm":
        base = GradientBoostingClassifier(random_state=1)
        # Hyperparameter search space (same as exam3.py tuning section)
        param_grid = {
            "n_estimators":  [100, 125, 150, 175, 200],  # Number of boosting stages
            "learning_rate": [0.1, 0.05, 0.01, 0.005],  # Shrinks each tree's contribution
            "subsample":     [0.7, 0.8, 0.9, 1.0],      # Fraction of samples per tree
            "max_features":  ["sqrt", "log2", 0.3, 0.5], # Features considered per split
        }

    # ── Random Forest ─────────────────────────────────────────────────────────
    elif model_name == "rf":
        base = RandomForestClassifier(random_state=1)
        param_grid = {
            "n_estimators":    [50, 75, 100, 125, 150],  # Number of trees in the forest
            "min_samples_leaf":[1, 2, 4, 5, 10],         # Min samples required at a leaf
            "max_features":    ["sqrt", "log2", 0.3, 0.5, None],  # Features per split
            "max_samples":     [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],   # Bootstrap sample size
        }

    # ── AdaBoost ──────────────────────────────────────────────────────────────
    elif model_name == "ada":
        base = AdaBoostClassifier(random_state=1)
        param_grid = {
            "n_estimators":  [50, 75, 100, 125, 150],    # Number of weak learners
            "learning_rate": [1.0, 0.5, 0.1, 0.01],     # Weight applied to each classifier
            # base estimator: tries decision stumps of different depths
            "estimator": [
                DecisionTreeClassifier(max_depth=1, random_state=1),  # stump
                DecisionTreeClassifier(max_depth=2, random_state=1),  # slightly deeper
                DecisionTreeClassifier(max_depth=3, random_state=1),  # more complex
            ],
        }
    else:
        # Guard clause: crash early with a clear message if bad input given
        raise ValueError(f"Unknown model: {model_name}. Choose gbm | rf | ada")

    # ── Hyperparameter Tuning with RandomizedSearchCV ─────────────────────────
    if tune and X_train is not None:
        log.info("Running RandomizedSearchCV for %s …", model_name)

        search = RandomizedSearchCV(
            estimator=base,               # The base model to tune
            param_distributions=param_grid, # The search space
            n_iter=50,                    # Try 50 random combinations (vs. all in GridSearch)
            n_jobs=-1,                    # Use ALL CPU cores (like make -j on Linux)
            scoring=scorer,               # Optimize for F1 score
            cv=5,                         # 5-fold cross-validation for each combo
            random_state=1,               # Reproducible random sampling
        )
        search.fit(X_train, y_train)      # Run the full search

        log.info("Best params: %s | CV F1: %.4f",
                 search.best_params_, search.best_score_)

        model = search.best_estimator_   # The model with the best params
        best_params = search.best_params_ # Dict of winning hyperparameters
        cv_score = search.best_score_    # The F1 score achieved

    else:
        # No tuning: use model with default hyperparameters
        model = base
        best_params = {}
        cv_score = None

    model.fit(X_train, y_train)  # Final fit on the full (resampled) training set
    return model, best_params, cv_score


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — METRICS HELPER
# Given a fitted model + dataset, returns a dict of 4 classification metrics.
# The prefix argument lets us tag metrics as "train_", "val_", or "test_".
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(model, X, y, prefix: str = "") -> dict:
    pred = model.predict(X)  # Generate class predictions (0 or 1) for each row

    return {
        f"{prefix}accuracy":  accuracy_score(y, pred),   # Overall correct rate
        f"{prefix}recall":    recall_score(y, pred),     # Sensitivity / true positive rate
        f"{prefix}precision": precision_score(y, pred),  # Positive predictive value
        f"{prefix}f1":        f1_score(y, pred),         # Balanced precision-recall score
    }
    # Example output: {"val_accuracy": 0.87, "val_recall": 0.91, ...}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN TRAINING PIPELINE (the MLflow experiment)
# This function orchestrates ALL steps and wraps them in a tracked MLflow run.
# Everything logged here appears in `mlflow ui` as a named experiment.
# Think: the main() of a daemon process — it ties all services together.
# ══════════════════════════════════════════════════════════════════════════════

def train(
    data_path: str = "EasyVisa.csv",          # Where to read the raw data
    model_name: str = "gbm",                  # Which model to train
    sampling: str = "over",                   # Which resampling strategy to use
    tune: bool = True,                        # Whether to run hyperparameter search
    experiment_name: str = "EasyVisa_Visa_Prediction",  # MLflow experiment bucket name
):
    # Creates or reuses a named experiment in MLflow's tracking server
    # All runs under this name are grouped together in the UI
    mlflow.set_experiment(experiment_name)

    # mlflow.start_run() opens a new run — like opening a log file for this job
    # run_name is a human-readable label visible in the MLflow UI
    with mlflow.start_run(run_name=f"{model_name}_{sampling}") as run:
        log.info("MLflow run id: %s", run.info.run_id)  # Unique UUID for this run

        # ── Log config parameters ────────────────────────────────────────────
        # mlflow.log_param() saves key/value metadata — appears in the "Parameters" tab
        # These are the INPUTS that define this experiment run
        mlflow.log_param("model_name", model_name)        # e.g. "gbm"
        mlflow.log_param("sampling_strategy", sampling)   # e.g. "over"
        mlflow.log_param("tuning_enabled", tune)          # True/False
        mlflow.log_param("data_path", data_path)          # Path to CSV
        mlflow.log_param("random_state", 1)               # Ensures reproducibility

        # ── Step 1: Load and preprocess the data ─────────────────────────────
        X, y = load_and_preprocess(data_path)
        # X is now a numeric DataFrame with one-hot encoded columns
        # y is a binary Series: 1=Certified, 0=Denied

        # ── Step 2: Split into train / val / test ────────────────────────────
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Log split sizes so we can audit data volume in MLflow UI
        mlflow.log_param("train_size", len(X_train))  # ~17,836
        mlflow.log_param("val_size",   len(X_val))    # ~6,880
        mlflow.log_param("test_size",  len(X_test))   # ~764
        mlflow.log_param("n_features", X_train.shape[1])  # Number of feature columns

        # ── Step 3: Apply class balancing to training data only ──────────────
        X_res, y_res = apply_sampling(X_train, y_train, strategy=sampling)

        mlflow.log_param("resampled_train_size", len(X_res))  # May differ from train_size

        # ── Step 4: Build and tune the model ─────────────────────────────────
        model, best_params, cv_score = build_model(
            model_name, tune=tune, X_train=X_res, y_train=y_res
        )

        # Log each winning hyperparameter into MLflow
        # str(v) handles complex objects like DecisionTreeClassifier instances
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", str(v))

        # Log the cross-validation F1 score from hyperparameter search
        if cv_score is not None:
            mlflow.log_metric("cv_best_f1", cv_score)

        # ── Step 5: Evaluate model on all three splits ────────────────────────
        train_metrics = compute_metrics(model, X_res,  y_res,  prefix="train_")
        val_metrics   = compute_metrics(model, X_val,  y_val,  prefix="val_")
        test_metrics  = compute_metrics(model, X_test, y_test, prefix="test_")

        # Merge all metric dicts and send to MLflow in one call
        # {**dict1, **dict2} is Python's dict unpacking / merge syntax
        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)
        # mlflow.log_metrics() saves key/value floats — appear in "Metrics" tab

        # Print val and test results to terminal log
        log.info("─── Validation Metrics ───────────────────────────")
        for k, v in val_metrics.items():
            log.info("  %-25s %.4f", k, v)  # %-25s = left-align in 25-char field

        log.info("─── Test Metrics ─────────────────────────────────")
        for k, v in test_metrics.items():
            log.info("  %-25s %.4f", k, v)

        # ── Step 6: Save feature importances as a CSV artifact ───────────────
        # hasattr() checks if the model has a feature_importances_ attribute
        # (tree-based models do; linear models don't)
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature":    X_train.columns,          # Column names
                "importance": model.feature_importances_,  # Importance scores (sum to 1)
            }).sort_values("importance", ascending=False)  # Sort highest first

            importance_path = "/tmp/feature_importances.csv"
            importance_df.to_csv(importance_path, index=False)  # Save locally first

            # Upload CSV to MLflow's artifact store under "reports/" folder
            mlflow.log_artifact(importance_path, artifact_path="reports")

            log.info("Top 5 features:\n%s", importance_df.head().to_string(index=False))

        # ── Step 7: Log the serialized model to MLflow ───────────────────────
        input_example = X_val.head(5)  # 5-row sample to auto-infer input schema

        mlflow.sklearn.log_model(
            sk_model=model,                          # The fitted sklearn model object
            artifact_path="model",                   # Folder name in artifact store
            input_example=input_example,             # Helps MLflow infer the signature
            registered_model_name=f"easyvisa_{model_name}",  # Registers in Model Registry
        )
        # After this call, the model can be loaded from any machine with:
        # mlflow.sklearn.load_model("models:/easyvisa_gbm/latest")

        log.info("Model logged to MLflow artifact store.")
        log.info("Run complete → mlflow ui  to inspect results.")

    return model  # Return the fitted model for programmatic use


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CLI ARGUMENT PARSER
# Defines the command-line interface — like getopt() in a shell script.
# Each add_argument() defines one flag, its type, choices, and help text.
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="EasyVisa Visa Approval — MLflow Training Pipeline"
    )

    parser.add_argument(
        "--data-path", default="EasyVisa.csv",
        help="Path to EasyVisa.csv (default: ./EasyVisa.csv)"
        # Usage: python train.py --data-path /data/EasyVisa.csv
    )
    parser.add_argument(
        "--model", default="gbm",
        choices=["gbm", "rf", "ada"],   # Only these 3 values are accepted
        help="Model to train: gbm | rf | ada  (default: gbm)"
    )
    parser.add_argument(
        "--sampling", default="over",
        choices=["over", "under", "original"],
        help="Resampling strategy: over | under | original  (default: over)"
    )
    parser.add_argument(
        "--no-tune", action="store_true",  # Flag: present = True, absent = False
        help="Skip RandomizedSearchCV hyperparameter tuning"
        # Usage: python train.py --no-tune  → faster, uses default hyperparams
    )
    parser.add_argument(
        "--experiment", default="EasyVisa_Visa_Prediction",
        help="MLflow experiment name"
    )
    return parser.parse_args()  # Returns a Namespace object: args.model, args.sampling, etc.


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ENTRY POINT
# Python runs this block when the script is executed directly (not imported).
# `if __name__ == "__main__"` is like checking $0 == "script.sh" in bash.
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()   # Parse flags from sys.argv (the command line)

    train(
        data_path=args.data_path,         # --data-path value
        model_name=args.model,            # --model value
        sampling=args.sampling,           # --sampling value
        tune=not args.no_tune,            # --no-tune flips tune to False
        experiment_name=args.experiment,  # --experiment value
    )
    # The `not args.no_tune` inversion:
    # --no-tune flag present  → args.no_tune=True  → tune=False (skip tuning)
    # --no-tune flag absent   → args.no_tune=False → tune=True  (run tuning)