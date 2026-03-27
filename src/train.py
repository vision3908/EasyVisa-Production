"""
train.py — EasyVisa Visa Approval Prediction
MLflow Pipeline-Ready Training Script
Converted from UT Austin AI/ML coursework (exam3.py)

WHAT THIS SCRIPT DOES (end-to-end):
  1. Loads EasyVisa.csv and cleans the data
  2. Splits into train / validation / test sets (70/27/3)
  3. Handles class imbalance with SMOTE or undersampling
  4. Trains a model with RandomizedSearchCV hyperparameter tuning
  5. Logs ALL params, metrics, and the model itself to MLflow
  6. Registers the model in MLflow Model Registry under a named version
     → api.py and Dockerfile pull directly from this registry

CLI USAGE:
    python src/train.py                          # GBM + SMOTE (best defaults)
    python src/train.py --model rf               # Random Forest + undersampling
    python src/train.py --model ada              # AdaBoost + SMOTE
    python src/train.py --sampling under         # Use undersampling instead of SMOTE
    python src/train.py --sampling original      # No resampling (for comparison)
    python src/train.py --no-tune               # Skip hyperparameter search (faster)
    python src/train.py --data-path data/EasyVisa.csv  # Explicit data path

AFTER RUNNING:
    mlflow ui                  # Launch dashboard → http://localhost:5000
    # Go to "Models" tab → see registered model ready for api.py to load
"""

# ─── Standard library imports ────────────────────────────────────────────────
import argparse       # Parses CLI flags (--model, --sampling, etc.)
import logging        # Structured logs — same pattern as Linux syslog
import os             # File path checks
import warnings       # Suppress noisy sklearn deprecation warnings

# ─── Third-party imports ──────────────────────────────────────────────────────
import mlflow                  # Experiment tracker + model registry
import mlflow.sklearn          # MLflow's scikit-learn integration
import numpy as np             # Array math (used internally by sklearn)
import pandas as pd            # DataFrame manipulation

# Resampling — handles class imbalance (more "Certified" than "Denied" in data)
from imblearn.over_sampling import SMOTE              # Synthetic minority oversampling
from imblearn.under_sampling import RandomUnderSampler # Majority class downsampling

# sklearn metrics utilities
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,   # % correct predictions overall
    f1_score,         # Harmonic mean of precision and recall — our primary metric
    precision_score,  # Of all "Certified" predictions, how many were actually Certified
    recall_score,     # Of all actual Certified cases, how many did we catch
)

# Model selection
from sklearn.model_selection import (
    RandomizedSearchCV,  # Efficient hyperparameter search (better than GridSearchCV)
    StratifiedKFold,     # K-fold that preserves class ratio in each fold
    cross_val_score,     # Cross-validation scoring
    train_test_split,    # Standard train/test split
)

# Classifiers we compare
from sklearn.ensemble import (
    AdaBoostClassifier,         # Boosting via weighted weak learners
    GradientBoostingClassifier, # Sequential boosting — our best model (GBM)
    RandomForestClassifier,     # Parallel bagging of decision trees
)
from sklearn.tree import DecisionTreeClassifier  # Base learner for AdaBoost

warnings.filterwarnings("ignore")  # Keep terminal output clean

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# Think of this like configuring rsyslog — structured, timestamped log lines
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,                              # INFO and above goes to terminal
    format="%(asctime)s  %(levelname)s  %(message)s",  # Timestamp + level + message
)
log = logging.getLogger(__name__)  # Logger scoped to this module


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(data_path: str) -> tuple:
    """
    Load EasyVisa.csv, clean the data, and encode features.

    Key decisions (matching your coursework exactly):
      - Negative employee counts are corrected with abs()  [33 bad records]
      - case_id is dropped (it's just a row number, not a real feature)
      - Target is binary encoded: Certified=1, Denied=0
      - All categorical features are one-hot encoded with get_dummies
        drop_first=True avoids the dummy variable trap (multicollinearity)

    Returns:
        X : pd.DataFrame — feature matrix
        y : pd.Series    — target labels (0/1)
    """
    log.info("Loading data from %s", data_path)
    visa = pd.read_csv(data_path)
    data = visa.copy()  # Work on a copy — never mutate the raw data

    # Fix data quality issue: 33 records have negative employee counts
    # abs() converts -500 → 500 (the sign was a data entry error)
    data["no_of_employees"] = abs(data["no_of_employees"])

    # Drop the case_id column — it's a unique identifier, not a predictive feature
    # Including it would cause the model to memorize IDs instead of learning patterns
    data.drop("case_id", axis=1, inplace=True)

    # Encode target variable: Certified → 1, Denied → 0
    # This is binary classification — 1 = the outcome we care about (visa approved)
    data["case_status"] = data["case_status"].apply(
        lambda x: 1 if x == "Certified" else 0
    )

    # Separate features (X) from target (y)
    X = data.drop(["case_status"], axis=1)
    y = data["case_status"]

    # One-hot encode ALL categorical columns at once
    # drop_first=True: for a column with N categories, creates N-1 dummies
    # This prevents perfect multicollinearity (two columns summing to 1 always)
    X = pd.get_dummies(X, drop_first=True)

    log.info("Dataset shape after preprocessing: %s", X.shape)
    log.info("Class distribution:\n%s", y.value_counts(normalize=True).to_string())

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TRAIN / VALIDATION / TEST SPLIT
# Matches the 70/27/3 split from your exam3.py coursework
# ─────────────────────────────────────────────────────────────────────────────
def split_data(X, y):
    """
    Three-way split: 70% train | 27% validation | 3% test

    Why three sets?
      - Train:      What the model learns from
      - Validation: Tune hyperparameters against this (not the test set!)
      - Test:       Final honest evaluation — touch it ONCE at the very end

    stratify=y ensures each split has the same Certified/Denied ratio as the full dataset.
    random_state=1 ensures reproducibility — same split every run.
    """
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )
    # Second split: 90% of temp → val (27% total), 10% of temp → test (3% total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=1, stratify=y_temp
    )
    log.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RESAMPLING — Handle Class Imbalance
# The EasyVisa dataset has more "Certified" than "Denied" cases
# Without resampling, the model learns to mostly predict "Certified"
# ─────────────────────────────────────────────────────────────────────────────
def apply_sampling(X_train, y_train, strategy: str = "over"):
    """
    Balance the training set using one of three strategies:

    'over'     → SMOTE: Creates SYNTHETIC new minority class samples
                 by interpolating between existing minority examples
                 + Adds information (doesn't throw data away)
                 - Slightly slower, may create noisy synthetic samples

    'under'    → RandomUnderSampler: Randomly REMOVES majority class samples
                 + Fast and simple
                 - Throws away real data (information loss)

    'original' → No resampling at all
                 Use this as a baseline to see how much resampling helps

    IMPORTANT: We only resample the TRAINING set.
    Validation and test sets stay imbalanced (they reflect real-world distribution).
    """
    if strategy == "over":
        log.info("Applying SMOTE oversampling …")
        # dropna() first — SMOTE fails if there are NaN values
        X_clean = X_train.dropna()
        y_clean = y_train.loc[X_clean.index]
        # sampling_strategy=1 → balance classes to 1:1 ratio
        # k_neighbors=5 → use 5 nearest neighbors to generate synthetic samples
        sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
        X_res, y_res = sm.fit_resample(X_clean, y_clean)
        log.info("After SMOTE: %s", dict(y_res.value_counts()))

    elif strategy == "under":
        log.info("Applying RandomUnderSampler …")
        # sampling_strategy=1 → reduce majority class to match minority class size
        rus = RandomUnderSampler(random_state=1, sampling_strategy=1)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        log.info("After undersampling: %s", dict(y_res.value_counts()))

    else:
        log.info("No resampling — using original data")
        X_res = X_train.dropna()
        y_res = y_train.loc[X_res.index]

    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MODEL FACTORY
# Builds and tunes one of three classifiers based on CLI argument
# ─────────────────────────────────────────────────────────────────────────────
def build_model(model_name: str, tune: bool = True,
                X_train=None, y_train=None):
    """
    Return a fitted model. Optionally runs RandomizedSearchCV tuning.

    RandomizedSearchCV vs GridSearchCV:
      - Grid: tries EVERY combination → very slow for large grids
      - Randomized: samples n_iter random combinations → fast + often finds near-optimal

    With n_iter=50 and cv=5: 50 random hyperparameter combos × 5 folds = 250 fits
    This is far more practical than GridSearchCV which might do 10,000+ fits.

    Model choices:
      'gbm' → GradientBoostingClassifier — builds trees SEQUENTIALLY,
              each tree corrects errors from the previous one.
              Best F1 on EasyVisa. DEFAULT.

      'rf'  → RandomForestClassifier — builds trees in PARALLEL,
              each on a random subset of data/features. Less overfit-prone.
              Best with undersampling.

      'ada' → AdaBoostClassifier — focuses on misclassified samples each round.
              Best with SMOTE oversampling.
    """
    scorer = metrics.make_scorer(metrics.f1_score)  # Optimize for F1, not accuracy

    if model_name == "gbm":
        base = GradientBoostingClassifier(random_state=1)
        param_grid = {
            "n_estimators":  [100, 125, 150, 175, 200],  # How many boosting rounds
            "learning_rate": [0.1, 0.05, 0.01, 0.005],  # Step size per round (shrinkage)
            "subsample":     [0.7, 0.8, 0.9, 1.0],       # Fraction of samples per tree
            "max_features":  ["sqrt", "log2", 0.3, 0.5], # Features per split
        }

    elif model_name == "rf":
        base = RandomForestClassifier(random_state=1)
        param_grid = {
            "n_estimators":     [50, 75, 100, 125, 150],    # Number of trees
            "min_samples_leaf": [1, 2, 4, 5, 10],           # Min samples at leaf node
            "max_features":     ["sqrt", "log2", 0.3, 0.5, None],  # Features per split
            "max_samples":      [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Bagging fraction
        }

    elif model_name == "ada":
        base = AdaBoostClassifier(random_state=1)
        param_grid = {
            "n_estimators":  [50, 75, 100, 125, 150],       # Boosting rounds
            "learning_rate": [1.0, 0.5, 0.1, 0.01],         # Contribution per learner
            "estimator":     [                               # Weak learner depth
                DecisionTreeClassifier(max_depth=1, random_state=1),  # "stump"
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
            n_iter=50,       # Try 50 random hyperparameter combos
            n_jobs=-1,       # Use all CPU cores in parallel
            scoring=scorer,  # Score each combo by F1
            cv=5,            # 5-fold cross-validation
            random_state=1,  # Reproducibility
        )
        search.fit(X_train, y_train)
        log.info("Best params: %s | CV F1: %.4f",
                 search.best_params_, search.best_score_)
        model = search.best_estimator_
        best_params = search.best_params_
        cv_score = search.best_score_
    else:
        # Skip tuning (--no-tune flag) — use base model with default params
        model = base
        best_params = {}
        cv_score = None

    # Final fit on the full (resampled) training set
    model.fit(X_train, y_train)
    return model, best_params, cv_score


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: METRICS HELPER
# Computes all four classification metrics in one call
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(model, X, y, prefix: str = "") -> dict:
    """
    Compute accuracy, recall, precision, and F1 for a given split.
    prefix lets us log train_ / val_ / test_ metrics separately in MLflow.
    
    Why F1 is our primary metric:
      - Accuracy is misleading on imbalanced data
        (predicting all "Certified" gives 67% accuracy but catches zero "Denied")
      - F1 balances precision and recall — it penalizes both false positives and negatives
    """
    pred = model.predict(X)
    return {
        f"{prefix}accuracy":  accuracy_score(y, pred),   # Overall correct %
        f"{prefix}recall":    recall_score(y, pred),     # Sensitivity (true positive rate)
        f"{prefix}precision": precision_score(y, pred),  # Positive predictive value
        f"{prefix}f1":        f1_score(y, pred),         # Primary metric
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: MAIN TRAINING PIPELINE
# This is the orchestrator — calls all the above functions in order,
# wraps everything inside a single MLflow run, and registers the model
# ─────────────────────────────────────────────────────────────────────────────
def train(
    data_path: str = "EasyVisa.csv",
    model_name: str = "gbm",
    sampling: str = "over",
    tune: bool = True,
    experiment_name: str = "EasyVisa_Visa_Prediction",
):
    """
    Full training pipeline. All params and metrics are logged to MLflow.
    The final model is registered in MLflow Model Registry.

    MLflow Model Registry is what makes api.py "MLflow-native":
      - Instead of loading a .pkl file, api.py calls:
            mlflow.sklearn.load_model("models:/easyvisa_gbm/latest")
      - This always loads the LATEST registered model version
      - When you retrain and register a new version, api.py picks it up
        automatically on next restart — no manual file copying needed
    """
    # Create or reuse the named experiment
    # All runs from this script will appear grouped under this experiment in the UI
    mlflow.set_experiment(experiment_name)

    # mlflow.start_run() opens a "run" — a single training attempt
    # Everything logged inside this block is attached to this run
    with mlflow.start_run(run_name=f"{model_name}_{sampling}") as run:
        log.info("MLflow run id: %s", run.info.run_id)

        # ── Log pipeline configuration as PARAMETERS ─────────────────────────
        # Parameters = your inputs / decisions before training starts
        mlflow.log_param("model_name",         model_name)
        mlflow.log_param("sampling_strategy",  sampling)
        mlflow.log_param("tuning_enabled",     tune)
        mlflow.log_param("data_path",          data_path)
        mlflow.log_param("random_state",       1)

        # ── Step 1: Load and preprocess data ──────────────────────────────────
        X, y = load_and_preprocess(data_path)

        # ── Step 2: Split into train / val / test ─────────────────────────────
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Log split sizes — useful for reproducibility audits
        mlflow.log_param("train_size",  len(X_train))
        mlflow.log_param("val_size",    len(X_val))
        mlflow.log_param("test_size",   len(X_test))
        mlflow.log_param("n_features",  X_train.shape[1])

        # ── Step 3: Resample training set only ────────────────────────────────
        # NEVER resample val or test — they must reflect real-world distribution
        X_res, y_res = apply_sampling(X_train, y_train, strategy=sampling)

        mlflow.log_param("resampled_train_size", len(X_res))

        # ── Step 4: Build and tune model ──────────────────────────────────────
        model, best_params, cv_score = build_model(
            model_name, tune=tune, X_train=X_res, y_train=y_res
        )

        # Log best hyperparameters found by RandomizedSearchCV
        for k, v in best_params.items():
            # Some values (like DecisionTreeClassifier objects) need str() conversion
            mlflow.log_param(f"best_{k}", str(v))

        if cv_score is not None:
            mlflow.log_metric("cv_best_f1", cv_score)  # CV score on training folds

        # ── Step 5: Evaluate on all three splits ──────────────────────────────
        # train_ metrics: checks for underfitting (too low = model didn't learn)
        # val_   metrics: checks for overfitting (much lower than train = overfit)
        # test_  metrics: the honest final score (only look at this at the very end)
        train_metrics = compute_metrics(model, X_res,  y_res,  prefix="train_")
        val_metrics   = compute_metrics(model, X_val,  y_val,  prefix="val_")
        test_metrics  = compute_metrics(model, X_test, y_test, prefix="test_")

        # Log all 12 metrics (4 metrics × 3 splits) to MLflow in one call
        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)

        log.info("─── Validation Metrics ───────────────────────────")
        for k, v in val_metrics.items():
            log.info("  %-25s %.4f", k, v)

        log.info("─── Test Metrics ─────────────────────────────────")
        for k, v in test_metrics.items():
            log.info("  %-25s %.4f", k, v)

        # ── Step 6: Save feature importances as a CSV artifact ────────────────
        # Artifacts are files attached to an MLflow run (reports, plots, configs)
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature":    X_train.columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)

            # Use a relative path (not /tmp/) — works on Windows and Linux
            importance_path = "feature_importances.csv"
            importance_df.to_csv(importance_path, index=False)
            # artifact_path="reports" places this file in a "reports/" subfolder
            # inside the MLflow run's artifact store
            mlflow.log_artifact(importance_path, artifact_path="reports")
            log.info("Top 5 features:\n%s", importance_df.head().to_string(index=False))

        # ── Step 7: Register model in MLflow Model Registry ───────────────────
        # This is the KEY step that makes api.py and Dockerfile MLflow-native.
        #
        # mlflow.sklearn.log_model does TWO things:
        #   1. Saves the model as an artifact in this run's folder
        #   2. registered_model_name → creates/updates a named model in the Registry
        #
        # After this runs, in the MLflow UI:
        #   - Go to "Models" tab → see "easyvisa_gbm"
        #   - Each training run creates a new version (Version 1, Version 2, etc.)
        #   - You can promote a version to "Staging" or "Production" stage
        #
        # api.py loads with: mlflow.sklearn.load_model("models:/easyvisa_gbm/latest")
        # Docker loads with the same URI via MLFLOW_MODEL_URI env var
        input_example = X_val.head(5)  # Used by MLflow to auto-generate model signature

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",               # Folder name inside the run's artifacts
            input_example=input_example,         # MLflow generates input/output schema
            registered_model_name=f"easyvisa_{model_name}",  # ← Registry name
        )

        log.info("Model registered in MLflow Model Registry as: easyvisa_%s", model_name)
        log.info("Run complete. Launch UI: mlflow ui → http://localhost:5000")
        log.info("Go to 'Models' tab to see the registered model and promote versions.")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: CLI ENTRY POINT
# Parses command-line arguments and calls train()
# This is what you run from the terminal: python src/train.py --model gbm
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="EasyVisa Visa Approval — MLflow Training Pipeline"
    )
    # --data-path: where is the CSV file?
    parser.add_argument(
        "--data-path", default="EasyVisa.csv",
        help="Path to EasyVisa.csv (default: ./EasyVisa.csv)"
    )
    # --model: which classifier to train?
    parser.add_argument(
        "--model", default="gbm",
        choices=["gbm", "rf", "ada"],
        help="Model to train: gbm | rf | ada  (default: gbm)"
    )
    # --sampling: how to handle class imbalance?
    parser.add_argument(
        "--sampling", default="over",
        choices=["over", "under", "original"],
        help="Resampling strategy: over | under | original  (default: over)"
    )
    # --no-tune: skip RandomizedSearchCV (useful for quick test runs)
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip RandomizedSearchCV hyperparameter tuning (faster, lower performance)"
    )
    # --experiment: MLflow experiment name (groups runs in the UI)
    parser.add_argument(
        "--experiment", default="EasyVisa_Visa_Prediction",
        help="MLflow experiment name (default: EasyVisa_Visa_Prediction)"
    )
    return parser.parse_args()


# Standard Python entry point guard
# __name__ == "__main__" only when you run this file directly
# When another module imports train.py, this block is skipped
if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data_path,
        model_name=args.model,
        sampling=args.sampling,
        tune=not args.no_tune,       # --no-tune flag inverts to tune=False
        experiment_name=args.experiment,
    )