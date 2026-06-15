import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import logging
import time
from pathlib import Path

# === Cyclical feature processing ===
def identify_cyclical_features(feature_names):
    cyclical_pairs = {}
    sin_features = [f for f in feature_names if f.endswith('_sin')]
    for sin_feature in sin_features:
        base_name = sin_feature[:-4]
        cos_feature = f"{base_name}_cos"
        if cos_feature in feature_names:
            cyclical_pairs[base_name] = [sin_feature, cos_feature]
    return cyclical_pairs

def preprocess_features_for_analysis(X):
    feature_names = X.columns
    cyclical_pairs = identify_cyclical_features(feature_names)
    X_processed = pd.DataFrame()
    for feature in feature_names:
        is_part_of_pair = False
        for pair in cyclical_pairs.values():
            if feature in pair:
                is_part_of_pair = True
                break
        if not is_part_of_pair:
            X_processed[feature] = X[feature]
    for base_name, pair in cyclical_pairs.items():
        sin_feature, cos_feature = pair
        X_processed[base_name] = np.sqrt(X[sin_feature]**2 + X[cos_feature]**2)
    return X_processed

# ========== CONFIGURATION ========== #
CSV_PATH = r"preprocessed_data2.csv"  # Update to your dataset path
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_TRIALS = 100
RESULTS_DIR = Path("optuna_xgb_results")
RESULTS_DIR.mkdir(exist_ok=True)

# ========== LOGGING ========== #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== DATA LOADING ========== #
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'is_successful' not in df.columns:
        raise ValueError("❌ Цільова змінна 'is_successful' не знайдена!")
    X = df.drop('is_successful', axis=1)
    y = df['is_successful']
    X = X.select_dtypes(exclude=['object'])
    X_processed = preprocess_features_for_analysis(X)
    return X_processed, y

# ========== OPTUNA OBJECTIVE ========== #
def objective(trial, X_train, y_train, base_params):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 2.0, step=0.05),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.01),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.01),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.5, step=0.01),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0, step=0.05)
    }
    model_params = base_params.copy()
    model_params.update(params)
    model = XGBClassifier(**model_params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='average_precision', n_jobs=-1)
    return cv_scores.mean()

# ========== MAIN SCRIPT ========== #
def main():
    logger.info("Завантаження та підготовка даних...")
    X, y = load_data(CSV_PATH)
    feature_names = list(X.columns)
    logger.info(f"Кількість ознак: {len(feature_names)}")
    logger.info(f"Розподіл класів: {y.value_counts().to_dict()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    class_counts = np.bincount(y_train)
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0
    }
    if len(class_counts) > 1 and class_counts[1] > 0:
        base_params['scale_pos_weight'] = class_counts[0] / class_counts[1]
    logger.info("Початок оптимізації Optuna...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    start_time = time.time()
    study.optimize(lambda trial: objective(trial, X_train, y_train, base_params), n_trials=N_TRIALS, show_progress_bar=True)
    optimization_time = time.time() - start_time
    logger.info(f"Оптимізація завершена за {optimization_time:.1f} секунд")
    logger.info(f"Найкращі параметри: {study.best_params}")
    logger.info(f"Найкращий середній AUC-PR (CV): {study.best_value:.6f}")
    # Final model evaluation
    best_params = base_params.copy()
    best_params.update(study.best_params)
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'auc_pr': average_precision_score(y_test, y_pred_proba)
    }
    logger.info(f"AUC-PR на тесті: {test_metrics['auc_pr']:.6f}")
    # Save results
    results = {
        'best_params': study.best_params,
        'best_cv_aucpr': study.best_value,
        'test_metrics': test_metrics,
        'feature_names': feature_names,
        'n_trials': N_TRIALS,
        'optimization_time_sec': optimization_time
    }
    results_path = RESULTS_DIR / "optuna_xgb_best_results.json"
    pd.Series(results).to_json(results_path, force_ascii=False, indent=2)
    logger.info(f"Результати збережено у {results_path}")

if __name__ == "__main__":
    main() 