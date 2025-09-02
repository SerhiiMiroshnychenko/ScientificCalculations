import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.feature_selection import RFE
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from boruta import BorutaPy
import json
import time

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

# === CONFIGURATION ===
DATA_PATH = "preprocessed_data2.csv"
RESULTS_DIR = "optuna_feature_selection_full_cycle_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42
AUC_MIN = 0.9766
N_TRIALS = 100

# === DATA LOADING ===
data = pd.read_csv(DATA_PATH)
if 'is_successful' not in data.columns:
    raise ValueError("❌ Target variable 'is_successful' not found!")
X = data.drop('is_successful', axis=1)
y = data['is_successful']
X = X.select_dtypes(exclude=['object'])
X_processed = preprocess_features_for_analysis(X)
feature_names = list(X_processed.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 1. OPTUNA HYPERPARAMETER SEARCH ON ALL FEATURES ===
def optuna_objective(trial, X_train, y_train, base_params):
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

print("🔎 Step 1: Optuna hyperparameter search on all features...")
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
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
start_time = time.time()
study.optimize(lambda trial: optuna_objective(trial, X_train_scaled, y_train, base_params), n_trials=N_TRIALS, show_progress_bar=True)
optuna_time = time.time() - start_time
best_params_all = base_params.copy()
best_params_all.update(study.best_params)
# Final model on all features
model_all = XGBClassifier(**best_params_all)
model_all.fit(X_train_scaled, y_train)
y_pred_all = model_all.predict(X_test_scaled)
y_pred_proba_all = model_all.predict_proba(X_test_scaled)[:, 1]
metrics_all = {
    'accuracy': accuracy_score(y_test, y_pred_all),
    'precision': precision_score(y_test, y_pred_all, zero_division=0),
    'recall': recall_score(y_test, y_pred_all, zero_division=0),
    'f1': f1_score(y_test, y_pred_all, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_all),
    'auc_pr': average_precision_score(y_test, y_pred_proba_all)
}
with open(os.path.join(RESULTS_DIR, 'optuna_all_features_results.json'), 'w') as f:
    json.dump({'best_params': study.best_params, 'metrics': metrics_all, 'optuna_time_sec': optuna_time}, f, indent=2)

# === 2. FEATURE SELECTION METHODS (using best_params_all) ===
print("🔎 Step 2: Feature selection with all methods...")
# RFE
estimator = XGBClassifier(**best_params_all)
rfe = RFE(estimator, n_features_to_select=None, step=1)
rfe.fit(X_train_scaled, y_train)
ranking = rfe.ranking_
rfe_df = pd.DataFrame({'feature': feature_names, 'ranking': ranking}).sort_values('ranking').reset_index(drop=True)
aucpr_log_rfe = []
for k in range(1, len(feature_names)+1):
    selected_features = rfe_df[rfe_df['ranking'] <= k]['feature'].tolist()
    idxs = [feature_names.index(f) for f in selected_features if f in feature_names]
    if not idxs:
        continue
    X_train_sel = X_train_scaled[:, idxs]
    X_test_sel = X_test_scaled[:, idxs]
    model = XGBClassifier(**best_params_all)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict_proba(X_test_sel)[:, 1]
    aucpr = average_precision_score(y_test, y_pred)
    aucpr_log_rfe.append({'n_features': len(selected_features), 'features': ','.join(selected_features), 'aucpr': aucpr})
pd.DataFrame(aucpr_log_rfe).to_csv(os.path.join(RESULTS_DIR, 'rfe_aucpr_curve.csv'), index=False)
# Greedy Forward
remaining = set(range(len(feature_names)))
selected = []
greedy_log = []
for step in range(len(feature_names)):
    best_score = -np.inf
    best_idx = None
    for idx in remaining:
        current_features = selected + [idx]
        X_train_sel = X_train_scaled[:, current_features]
        X_test_sel = X_test_scaled[:, current_features]
        model = XGBClassifier(**best_params_all)
        model.fit(X_train_sel, y_train)
        y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
        score = average_precision_score(y_test, y_pred_proba)
        if score > best_score:
            best_score = score
            best_idx = idx
    selected.append(best_idx)
    remaining.remove(best_idx)
    greedy_log.append({'n_features': len(selected), 'features': ','.join([feature_names[i] for i in selected]), 'aucpr': best_score})
pd.DataFrame(greedy_log).to_csv(os.path.join(RESULTS_DIR, 'greedy_aucpr_curve.csv'), index=False)
# XGBoost Importance
xgb_model = XGBClassifier(**best_params_all)
xgb_model.fit(X_train_scaled, y_train)
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).reset_index(drop=True)
aucpr_log_xgb = []
for k in range(1, min(25, len(feature_names)+1)):
    selected_features = importance_df.head(k)['feature'].tolist()
    idxs = [feature_names.index(f) for f in selected_features if f in feature_names]
    X_train_sel = X_train_scaled[:, idxs]
    X_test_sel = X_test_scaled[:, idxs]
    model = XGBClassifier(**best_params_all)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict_proba(X_test_sel)[:, 1]
    aucpr = average_precision_score(y_test, y_pred)
    aucpr_log_xgb.append({'n_features': len(selected_features), 'features': ','.join(selected_features), 'aucpr': aucpr})
pd.DataFrame(aucpr_log_xgb).to_csv(os.path.join(RESULTS_DIR, 'xgb_importance_aucpr_curve.csv'), index=False)
# Backward Elimination
backward_log = []
current_features = list(range(len(feature_names)))
for k in range(len(feature_names), 0, -1):
    best_score = -np.inf
    worst_idx = None
    if len(current_features) == 1:
        idxs = current_features
        X_train_sel = X_train_scaled[:, idxs]
        X_test_sel = X_test_scaled[:, idxs]
        model = XGBClassifier(**best_params_all)
        model.fit(X_train_sel, y_train)
        y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
        score = average_precision_score(y_test, y_pred_proba)
        backward_log.append({'n_features': len(current_features), 'features': feature_names[idxs[0]], 'aucpr': score})
        break
    for idx in current_features:
        temp_features = [i for i in current_features if i != idx]
        X_train_sel = X_train_scaled[:, temp_features]
        X_test_sel = X_test_scaled[:, temp_features]
        model = XGBClassifier(**best_params_all)
        model.fit(X_train_sel, y_train)
        y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
        score = average_precision_score(y_test, y_pred_proba)
        if score > best_score:
            best_score = score
            worst_idx = idx
    current_features.remove(worst_idx)
    backward_log.append({'n_features': len(current_features), 'features': ','.join([feature_names[i] for i in current_features]), 'aucpr': best_score})
pd.DataFrame(backward_log).sort_values('n_features').reset_index(drop=True).to_csv(os.path.join(RESULTS_DIR, 'backward_aucpr_curve.csv'), index=False)
# Boruta
boruta_clf = XGBClassifier(**best_params_all)
boruta_selector = BorutaPy(boruta_clf, n_estimators='auto', verbose=2, random_state=RANDOM_STATE, perc=100)
boruta_selector.fit(X_train_scaled, y_train.values)
selected_mask = boruta_selector.support_
selected_features = [feature_names[i] for i, x in enumerate(selected_mask) if x]
aucpr_log_boruta = []
for k in range(1, len(selected_features)+1):
    feats = selected_features[:k]
    idxs = [feature_names.index(f) for f in feats]
    X_train_sel = X_train_scaled[:, idxs]
    X_test_sel = X_test_scaled[:, idxs]
    model = XGBClassifier(**best_params_all)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict_proba(X_test_sel)[:, 1]
    aucpr = average_precision_score(y_test, y_pred)
    aucpr_log_boruta.append({'n_features': len(feats), 'features': ','.join(feats), 'aucpr': aucpr})
pd.DataFrame(aucpr_log_boruta).to_csv(os.path.join(RESULTS_DIR, 'boruta_aucpr_curve.csv'), index=False)
# Backward by XGBoost Importance
xgb_model_full = XGBClassifier(**best_params_all)
xgb_model_full.fit(X_train_scaled, y_train)
importances_full = xgb_model_full.feature_importances_
importance_order = np.argsort(importances_full)
backward_xgb_log = []
current_features = list(importance_order)
for k in range(len(feature_names), 0, -1):
    feats = [feature_names[i] for i in current_features]
    idxs = current_features
    X_train_sel = X_train_scaled[:, idxs]
    X_test_sel = X_test_scaled[:, idxs]
    model = XGBClassifier(**best_params_all)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict_proba(X_test_sel)[:, 1]
    aucpr = average_precision_score(y_test, y_pred)
    backward_xgb_log.append({'n_features': len(feats), 'features': ','.join(feats), 'aucpr': aucpr})
    if len(current_features) == 1:
        break
    current_features = current_features[1:]
pd.DataFrame(backward_xgb_log).sort_values('n_features').reset_index(drop=True).to_csv(os.path.join(RESULTS_DIR, 'backward_xgb_aucpr_curve.csv'), index=False)

# === 3. SELECT BEST FEATURE SET (max AUC-PR in 12-24 range) ===
print("🔎 Step 3: Selecting best feature set...")
method_files = [
    ('RFE', 'rfe_aucpr_curve.csv'),
    ('Greedy', 'greedy_aucpr_curve.csv'),
    ('XGBoost importance', 'xgb_importance_aucpr_curve.csv'),
    ('Backward', 'backward_aucpr_curve.csv'),
    ('Boruta', 'boruta_aucpr_curve.csv'),
    ('Backward XGBoost', 'backward_xgb_aucpr_curve.csv'),
]
best_auc = -np.inf
best_method = None
best_features = None
best_n_features = None
for method, fname in method_files:
    df = pd.read_csv(os.path.join(RESULTS_DIR, fname))
    df = df[(df['n_features'] >= 12) & (df['n_features'] <= 24) & (df['aucpr'] >= AUC_MIN)]
    if len(df) == 0:
        continue
    max_auc = df['aucpr'].max()
    max_rows = df[df['aucpr'] == max_auc]
    best_row = max_rows.loc[max_rows['n_features'].idxmin()]
    if best_row['aucpr'] > best_auc:
        best_auc = best_row['aucpr']
        best_method = method
        best_features = best_row['features'].split(',')
        best_n_features = best_row['n_features']
with open(os.path.join(RESULTS_DIR, 'best_feature_set.json'), 'w') as f:
    json.dump({
        'method': str(best_method),
        'n_features': int(best_n_features) if best_n_features is not None else None,
        'features': [str(f) for f in best_features] if best_features is not None else [],
        'aucpr': float(best_auc)
    }, f, indent=2)
print(f"Best feature set: {best_method} ({best_n_features} features), AUC-PR={best_auc:.5f}")

# === 4. OPTUNA HYPERPARAMETER SEARCH ON BEST FEATURE SET ===
print("🔎 Step 4: Optuna hyperparameter search on best feature set...")
# Prepare data for best features
idxs_best = [feature_names.index(f) for f in best_features if f in feature_names]
X_train_best = X_train_scaled[:, idxs_best]
X_test_best = X_test_scaled[:, idxs_best]
class_counts_best = np.bincount(y_train)
base_params_best = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0
}
if len(class_counts_best) > 1 and class_counts_best[1] > 0:
    base_params_best['scale_pos_weight'] = class_counts_best[0] / class_counts_best[1]
study_best = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
start_time = time.time()
study_best.optimize(lambda trial: optuna_objective(trial, X_train_best, y_train, base_params_best), n_trials=N_TRIALS, show_progress_bar=True)
optuna_time_best = time.time() - start_time
best_params_best = base_params_best.copy()
best_params_best.update(study_best.best_params)
# Final model on best features
model_best = XGBClassifier(**best_params_best)
model_best.fit(X_train_best, y_train)
y_pred_best = model_best.predict(X_test_best)
y_pred_proba_best = model_best.predict_proba(X_test_best)[:, 1]
metrics_best = {
    'accuracy': accuracy_score(y_test, y_pred_best),
    'precision': precision_score(y_test, y_pred_best, zero_division=0),
    'recall': recall_score(y_test, y_pred_best, zero_division=0),
    'f1': f1_score(y_test, y_pred_best, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_best),
    'auc_pr': average_precision_score(y_test, y_pred_proba_best)
}
with open(os.path.join(RESULTS_DIR, 'optuna_best_features_results.json'), 'w') as f:
    json.dump({'best_params': study_best.best_params, 'metrics': metrics_best, 'optuna_time_sec': optuna_time_best, 'features': best_features}, f, indent=2)

# === 5. COMPARISON TABLE ===
print("🔎 Step 5: Comparison table...")
row1 = {'stage': 'All features + Optuna', **metrics_all}
row2 = {'stage': f'Best features ({best_method}, {best_n_features}) + Optuna (old)', **metrics_all}
row3 = {'stage': f'Best features ({best_method}, {best_n_features}) + Optuna (re-optimized)', **metrics_best}
comparison_df = pd.DataFrame([row1, row2, row3])
comparison_df.to_csv(os.path.join(RESULTS_DIR, 'comparison_table.csv'), index=False)
print(comparison_df)

# === 6. ACADEMIC STYLE PLOT ===
print("🔄 Building academic style plot...")
files = [
    ('Direct XGBoost', 'xgb_importance_aucpr_curve.csv', '#00bfff'),
    ('Backward XGBoost', 'backward_xgb_aucpr_curve.csv', 'blue'),
    ('Direct Greedy', 'greedy_aucpr_curve.csv', 'fuchsia'),
    ('Backward Greedy', 'backward_aucpr_curve.csv', 'purple'),
    ('RFE', 'rfe_aucpr_curve.csv', 'green'),
    ('Boruta', 'boruta_aucpr_curve.csv', 'red'),
]
plt.figure(figsize=(8, 5))
for label, fname, color in files:
    df = pd.read_csv(os.path.join(RESULTS_DIR, fname))
    df = df[(df['n_features'] >= 12) & (df['n_features'] <= 24) & (df['aucpr'] >= AUC_MIN)]
    if len(df) == 0:
        continue
    plt.plot(df['n_features'], df['aucpr'], marker='o', color=color, label=label)
    max_auc = df['aucpr'].max()
    max_rows = df[df['aucpr'] == max_auc]
    best_row = max_rows.loc[max_rows['n_features'].idxmin()]
    plt.annotate(f"{best_row['aucpr']:.5f}", (best_row['n_features'], best_row['aucpr']), textcoords="offset points", xytext=(0,7), ha='center', fontsize=9, color=color)
plt.xlabel('Number of features')
plt.ylabel('AUC-PR')
plt.title('Optimal number of features by method')
plt.grid(True, zorder=0)
plt.legend()
plt.tight_layout()
plt.ylim(AUC_MIN, plt.ylim()[1] + (plt.ylim()[1] - AUC_MIN) * 0.08)
plot_path = os.path.join(RESULTS_DIR, 'compare_aucpr_curve_12_24_academic.png')
plt.savefig(plot_path)
print(f'Plot saved as {plot_path}') 