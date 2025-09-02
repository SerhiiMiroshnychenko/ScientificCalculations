import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
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
RESULTS_DIR = "optuna_feature_selection_full_cycle_baseline_vs_optuna_vs_greedy_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42
AUC_MIN = 0.9778
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

# === 1. BASELINE: XGBoost with default parameters on all features ===
print("🔎 Step 1: Baseline XGBoost (default params) on all features...")
baseline_model = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
baseline_model.fit(X_train_scaled, y_train)
y_pred_baseline = baseline_model.predict(X_test_scaled)
y_pred_proba_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]
baseline_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_baseline),
    'precision': precision_score(y_test, y_pred_baseline, zero_division=0),
    'recall': recall_score(y_test, y_pred_baseline, zero_division=0),
    'f1': f1_score(y_test, y_pred_baseline, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_baseline),
    'auc_pr': average_precision_score(y_test, y_pred_proba_baseline)
}
conf_matrix_baseline = confusion_matrix(y_test, y_pred_baseline)

# === 2. OPTUNA HYPERPARAMETER SEARCH ON ALL FEATURES ===
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

print("🔎 Step 2: Optuna hyperparameter search on all features...")
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
conf_matrix_all = confusion_matrix(y_test, y_pred_all)
with open(os.path.join(RESULTS_DIR, 'optuna_all_features_results.json'), 'w') as f:
    json.dump({'best_params': study.best_params, 'metrics': metrics_all, 'optuna_time_sec': optuna_time}, f, indent=2)

# === 3. GREEDY FORWARD SELECTION (Best features) ===
print("🔎 Step 3: Greedy forward selection (best features)...")
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
# Вибираємо найкращий набір (максимум AUC-PR у діапазоні 12-24)
greedy_df = pd.DataFrame(greedy_log)
greedy_df = greedy_df[(greedy_df['n_features'] >= 12) & (greedy_df['n_features'] <= 24)]
best_row = greedy_df.loc[greedy_df['aucpr'].idxmax()]
best_features = best_row['features'].split(',')
best_n_features = int(best_row['n_features'])
print(f"Best Greedy set: {best_n_features} features, AUC-PR={best_row['aucpr']:.5f}")
# Оцінка моделі на best features + Optuna
idxs_best = [feature_names.index(f) for f in best_features if f in feature_names]
X_train_best = X_train_scaled[:, idxs_best]
X_test_best = X_test_scaled[:, idxs_best]
model_best = XGBClassifier(**best_params_all)
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
conf_matrix_best = confusion_matrix(y_test, y_pred_best)

# === 4. COMPARISON TABLE ===
print("🔎 Step 4: Comparison table...")
row1 = {'stage': 'All features (default)', **baseline_metrics}
row2 = {'stage': 'All features + Optuna', **metrics_all}
row3 = {'stage': f'Best features (Greedy, {best_n_features}) + Optuna', **metrics_best}
comparison_df = pd.DataFrame([row1, row2, row3])
comparison_df.to_csv(os.path.join(RESULTS_DIR, 'comparison_table.csv'), index=False)
print(comparison_df)

# === 5. CONFUSION MATRICES VISUALIZATION ===
print("🔄 Building confusion matrix plots...")
labels = np.unique(y_test)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, cm, title in zip(axes, [conf_matrix_baseline, conf_matrix_all, conf_matrix_best],
                        ['All features (default)', 'All features + Optuna', f'Best features (Greedy, {best_n_features}) + Optuna']):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'))

# === 6. METRICS BARPLOT ===
print("🔄 Building metrics barplot...")
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'auc_pr']
metrics_data = [
    [baseline_metrics[m] for m in metrics_to_plot],
    [metrics_all[m] for m in metrics_to_plot],
    [metrics_best[m] for m in metrics_to_plot],
]
bar_width = 0.2
x = np.arange(len(metrics_to_plot))
plt.figure(figsize=(10, 5))
plt.bar(x - bar_width, metrics_data[0], width=bar_width, label='All features (default)')
plt.bar(x, metrics_data[1], width=bar_width, label='All features + Optuna')
plt.bar(x + bar_width, metrics_data[2], width=bar_width, label=f'Best features (Greedy, {best_n_features}) + Optuna')
plt.xticks(x, metrics_to_plot)
plt.ylabel('Score')
plt.title('Comparison of model metrics')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metrics_comparison_barplot.png'))
print('All done!') 