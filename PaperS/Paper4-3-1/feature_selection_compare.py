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
RESULTS_DIR = "feature_selection_compare_results"
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
# Замість Optuna підставляємо готові оптимальні параметри
optimal_params = {
    'subsample': 1.0,
    'reg_lambda': 2.0,
    'reg_alpha': 0.5,
    'n_estimators': 800,
    'min_child_weight': 3,
    'max_depth': 7,
    'learning_rate': 0.1,
    'gamma': 0.2,
    'colsample_bytree': 0.7,
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0,
    'tree_method': 'hist',
    'device': 'cuda',
}

# Final model on all features
model_all = XGBClassifier(**optimal_params)
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
with open(os.path.join(RESULTS_DIR, 'optimal_params_results.json'), 'w') as f:
    json.dump({'optimal_params': optimal_params, 'metrics': metrics_all}, f, indent=2)

# === 3. FEATURE SELECTION METHODS (using optimal_params) ===
print("🔎 Step 3: Feature selection with all methods...")
# RFE
estimator = XGBClassifier(**optimal_params)
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
    model = XGBClassifier(**optimal_params)
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
        model = XGBClassifier(**optimal_params)
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
xgb_model = XGBClassifier(**optimal_params)
xgb_model.fit(X_train_scaled, y_train)
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).reset_index(drop=True)
aucpr_log_xgb = []
for k in range(1, min(25, len(feature_names)+1)):
    selected_features = importance_df.head(k)['feature'].tolist()
    idxs = [feature_names.index(f) for f in selected_features if f in feature_names]
    X_train_sel = X_train_scaled[:, idxs]
    X_test_sel = X_test_scaled[:, idxs]
    model = XGBClassifier(**optimal_params)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict_proba(X_test_sel)[:, 1]
    aucpr = average_precision_score(y_test, y_pred)
    aucpr_log_xgb.append({'n_features': len(selected_features), 'features': ','.join(selected_features), 'aucpr': aucpr})
pd.DataFrame(aucpr_log_xgb).to_csv(os.path.join(RESULTS_DIR, 'xgb_importance_aucpr_curve.csv'), index=False)
# Backward Elimination
backward_log = []
current_features = list(range(len(feature_names)))
X_train_sel = X_train_scaled[:, current_features]
X_test_sel = X_test_scaled[:, current_features]
model = XGBClassifier(**optimal_params)
model.fit(X_train_sel, y_train)
y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
score = average_precision_score(y_test, y_pred_proba)
backward_log.append({'n_features': len(current_features), 'features': ','.join([feature_names[i] for i in current_features]), 'aucpr': score})
for k in range(len(feature_names), 0, -1):
    best_score = -np.inf
    worst_idx = None
    if len(current_features) == 1:
        idxs = current_features
        X_train_sel = X_train_scaled[:, idxs]
        X_test_sel = X_test_scaled[:, idxs]
        model = XGBClassifier(**optimal_params)
        model.fit(X_train_sel, y_train)
        y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
        score = average_precision_score(y_test, y_pred_proba)
        backward_log.append({'n_features': len(current_features), 'features': feature_names[idxs[0]], 'aucpr': score})
        break
    for idx in current_features:
        temp_features = [i for i in current_features if i != idx]
        X_train_sel = X_train_scaled[:, temp_features]
        X_test_sel = X_test_scaled[:, temp_features]
        model = XGBClassifier(**optimal_params)
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
boruta_clf = XGBClassifier(**optimal_params)
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
    model = XGBClassifier(**optimal_params)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict_proba(X_test_sel)[:, 1]
    aucpr = average_precision_score(y_test, y_pred)
    aucpr_log_boruta.append({'n_features': len(feats), 'features': ','.join(feats), 'aucpr': aucpr})
pd.DataFrame(aucpr_log_boruta).to_csv(os.path.join(RESULTS_DIR, 'boruta_aucpr_curve.csv'), index=False)
# Backward by XGBoost Importance
xgb_model_full = XGBClassifier(**optimal_params)
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
    model = XGBClassifier(**optimal_params)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict_proba(X_test_sel)[:, 1]
    aucpr = average_precision_score(y_test, y_pred)
    backward_xgb_log.append({'n_features': len(feats), 'features': ','.join(feats), 'aucpr': aucpr})
    if len(current_features) == 1:
        break
    current_features = current_features[1:]
pd.DataFrame(backward_xgb_log).sort_values('n_features').reset_index(drop=True).to_csv(os.path.join(RESULTS_DIR, 'backward_xgb_aucpr_curve.csv'), index=False)

# === 4. COMPARISON TABLE (Baseline, Optuna, Greedy+Optuna) ===
print("🔎 Step 4: Comparison table...")
# Автоматичний вибір найкращого набору серед усіх методів
method_files = [
    ('Прямий XGBoost', 'xgb_importance_aucpr_curve.csv'),
    ('Зворотній XGBoost', 'backward_xgb_aucpr_curve.csv'),
    ('Прямий Greedy', 'greedy_aucpr_curve.csv'),
    ('Зворотній Greedy', 'backward_aucpr_curve.csv'),
    ('RFE', 'rfe_aucpr_curve.csv'),
    ('Boruta', 'boruta_aucpr_curve.csv'),
]
best_auc = -np.inf
best_method = None
best_features = None
best_n_features = None
for method_name, fname in method_files:
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    df = df[(df['n_features'] >= 12) & (df['n_features'] <= 24)]
    if len(df) == 0:
        continue
    max_auc = df['aucpr'].max()
    max_rows = df[df['aucpr'] == max_auc]
    best_row = max_rows.loc[max_rows['n_features'].idxmin()]
    if best_row['aucpr'] > best_auc:
        best_auc = best_row['aucpr']
        best_method = method_name
        best_features = best_row['features'].split(',')
        best_n_features = int(best_row['n_features'])
print(f"Best set: {best_method}, {best_n_features} features, AUC-PR={best_auc:.5f}")
# Оцінка моделі на best features + Optuna
idxs_best = [feature_names.index(f) for f in best_features if f in feature_names]
X_train_best = X_train_scaled[:, idxs_best]
X_test_best = X_test_scaled[:, idxs_best]
model_best = XGBClassifier(**optimal_params)
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
row1 = {'stage': 'All features (default)', **baseline_metrics}
row2 = {'stage': 'All features + Optuna', **metrics_all}
row3 = {'stage': f'Best features ({best_method}, {best_n_features}) + Optuna', **metrics_best}
comparison_df = pd.DataFrame([row1, row2, row3])
comparison_df.to_csv(os.path.join(RESULTS_DIR, 'comparison_table.csv'), index=False)
print(comparison_df)

# === 5. CONFUSION MATRICES VISUALIZATION ===
print("🔄 Building confusion matrix plots...")
labels = np.unique(y_test)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, cm, title in zip(
    axes,
    [conf_matrix_baseline, conf_matrix_all, conf_matrix_best],
    [
        'All features (default)',
        'All features + Optuna',
        f'Best features ({best_method}, {best_n_features}) + Optuna'
    ]
):
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

# === 7. ACADEMIC STYLE PLOT ===
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
print('All done!') 