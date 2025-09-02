import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from boruta import BorutaPy

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

# === PARAMETERS ===
DATA_PATH = "preprocessed_data2.csv"
RESULTS_DIR = "compare_methods_with_academic_plot_results"
RANDOM_STATE = 42
AUC_MIN = 0.9766

optimal_params = {
    'subsample': 0.97,
    'reg_lambda': 0.35,
    'reg_alpha': 0.9,
    'n_estimators': 750,
    'min_child_weight': 2,
    'max_depth': 12,
    'learning_rate': 0.02,
    'gamma': 0.7,
    'colsample_bytree': 0.69,
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0,
    'tree_method': 'hist',
    'device': 'cuda',
}

os.makedirs(RESULTS_DIR, exist_ok=True)

# === DATA LOADING AND PREPROCESSING ===
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

# === RFE ===
print("🔄 Running RFE...")
estimator = XGBClassifier(**optimal_params)
rfe = RFE(estimator, n_features_to_select=None, step=1)
rfe.fit(X_train_scaled, y_train)
ranking = rfe.ranking_
rfe_df = pd.DataFrame({
    'feature': feature_names,
    'ranking': ranking
}).sort_values('ranking').reset_index(drop=True)
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
    aucpr_log_rfe.append({
        'n_features': len(selected_features),
        'features': ','.join(selected_features),
        'aucpr': aucpr
    })
aucpr_df_rfe = pd.DataFrame(aucpr_log_rfe)
aucpr_df_rfe.to_csv(os.path.join(RESULTS_DIR, 'rfe_aucpr_curve.csv'), index=False)

# === Greedy Feature Selection (Forward) ===
print("🔄 Running Greedy Forward selection...")
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
    greedy_log.append({
        'n_features': len(selected),
        'features': ','.join([feature_names[i] for i in selected]),
        'aucpr': best_score
    })
aucpr_df_greedy = pd.DataFrame(greedy_log)
aucpr_df_greedy.to_csv(os.path.join(RESULTS_DIR, 'greedy_aucpr_curve.csv'), index=False)

# === XGBoost Importance Feature Selection ===
print("🔄 Running XGBoost importance selection...")
xgb_model = XGBClassifier(**optimal_params)
xgb_model.fit(X_train_scaled, y_train)
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False).reset_index(drop=True)
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
    aucpr_log_xgb.append({
        'n_features': len(selected_features),
        'features': ','.join(selected_features),
        'aucpr': aucpr
    })
aucpr_df_xgb = pd.DataFrame(aucpr_log_xgb)
aucpr_df_xgb.to_csv(os.path.join(RESULTS_DIR, 'xgb_importance_aucpr_curve.csv'), index=False)

# === Backward Elimination ===
print("🔄 Running Backward elimination...")
backward_log = []
current_features = list(range(len(feature_names)))
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
        backward_log.append({
            'n_features': len(current_features),
            'features': feature_names[idxs[0]],
            'aucpr': score
        })
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
    backward_log.append({
        'n_features': len(current_features),
        'features': ','.join([feature_names[i] for i in current_features]),
        'aucpr': best_score
    })
aucpr_df_backward = pd.DataFrame(backward_log)
aucpr_df_backward = aucpr_df_backward.sort_values('n_features').reset_index(drop=True)
aucpr_df_backward.to_csv(os.path.join(RESULTS_DIR, 'backward_aucpr_curve.csv'), index=False)

# === Boruta with XGBoost ===
print("🔄 Running Boruta with XGBoost...")
boruta_clf = XGBClassifier(**optimal_params)
boruta_selector = BorutaPy(
    boruta_clf,
    n_estimators='auto',
    verbose=2,
    random_state=RANDOM_STATE,
    perc=100
)
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
    aucpr_log_boruta.append({
        'n_features': len(feats),
        'features': ','.join(feats),
        'aucpr': aucpr
    })
aucpr_df_boruta = pd.DataFrame(aucpr_log_boruta)
aucpr_df_boruta.to_csv(os.path.join(RESULTS_DIR, 'boruta_aucpr_curve.csv'), index=False)

# === Backward by XGBoost Importance ===
print("🔄 Running Backward by XGBoost importance...")
xgb_model_full = XGBClassifier(**optimal_params)
xgb_model_full.fit(X_train_scaled, y_train)
importances_full = xgb_model_full.feature_importances_
importance_order = np.argsort(importances_full)  # from least to most important
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
    backward_xgb_log.append({
        'n_features': len(feats),
        'features': ','.join(feats),
        'aucpr': aucpr
    })
    if len(current_features) == 1:
        break
    current_features = current_features[1:]
aucpr_df_backward_xgb = pd.DataFrame(backward_xgb_log)
aucpr_df_backward_xgb = aucpr_df_backward_xgb.sort_values('n_features').reset_index(drop=True)
aucpr_df_backward_xgb.to_csv(os.path.join(RESULTS_DIR, 'backward_xgb_aucpr_curve.csv'), index=False)

# === Academic style plot (like plot_aucpr_from_results.py) ===
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