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

# === Функції для обробки циклічних ознак ===
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

# === Параметри ===
DATA_PATH = r"D:\PROJECTs\MY\ScientificCalculations\SC\ScientificCalculations\PaperS\Paper4\preprocessed_data2.csv"
RESULTS_DIR = "compare_rfe_greedy_xgb_backward_aucpr_curve_results"
RANDOM_STATE = 42

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

os.makedirs(RESULTS_DIR, exist_ok=True)

# === Завантаження та підготовка даних ===
data = pd.read_csv(DATA_PATH)
if 'is_successful' not in data.columns:
    raise ValueError("❌ Цільова змінна 'is_successful' не знайдена!")
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
print("🔄 Запуск RFE...")
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
        'aucpr': aucpr
    })
aucpr_df_rfe = pd.DataFrame(aucpr_log_rfe)
aucpr_df_rfe.to_csv(os.path.join(RESULTS_DIR, 'rfe_aucpr_curve.csv'), index=False)

# === Greedy Feature Selection (Forward) ===
print("🔄 Запуск жадібного відбору ознак (Forward)...")
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
        'aucpr': best_score
    })
aucpr_df_greedy = pd.DataFrame(greedy_log)
aucpr_df_greedy.to_csv(os.path.join(RESULTS_DIR, 'greedy_aucpr_curve.csv'), index=False)

# === XGBoost Importance Feature Selection ===
print("🔄 Визначення важливості ознак XGBoost...")
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
        'aucpr': aucpr
    })
aucpr_df_xgb = pd.DataFrame(aucpr_log_xgb)
aucpr_df_xgb.to_csv(os.path.join(RESULTS_DIR, 'xgb_importance_aucpr_curve.csv'), index=False)

# === Backward Elimination ===
print("🔄 Запуск жадібного відбору ознак (Backward elimination)...")
backward_log = []
current_features = list(range(len(feature_names)))
for k in range(len(feature_names), 0, -1):
    best_score = -np.inf
    worst_idx = None
    # Якщо залишилась одна ознака — не видаляємо, просто рахуємо
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
            'aucpr': score
        })
        break
    # Перебираємо всі можливі видалення
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
    # Видаляємо ту ознаку, видалення якої найменше погіршує (або найбільше покращує) метрику
    current_features.remove(worst_idx)
    backward_log.append({
        'n_features': len(current_features),
        'aucpr': best_score
    })
# Сортуємо за кількістю ознак
aucpr_df_backward = pd.DataFrame(backward_log)
aucpr_df_backward = aucpr_df_backward.sort_values('n_features').reset_index(drop=True)
aucpr_df_backward.to_csv(os.path.join(RESULTS_DIR, 'backward_aucpr_curve.csv'), index=False)

# === Візуалізація для діапазону 12-24 ===
plt.figure(figsize=(8, 5))
# RFE — темно-блакитна
aucpr_df_rfe_12_24 = aucpr_df_rfe[(aucpr_df_rfe['n_features'] >= 12) & (aucpr_df_rfe['n_features'] <= 24)]
plt.plot(aucpr_df_rfe_12_24['n_features'], aucpr_df_rfe_12_24['aucpr'], marker='o', color='#0077b6', label='RFE')
for x, y in zip(aucpr_df_rfe_12_24['n_features'], aucpr_df_rfe_12_24['aucpr']):
    plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8, color='#0077b6')
# Greedy — зелена
aucpr_df_greedy_12_24 = aucpr_df_greedy[(aucpr_df_greedy['n_features'] >= 12) & (aucpr_df_greedy['n_features'] <= 24)]
plt.plot(aucpr_df_greedy_12_24['n_features'], aucpr_df_greedy_12_24['aucpr'], marker='o', color='green', label='Greedy')
for x, y in zip(aucpr_df_greedy_12_24['n_features'], aucpr_df_greedy_12_24['aucpr']):
    plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8, color='green')
# XGBoost importance — помаранчевий
aucpr_df_xgb_12_24 = aucpr_df_xgb[(aucpr_df_xgb['n_features'] >= 12) & (aucpr_df_xgb['n_features'] <= 24)]
plt.plot(aucpr_df_xgb_12_24['n_features'], aucpr_df_xgb_12_24['aucpr'], marker='o', color='orange', label='XGBoost importance')
for x, y in zip(aucpr_df_xgb_12_24['n_features'], aucpr_df_xgb_12_24['aucpr']):
    plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8, color='orange')
# Backward elimination — пурпурний
aucpr_df_backward_12_24 = aucpr_df_backward[(aucpr_df_backward['n_features'] >= 12) & (aucpr_df_backward['n_features'] <= 24)]
plt.plot(aucpr_df_backward_12_24['n_features'], aucpr_df_backward_12_24['aucpr'], marker='o', color='purple', label='Backward elimination')
for x, y in zip(aucpr_df_backward_12_24['n_features'], aucpr_df_backward_12_24['aucpr']):
    plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8, color='purple')
plt.xlabel('Кількість ознак')
plt.ylabel('AUC-PR')
plt.title('Залежність AUC-PR від кількості ознак (12-24): RFE, Greedy, XGBoost, Backward')
plt.grid(True)
plt.legend()
plt.tight_layout()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax + (ymax - ymin) * 0.08)
plot_path = os.path.join(RESULTS_DIR, 'compare_aucpr_curve_12_24.png')
plt.savefig(plot_path)
print(f"Об'єднаний графік (12-24) збережено у {os.path.abspath(plot_path)}") 