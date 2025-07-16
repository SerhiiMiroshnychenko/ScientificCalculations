import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
RESULTS_DIR = "greedy_aucpr_curve_fullcycle_results"
AUC_LOG_CSV = os.path.join(RESULTS_DIR, "greedy_aucpr_curve.csv")
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

# === Greedy Feature Selection ===
print("🔄 Запуск жадібного відбору ознак...")
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
    print(f"Крок {step+1}: {feature_names[best_idx]} (AUC-PR: {best_score:.4f})")

greedy_aucpr_df = pd.DataFrame(greedy_log)
greedy_aucpr_df.to_csv(AUC_LOG_CSV, index=False)
print(f"AUC-PR для кожного k збережено у {AUC_LOG_CSV}")

# === Візуалізація залежності AUC-PR від кількості ознак (тільки для топ-12) ===
plt.figure(figsize=(8, 5))
greedy_aucpr_df = pd.read_csv(AUC_LOG_CSV)
greedy_aucpr_df_top12 = greedy_aucpr_df.head(12)
plt.plot(greedy_aucpr_df_top12['n_features'], greedy_aucpr_df_top12['aucpr'], marker='o')
for x, y in zip(greedy_aucpr_df_top12['n_features'], greedy_aucpr_df_top12['aucpr']):
    plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=9)
plt.xlabel('Кількість ознак')
plt.ylabel('AUC-PR')
plt.title('Залежність AUC-PR від кількості ознак (Greedy, топ-12)')
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'greedy_aucpr_curve_top12.png')
plt.savefig(plot_path)
print(f"Графік (топ-12) збережено у {os.path.abspath(plot_path)}")

# === Візуалізація для діапазону 12-24 ===
greedy_aucpr_df_12_24 = greedy_aucpr_df[(greedy_aucpr_df['n_features'] >= 12) & (greedy_aucpr_df['n_features'] <= 24)]
plt.figure(figsize=(8, 5))
plt.plot(greedy_aucpr_df_12_24['n_features'], greedy_aucpr_df_12_24['aucpr'], marker='o')
for x, y in zip(greedy_aucpr_df_12_24['n_features'], greedy_aucpr_df_12_24['aucpr']):
    plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8)
plt.xlabel('Кількість ознак')
plt.ylabel('AUC-PR')
plt.title('Залежність AUC-PR від кількості ознак (Greedy, 12-24)')
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'greedy_aucpr_curve_12_24.png')
plt.savefig(plot_path)
print(f"Графік (12-24) збережено у {os.path.abspath(plot_path)}")