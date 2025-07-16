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
from boruta import BorutaPy

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
RESULTS_DIR = "boruta_aucpr_curve_results"
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

# === Boruta з XGBoost ===
print("🔄 Запуск Boruta з XGBoost...")
# BorutaPy очікує RandomForest або об'єкти з fit/predict/feature_importances_
boruta_clf = XGBClassifier(**optimal_params)
boruta_selector = BorutaPy(
    boruta_clf,
    n_estimators='auto',
    verbose=2,
    random_state=RANDOM_STATE,
    perc=100
)
boruta_selector.fit(X_train_scaled, y_train.values)

# Всі ознаки, які Boruta позначила як важливі (True)
selected_mask = boruta_selector.support_
selected_features = [feature_names[i] for i, x in enumerate(selected_mask) if x]

# Ознаки у порядку важливості Boruta (Boruta не повертає ранжування, тому беремо важливість з XGBoost)
boruta_ranks = boruta_selector.ranking_
boruta_order = np.argsort(boruta_ranks)
ordered_features = [feature_names[i] for i in boruta_order]

# === AUC-PR для k найкращих ознак у порядку Boruta (тільки ті, що позначені як важливі) ===
aucpr_log = []
for k in range(1, len(selected_features)+1):
    feats = selected_features[:k]
    idxs = [feature_names.index(f) for f in feats]
    X_train_sel = X_train_scaled[:, idxs]
    X_test_sel = X_test_scaled[:, idxs]
    model = XGBClassifier(**optimal_params)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict_proba(X_test_sel)[:, 1]
    aucpr = average_precision_score(y_test, y_pred)
    aucpr_log.append({
        'n_features': len(feats),
        'features': ','.join(feats),
        'aucpr': aucpr
    })
aucpr_df = pd.DataFrame(aucpr_log)
aucpr_df.to_csv(os.path.join(RESULTS_DIR, 'boruta_aucpr_curve.csv'), index=False)

# === Знаходимо найкращу комбінацію ===
best_row = aucpr_df.loc[aucpr_df['aucpr'].idxmax()]
print(f"\nНайкращий результат: {best_row['aucpr']:.4f} для {int(best_row['n_features'])} ознак: {best_row['features']}")

# === Візуалізація ===
plt.figure(figsize=(8, 5))
plt.plot(aucpr_df['n_features'], aucpr_df['aucpr'], marker='o', color='red', label='Boruta (XGBoost)')
for x, y in zip(aucpr_df['n_features'], aucpr_df['aucpr']):
    plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8, color='red')
plt.xlabel('Кількість ознак')
plt.ylabel('AUC-PR')
plt.title('Залежність AUC-PR від кількості ознак (Boruta, XGBoost)')
plt.grid(True)
plt.legend()
plt.tight_layout()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax + (ymax - ymin) * 0.08)
plot_path = os.path.join(RESULTS_DIR, 'boruta_aucpr_curve.png')
plt.savefig(plot_path)
print(f"Графік збережено у {os.path.abspath(plot_path)}") 