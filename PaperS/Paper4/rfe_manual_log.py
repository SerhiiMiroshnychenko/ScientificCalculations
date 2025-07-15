import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

# === Параметри ===
DATA_PATH = r'D:\PROJECTs\MY\ScientificCalculations\SC\ScientificCalculations\PaperS\Paper4\preprocessed_data2.csv'
TARGET = 'is_successful'
RANDOM_STATE = 42

# === Завантаження та підготовка даних ===
data = pd.read_csv(DATA_PATH)
X = data.drop(TARGET, axis=1)
y = data[TARGET]
X = X.select_dtypes(include=[np.number])
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === RFE вручну ===
remaining_features = feature_names.copy()
ranking = {}
metrics_log = []

while len(remaining_features) > 0:
    # Тренуємо модель на поточному наборі ознак
    idxs = [feature_names.index(f) for f in remaining_features]
    X_tr = X_train_scaled[:, idxs]
    X_te = X_test_scaled[:, idxs]
    model = XGBClassifier(
        subsample=1.0, reg_lambda=2.0, reg_alpha=0.5, n_estimators=800,
        min_child_weight=3, max_depth=7, learning_rate=0.1, gamma=0.2,
        colsample_bytree=0.7, objective='binary:logistic', eval_metric='aucpr',
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, tree_method='gpu_hist', predictor='gpu_predictor'
    )
    model.fit(X_tr, y_train)
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    aucpr = average_precision_score(y_test, y_pred_proba)
    # Логування
    metrics_log.append({
        'n_features': len(remaining_features),
        'features': ','.join(remaining_features),
        'aucpr': aucpr
    })
    # Визначаємо найменш важливу ознаку
    importances = model.feature_importances_
    min_idx = np.argmin(importances)
    removed_feature = remaining_features[min_idx]
    ranking[removed_feature] = len(remaining_features)
    # Видаляємо цю ознаку
    del remaining_features[min_idx]

# Формуємо результати
ranking_df = pd.DataFrame([
    {'feature': f, 'ranking': r} for f, r in ranking.items()
]).sort_values('ranking')
metrics_df = pd.DataFrame(metrics_log)

# Зберігаємо результати
ranking_df.to_csv('rfe_ranking_manual.csv', index=False)
metrics_df.to_csv('rfe_metrics_log.csv', index=False)

print('RFE ранжування збережено у rfe_ranking_manual.csv')
print('Лог метрик збережено у rfe_metrics_log.csv') 