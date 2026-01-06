# -*- coding: utf-8 -*-
"""
Скрипт для створення зведеної таблиці значущості ознак за 5 методами
та збереження її у форматі XLSX.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import dcor
import warnings

warnings.filterwarnings('ignore')

# --- Функції для обробки та аналізу (адаптовано з analyse.py) ---

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
    """
    Створює новий DataFrame де циклічні ознаки (sin/cos) об'єднані
    """
    feature_names = X.columns
    cyclical_pairs = identify_cyclical_features(feature_names)
    
    # Створюємо новий DataFrame для зберігання оброблених даних
    X_processed = pd.DataFrame()
    
    # Копіюємо оригінальні ознаки, які не є частиною циклічних пар
    for feature in feature_names:
        is_part_of_pair = False
        for pair in cyclical_pairs.values():
            if feature in pair:
                is_part_of_pair = True
                break
        
        if not is_part_of_pair:
            X_processed[feature] = X[feature]
    
    # Об'єднуємо циклічні ознаки
    for base_name, pair in cyclical_pairs.items():
        sin_feature, cos_feature = pair
        X_processed[base_name] = np.sqrt(X[sin_feature]**2 + X[cos_feature]**2)
    
    return X_processed

def calculate_auc_importance(X, y):
    importances = []
    for feature in X.columns:
        auc = roc_auc_score(y, X[feature])
        importances.append(max(auc, 1 - auc))
    return importances

def calculate_dcor_importance(X, y):
    importances = []
    y_array = y.values.reshape(-1, 1)
    for feature in X.columns:
        x_array = X[feature].values.reshape(-1, 1)
        dc = dcor.distance_correlation(x_array, y_array)
        importances.append(dc)
    return importances

# --- Основний блок скрипта ---

def main():
    """
    Основна функція для виконання аналізу та створення XLSX файлу.
    """
    file_path = 'preprocessed_data.csv'
    output_excel_file = 'feature_importance_summary_values.xlsx'

    # 1. Завантаження даних
    print("Завантаження даних...")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Помилка: файл '{file_path}' не знайдено. Переконайтесь, що він у тому ж каталозі.")
        return

    print(f"Завантажено набір даних розміром {data.shape}")

    # 2. Підготовка даних
    X = data.drop('is_successful', axis=1)
    y = data['is_successful']
    X_numeric = X.select_dtypes(exclude=['object'])
    
    print("Обробка циклічних ознак...")
    X_processed = preprocess_features_for_analysis(X_numeric)
    feature_names = X_processed.columns
    print(f"Аналізуємо {len(feature_names)} ознак.")

    # 3. Обчислення значущості за 5 методами
    print("Обчислення значущості за методом AUC...")
    auc_importances = calculate_auc_importance(X_processed, y)

    print("Обчислення значущості за методом Mutual Information...")
    mi_importances = mutual_info_classif(X_processed, y, random_state=42)

    print("Обчислення значущості за методом Distance Correlation...")
    dcor_importances = calculate_dcor_importance(X_processed, y)

    print("Обчислення значущості за допомогою Логістичної регресії...")
    scaler = StandardScaler()
    X_processed_scaled = scaler.fit_transform(X_processed)
    log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    log_reg.fit(X_processed_scaled, y)
    log_reg_importances = np.abs(log_reg.coef_[0])

    print("Обчислення значущості за допомогою Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_processed, y)
    dt_importances = dt.feature_importances_

    # 4. Створення зведеного DataFrame
    print("Створення зведеної таблиці...")
    summary_df = pd.DataFrame({
        'Ознака': feature_names,
        'AUC': auc_importances,
        'MutualInformation': mi_importances,
        'DistanceCorrelation': dcor_importances,
        'LogisticRegression': log_reg_importances,
        'DecisionTree': dt_importances
    })

    # Збереження повної (неокругленої) таблиці у CSV
    output_csv_full_file = 'feature_importance_summary_values_full.csv'
    try:
        summary_df.to_csv(output_csv_full_file, index=False, encoding='utf-8')
        print(f"Повну таблицю значущості (без округлення) успішно збережено у файл: {output_csv_full_file}")
    except Exception as e:
        print(f"Помилка при збереженні CSV файлу з повними значеннями: {e}")

    # Округлення значень до 4 знаків після коми
    numeric_cols = summary_df.columns.drop('Ознака')
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    # Сортування за середнім значенням для наочності (опціонально, але корисно)
    summary_df['Середня_значущість'] = summary_df[numeric_cols].mean(axis=1)
    summary_df = summary_df.sort_values(by='Середня_значущість', ascending=False).drop(columns=['Середня_значущість'])


    # 5. Збереження у XLSX файл
    try:
        summary_df.to_excel(output_excel_file, index=False, engine='openpyxl')
        print(f"Таблицю значущості успішно збережено у файл: {output_excel_file}")
    except Exception as e:
        print(f"Помилка при збереженні XLSX файлу: {e}")
        print("Спробуйте встановити необхідну бібліотеку: pip install openpyxl")

if __name__ == '__main__':
    main()
