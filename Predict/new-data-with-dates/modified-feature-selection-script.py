import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Встановлюємо Agg бекенд
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, f_classif, RFE, SelectKBest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier  # Додано для аналізу точності моделей та важливості ознак
from tabulate import tabulate
import os
import joblib
from datetime import datetime

# Створюємо директорію для збереження результатів
results_dir = f"feature_selection_results"
os.makedirs(results_dir, exist_ok=True)

# Налаштування логування
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{results_dir}/feature_selection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Створюємо словник для перекладу назв ознак
feature_names_ua = {
    'order_amount': 'Сума замовлення',
    'order_messages': 'Кількість повідомлень',
    'order_changes': 'Кількість змін в замовлені',
    'partner_success_rate': 'Сердній % успішних замовлень клієнта',
    'partner_total_orders': 'Кількість замовлень клієнта',
    'partner_order_age_days': 'Термін співпраці',
    'partner_avg_amount': 'Середня сума замовлень клієнта',
    'partner_success_avg_amount': 'Середня сума успішних замовлень клієнта',
    'partner_fail_avg_amount': 'Середня сума невдалих замовлень клієнта',
    'partner_total_messages': 'Загальна кількість повідомлень клієнта',
    'partner_success_avg_messages': 'Середня кількість повідомлень успішних замовлень',
    'partner_fail_avg_messages': 'Середня кількість повідомлень невдалих замовлень',
    'partner_avg_changes': 'Середня кількість змін в замовленях клієнта',
    'partner_success_avg_changes': 'Середня кількість змін в успішних замовленях клієнта',
    'partner_fail_avg_changes': 'Середня кількість змін в невдалих замовленях клієнта',
    'day_of_week': 'День тижня',
    'month': 'Місяць',
    'quarter': 'Квартал',
    'hour_of_day': 'Година доби',
    'order_lines_count': 'Кількість позицій в замовленні',
    'discount_total': 'Загальна знижка',
    'salesperson': 'Менеджер',
    'source': 'Джерело замовлення',
    'create_date_months': 'Місяці від найранішої дати'
}

# Функція для отримання українських назв ознак
def get_ua_feature_name(feature):
    """
    Повертає українську назву ознаки, якщо вона є в словнику

    Args:
        feature (str): Назва ознаки англійською

    Returns:
        str: Назва ознаки українською або оригінальна назва
    """
    return feature_names_ua.get(feature, feature)

# Створюємо функцію для обчислення вартості ознак різними методами
def evaluate_features(X, y, cv=5):
    """
    Функція для оцінки важливості ознак різними методами

    Args:
        X (pd.DataFrame): Набір ознак
        y (pd.Series): Цільова змінна
        cv (int): Кількість фолдів для крос-валідації

    Returns:
        tuple: Кортеж з DataFrame результатів та словником моделей
    """
    logger.info(f"Оцінюємо важливість {X.shape[1]} ознак...")
    features = X.columns
    n_features = len(features)
    results_dict = {}
    models_dict = {}

    # Додаємо перевірку на мультиколінеарність
    correlation_matrix = X.corr().abs()
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(upper.columns[i], upper.columns[j]) for i in range(len(upper.columns))
                       for j in range(len(upper.columns)) if i < j and upper.iloc[i, j] > 0.8]

    if high_corr_pairs:
        logger.warning(f"Виявлено {len(high_corr_pairs)} пар ознак з високою кореляцією (>0.8):")
        for pair in high_corr_pairs[:5]:
            logger.warning(f"  {pair[0]} <-> {pair[1]}: {correlation_matrix.loc[pair[0], pair[1]]:.3f}")
        if len(high_corr_pairs) > 5:
            logger.warning(f"  ... та {len(high_corr_pairs)-5} інших пар")

    # 1. Mutual Information
    logger.info("Обчислюємо Mutual Information...")
    mi_scores = mutual_info_classif(X, y, random_state=42, discrete_features='auto')
    results_dict['MI Score'] = dict(zip(features, mi_scores))

    # 2. ANOVA F-test
    logger.info("Обчислюємо ANOVA F-test...")
    f_scores, _ = f_classif(X, y)
    results_dict['F Score'] = dict(zip(features, f_scores))

    # 3. Spearman Correlation
    logger.info("Обчислюємо Spearman Correlation...")
    spearman_corr = X.corrwith(y, method="spearman").abs()
    results_dict['Spearman Score'] = dict(zip(features, spearman_corr))

    # 5. Logistic Regression Coefficients
    logger.info("Обчислюємо Logistic Regression Coefficients...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_lr = LogisticRegression(max_iter=10000, random_state=42, solver='liblinear',
                                  class_weight='balanced')
    model_lr.fit(X_scaled, y)
    lr_coefficients = model_lr.coef_[0]
    results_dict['LR Coefficient'] = dict(zip(features, lr_coefficients))
    results_dict['Absolute Coefficient'] = dict(zip(features, np.abs(lr_coefficients)))
    models_dict['LogisticRegression'] = model_lr
    models_dict['Scaler'] = scaler

    # 4. Recursive Feature Elimination (RFE) з крос-валідацією
    logger.info("Запускаємо Recursive Feature Elimination (RFE) з крос-валідацією...")
    n_features_to_select = max(10, n_features // 3)
    rfe_selector = RFE(estimator=model_lr, n_features_to_select=n_features_to_select, step=1)
    rfe_selector.fit(X_scaled, y)
    rfe_selected = rfe_selector.support_
    results_dict['RFE Selected'] = dict(zip(features, rfe_selected))
    models_dict['RFE'] = rfe_selector

    # 6. Decision Tree Feature Importance з крос-валідацією
    logger.info("Обчислюємо Decision Tree Feature Importance з крос-валідацією...")
    model_dt = DecisionTreeClassifier(random_state=42)
    dt_importances = []
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for train_idx, val_idx in cv_obj.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        model_dt.fit(X_train, y_train)
        dt_importances.append(model_dt.feature_importances_)
    dt_importance = np.mean(dt_importances, axis=0)
    results_dict['DT Score'] = dict(zip(features, dt_importance))
    models_dict['DecisionTree'] = model_dt

    # 7. Random Forest Importance з крос-валідацією
    logger.info("Обчислюємо Random Forest Importance з крос-валідацією...")
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    importances = []
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for train_idx, val_idx in cv_obj.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        model_rf.fit(X_train, y_train)
        importances.append(model_rf.feature_importances_)
    rf_importance = np.mean(importances, axis=0)
    results_dict['RF Score'] = dict(zip(features, rf_importance))
    models_dict['RandomForest'] = model_rf

    # 8. XGBoost Feature Importance з крос-валідацією
    logger.info("Обчислюємо XGBoost Feature Importance з крос-валідацією...")
    model_xgb = xgb.XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1,
                                  scale_pos_weight=(len(y) - sum(y)) / sum(y))
    importances = []
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for train_idx, val_idx in cv_obj.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        model_xgb.fit(X_train, y_train)
        importances.append(model_xgb.feature_importances_)
    xgb_importance = np.mean(importances, axis=0)
    results_dict['XGBoost Score'] = dict(zip(features, xgb_importance))
    models_dict['XGBoost'] = model_xgb

    # Створюємо DataFrame з результатами
    results_df = pd.DataFrame({col: pd.Series(results_dict[col]) for col in results_dict})
    results_df.index.name = 'Feature'
    results_df.reset_index(inplace=True)

    # Обчислюємо ранги (чим більше значення – тим важливіше)
    rank_columns = ['MI Score', 'F Score', 'Spearman Score', 'Absolute Coefficient', 'DT Score', 'RF Score', 'XGBoost Score']
    for col in rank_columns:
        results_df[f'{col} Rank'] = results_df[col].rank(method='average')

    # Ранг для RFE: вибрані ознаки отримують більше значення
    results_df['RFE Rank'] = results_df['RFE Selected'] * n_features

    # Обчислюємо загальний рейтинг (сума всіх рангів)
    results_df['Total Importance Score'] = results_df[[
        'MI Score Rank', 'F Score Rank', 'Spearman Score Rank', 'RFE Rank',
        'Absolute Coefficient Rank', 'DT Score Rank', 'RF Score Rank', 'XGBoost Score Rank'
    ]].sum(axis=1)

    # Нормалізуємо до 100%
    max_score = results_df['Total Importance Score'].max()
    results_df['Total Importance Rank'] = round((results_df['Total Importance Score'] / max_score) * 100)

    # Сортуємо за спаданням (найважливіші ознаки зверху)
    results_df = results_df.sort_values(by='Total Importance Rank', ascending=False)

    return results_df, models_dict

# Функція для оцінки моделей з різною кількістю ознак
def evaluate_models_with_features(X, y, results_df, top_n_features=None, cv=5):
    """
    Оцінює моделі з різною кількістю найважливіших ознак

    Args:
        X (pd.DataFrame): Повний набір ознак
        y (pd.Series): Цільова змінна
        results_df (pd.DataFrame): DataFrame з результатами оцінки ознак
        top_n_features (list): Список з кількостями ознак для перевірки. Якщо None, тестує всі ознаки від 1 до max
        cv (int): Кількість фолдів для крос-валідації

    Returns:
        pd.DataFrame: DataFrame з результатами
    """
    # Якщо top_n_features не вказано, тестуємо всі варіанти від 1 до кількості ознак
    if top_n_features is None:
        top_n_features = list(range(1, min(len(results_df) + 1, 26)))  # Обмежуємо максимум 25 ознаками

    logger.info(f"Оцінюємо моделі з різною кількістю ознак: від {min(top_n_features)} до {max(top_n_features)}...")
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=10000, random_state=42,
                                                 class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42,
                                               class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(eval_metric="logloss", random_state=42,
                                     scale_pos_weight=(len(y) - sum(y)) / sum(y)),
        'DecisionTree': DecisionTreeClassifier(random_state=42)  # Додано для аналізу точності
    }

    # Метрики, які будемо відстежувати
    metrics = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'roc_auc': 'ROC AUC'
    }

    results = []
    # Отримуємо список ознак, відсортований за важливістю
    all_features = results_df['Feature'].tolist()

    for n in top_n_features:
        selected_features = all_features[:n]  # Беремо перші n ознак за важливістю
        X_selected = X[selected_features]
        logger.info(f"Тестуємо моделі з {n} найважливішими ознаками...")

        for model_name, model in models.items():
            # Для логістичної регресії масштабуємо дані
            if model_name == 'LogisticRegression':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_selected)
            else:
                X_scaled = X_selected.copy()

            # Використовуємо cross_validate для отримання різних метрик
            cv_results = cross_val_score(model, X_scaled, y, cv=cv_obj,
                                         scoring='accuracy', n_jobs=-1)
            accuracy = cv_results.mean()
            accuracy_std = cv_results.std()

            # Обчислюємо інші метрики через крос-валідацію
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            from sklearn.model_selection import cross_val_predict

            y_pred = cross_val_predict(model, X_scaled, y, cv=cv_obj, n_jobs=-1)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)

            # Для ROC AUC потрібні ймовірності, які не всі моделі можуть видати
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = cross_val_predict(model, X_scaled, y, cv=cv_obj,
                                                method='predict_proba', n_jobs=-1)
                    roc_auc = roc_auc_score(y, y_proba[:, 1])
                else:
                    roc_auc = float('nan')
            except:
                roc_auc = float('nan')

            results.append({
                'Model': model_name,
                'N Features': n,
                'Accuracy': accuracy,
                'Std': accuracy_std,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
                'Features': ', '.join(selected_features[:5]) + ('...' if n > 5 else '')
            })

    results_df = pd.DataFrame(results)

    # Створюємо візуалізації для різних метрик
    for metric_name, metric_label in metrics.items():
        if metric_name == 'accuracy':
            metric_col = 'Accuracy'
        elif metric_name == 'precision':
            metric_col = 'Precision'
        elif metric_name == 'recall':
            metric_col = 'Recall'
        elif metric_name == 'f1':
            metric_col = 'F1 Score'
        elif metric_name == 'roc_auc':
            metric_col = 'ROC AUC'
        else:
            continue

        plt.figure(figsize=(12, 8))
        for model in models.keys():
            model_results = results_df[results_df['Model'] == model]
            plt.plot(model_results['N Features'], model_results[metric_col],
                     marker='o', linestyle='-', label=model)

        plt.title(f"{metric_label} залежно від кількості ознак")
        plt.xlabel("Кількість ознак")
        plt.ylabel(metric_label)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{metric_name}_vs_features.png")

    return results_df

# Основна функція
def main():
    logger.info("Запуск скрипту вибору ознак...")

    try:
        # 1. Завантажуємо CSV-файл
        logger.info("Завантажуємо дані з CSV-файлу...")
        df = pd.read_csv("cleanest_data.csv")
        logger.info(f"Завантажено {df.shape[0]} рядків та {df.shape[1]} стовпців")

        # Аналіз дисбалансу класів
        class_counts = df['is_successful'].value_counts()
        logger.info(f"Розподіл класів: {class_counts.to_dict()}")
        if class_counts.min() / class_counts.max() < 0.2:
            logger.warning("Виявлено суттєвий дисбаланс класів")

        # 2. Перевірка та обробка пропущених значень
        logger.info("Перевіряємо наявність пропущених значень...")
        missing_values = df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        if not columns_with_missing.empty:
            logger.warning(f"Знайдено стовпці з пропущеними значеннями: {columns_with_missing}")
            num_imputer = SimpleImputer(strategy='median')
            cat_imputer = SimpleImputer(strategy='most_frequent')
            num_columns = df.select_dtypes(include=['number']).columns
            cat_columns = df.select_dtypes(include=['object']).columns
            df[num_columns] = num_imputer.fit_transform(df[num_columns])
            df[cat_columns] = cat_imputer.fit_transform(df[cat_columns])
            logger.info("Пропущені значення заповнено")
        else:
            logger.info("Пропущених значень не виявлено")

        # 3. Кодуємо категоріальні змінні
        logger.info("Кодуємо категоріальні змінні...")
        cat_columns = df.select_dtypes(include=['object']).columns.tolist()
        ignore_cols = ['order_id', 'state']
        cat_columns = [col for col in cat_columns if col not in ignore_cols]
        df_encoded = df.copy()
        encoders = {}
        for col in cat_columns:
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
            encoders[col] = encoder

        # 1. create_date: перетворення у місяці від найранішої дати
        try:
            df['create_date'] = pd.to_datetime(df['create_date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            min_date = df['create_date'].min()
            df['create_date_months'] = ((df['create_date'] - min_date).dt.days / 30.44).round(
                2)  # приблизне число місяців
            logger.info(f"Додано числову колонку 'create_date_months': місяці від {min_date}")
        except Exception as e:
            logger.warning(f"Помилка при обробці create_date: {e}")

        # 4. Вибираємо числові змінні
        num_columns = df.select_dtypes(include=['number']).columns.tolist()
        if 'is_successful' in num_columns:
            num_columns.remove('is_successful')

        # 5. Формуємо повний набір ознак
        X_full = df_encoded[num_columns + cat_columns]
        y = df_encoded['is_successful']

        # 6. Оцінюємо важливість ознак
        results_df, models_dict = evaluate_features(X_full, y, cv=5)

        # 7. Оцінюємо моделі з різною кількістю ознак
        # performance_df = evaluate_models_with_features(X_full, y, results_df)

        # 8. Візуалізація результатів
        logger.info("Створюємо візуалізації...")

        # 8.1 Візуалізація ТОП-15 найважливіших ознак
        plt.figure(figsize=(12, 8))
        top_features = results_df[['Feature', 'Total Importance Rank']].head(15)
        # Додаємо українські назви
        top_features['Feature_UA'] = top_features['Feature'].apply(get_ua_feature_name)
        sns.barplot(x='Total Importance Rank', y='Feature_UA', hue='Feature_UA', data=top_features, palette='coolwarm', legend=False)
        plt.xlabel("Total Importance Rank (Чим більший – тим важливіше)")
        plt.ylabel("Ознаки")
        plt.title("ТОП-15 найважливіших ознак")
        plt.tight_layout()
        plt.savefig(f"{results_dir}/top_features.png")

        # 8.2 Візуалізація розподілу оцінок для ТОП-10 ознак
        top10_features = results_df['Feature'].head(10).tolist()
        scores_data = []
        # Змінюємо порядок методів відповідно до ТЗ
        methods = ['MI Score', 'F Score', 'Spearman Score', 'RFE Selected', 'Absolute Coefficient', 'DT Score', 'RF Score', 'XGBoost Score']
        for method in methods:
            for feature in top10_features:
                score = results_df.loc[results_df['Feature'] == feature, method].values[0]
                max_score = results_df[method].max()
                norm_score = score / max_score if max_score > 0 else 0
                # Додаємо українську назву
                feature_ua = get_ua_feature_name(feature)
                scores_data.append({'Feature': feature, 'Feature_UA': feature_ua, 'Method': method, 'Normalized Score': norm_score})
        scores_df = pd.DataFrame(scores_data)

        # Визначаємо кольори для методів
        method_colors = {
            'MI Score': 'blue',           # синій
            'F Score': 'red',             # червоний
            'Spearman Score': 'skyblue',  # блакитний
            'RFE Selected': 'purple',     # пурпурний
            'Absolute Coefficient': 'orange',  # помаранчовий
            'DT Score': 'lightgreen',     # світло зелений
            'RF Score': 'forestgreen',          # середньо зелений
            'XGBoost Score': 'darkgreen'  # темно зелений
        }

        plt.figure(figsize=(14, 10))
        # Створюємо власну палітру кольорів
        palette = [method_colors[method] for method in scores_df['Method'].unique()]
        sns.barplot(x='Feature_UA', y='Normalized Score', hue='Method', data=scores_df, palette=palette)
        plt.xticks(rotation=45, ha='right')
        plt.title("Порівняння нормалізованих оцінок для ТОП-10 ознак за різними методами")
        plt.tight_layout()
        plt.savefig(f"{results_dir}/feature_scores_comparison.png")

        # 8.3 Візуалізація продуктивності моделей
        # plt.figure(figsize=(12, 8))
        # sns.lineplot(x='N Features', y='Accuracy', hue='Model', data=performance_df, marker='o')
        # plt.title("Точність моделей залежно від кількості ознак")
        # plt.xlabel("Кількість ознак")
        # plt.ylabel("Точність (Accuracy)")
        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.tight_layout()
        # plt.savefig(f"{results_dir}/model_performance.png")

        # 8.4 Візуалізація кореляційної матриці для ТОП-15 ознак
        plt.figure(figsize=(14, 12))
        top15_features = results_df['Feature'].head(15).tolist()
        correlation_matrix = X_full[top15_features].corr()

        # Створюємо DataFrame з українськими назвами для кореляційної матриці
        correlation_matrix_ua = correlation_matrix.copy()
        ua_index = [get_ua_feature_name(feature) for feature in correlation_matrix.index]
        ua_columns = [get_ua_feature_name(feature) for feature in correlation_matrix.columns]
        correlation_matrix_ua.index = ua_index
        correlation_matrix_ua.columns = ua_columns

        mask = np.triu(np.ones_like(correlation_matrix_ua, dtype=bool))
        sns.heatmap(correlation_matrix_ua, mask=mask, cmap='coolwarm', annot=True,
                    fmt='.2f', linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("Кореляційна матриця ТОП-15 ознак")
        plt.tight_layout()
        plt.savefig(f"{results_dir}/correlation_matrix.png")

        # 9. Зберігаємо результати
        logger.info("Зберігаємо результати...")
        results_df.to_csv(f"{results_dir}/feature_selection_results.csv", index=False)
        # performance_df.to_csv(f"{results_dir}/model_performance.csv", index=False)
        for model_name, model in models_dict.items():
            joblib.dump(model, f"{results_dir}/{model_name}_model.pkl")
        joblib.dump(encoders, f"{results_dir}/label_encoders.pkl")
        for n in [5, 10, 15, 20, 25]:
            if n <= len(results_df):
                top_n = results_df['Feature'].head(n).tolist()
                with open(f"{results_dir}/top_{n}_features.txt", 'w') as f:
                    for i, feature in enumerate(top_n, 1):
                        f.write(f"{i}. {feature}\n")

        # 10. Виводимо таблицю
        # Змінюємо порядок стовпців відповідно до ТЗ
        display_columns = ['Feature', 'MI Score', 'F Score', 'Spearman Score',
                           'LR Coefficient', 'DT Score', 'RF Score', 'XGBoost Score', 'Total Importance Rank']
        display_df = results_df[display_columns].round(4)

        # Додаємо українські назви ознак
        display_df['Feature_UA'] = display_df['Feature'].apply(get_ua_feature_name)
        # Переставляємо стовпець з українською назвою на друге місце
        cols = display_df.columns.tolist()
        cols.remove('Feature_UA')
        cols.insert(1, 'Feature_UA')
        display_df = display_df[cols]

        # Створюємо окремі таблиці для кожного методу з 5 найважливішими ознаками
        print("\nТаблиці з 5 найважливішими ознаками за кожним методом:\n")

        # Словник з назвами методів та відповідними стовпцями
        methods = {
            'Mutual Information': 'MI Score',
            'ANOVA F-test': 'F Score',
            'Spearman Correlation': 'Spearman Score',
            'Logistic Regression Coefficients': 'LR Coefficient',
            'Decision Tree Feature Importance': 'DT Score',
            'Random Forest Importance': 'RF Score',
            'XGBoost Feature Importance': 'XGBoost Score'
        }

        # Створюємо та виводимо окремі таблиці для кожного методу
        for method_name, method_column in methods.items():
            # Сортуємо дані за відповідним методом (у порядку спадання, тобто найважливіші ознаки будуть спочатку)
            if method_name == 'Logistic Regression Coefficients':
                # Для логістичної регресії сортуємо за абсолютними значеннями, але виводимо оригінальні
                method_df = results_df.copy()
                # Створюємо тимчасовий стовпець з абсолютними значеннями для сортування
                method_df['Abs_LR'] = method_df['LR Coefficient'].abs()
                # Сортуємо за абсолютними значеннями і беремо топ-5
                method_df = method_df.sort_values(by='Abs_LR', ascending=False).head(5)
                # Видаляємо тимчасовий стовпець
                method_df.drop('Abs_LR', axis=1, inplace=True)
            else:
                method_df = results_df.sort_values(by=method_column, ascending=False).head(5)

            # Додаємо українські назви
            method_df['Feature_UA'] = method_df['Feature'].apply(get_ua_feature_name)

            # Вибираємо та форматуємо дані для виведення
            display_method_df = method_df[['Feature', 'Feature_UA', method_column]].round(4)

            # Виводимо назву методу та дані через print замість logger.info
            print(f"\n{method_name} - 5 найважливіших ознак:")
            method_table = tabulate(display_method_df, headers=['Ознака', 'Назва українською', 'Значення'], tablefmt='fancy_grid')
            print(f"\n{method_table}")

            # Зберігаємо окрему таблицю у файл
            method_filename = method_name.lower().replace(' ', '_')
            display_method_df.to_csv(f"{results_dir}/{method_filename}_top5.csv", index=False)

        print("\nПорівняння впливу всіх факторів за рейтингами:\n")
        print(tabulate(display_df.head(20), headers='keys', tablefmt='fancy_grid'))
        print(f"\nНайкраща продуктивність моделей:")
        # best_performance = performance_df.loc[performance_df.groupby('Model')['Accuracy'].idxmax()]
        # print(tabulate(best_performance, headers='keys', tablefmt='fancy_grid'))
        logger.info(f"\nАналіз завершено успішно. Результати збережено у директорії: {results_dir}")

    except Exception as e:
        logger.error(f"Помилка під час виконання скрипту: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
