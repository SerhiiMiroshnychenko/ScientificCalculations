import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # Встановлюємо Agg бекенд
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats
from tabulate import tabulate
import os
import json
import datetime
import logging

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Створюємо директорію для збереження результатів
results_dir = "combined_feature_importance_results"
os.makedirs(results_dir, exist_ok=True)

# Словник перекладів назв ознак
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
    'source': 'Джерело замовлення'
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

def load_data(file_path, group_column='is_successful'):
    """
    Завантажує дані з файлу

    Args:
        file_path (str): Шлях до файлу даних
        group_column (str): Назва колонки для групування

    Returns:
        pd.DataFrame: Завантажений датафрейм
        list: Список колонок для аналізу
    """
    logger.info(f"Завантаження даних з {file_path}...")

    # Завантаження даних
    df = pd.read_csv(file_path)

    # Перетворення стовпця групування на числовий тип (0 або 1), якщо це ще не зроблено
    df[group_column] = df[group_column].astype(int)

    # Визначення колонок для аналізу (всі числові колонки)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Видаляємо колонку групування зі списку, якщо вона числова
    if group_column in numeric_columns:
        numeric_columns.remove(group_column)

    # Замінюємо від'ємні значення на 0 в числових колонках
    for col in numeric_columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            logger.info(f"Знайдено {negative_count} від'ємних значень у колонці {col}. Замінюємо їх на 0.")
            df[col] = df[col].apply(lambda x: max(0, x) if not pd.isna(x) else x)

    logger.info(f"Завантажено {len(df)} записів.")
    logger.info(f"Для аналізу обрано {len(numeric_columns)} числових колонок.")

    # Видалення ідентифікаторів та дат з набору ознак
    exclude_columns = [group_column, 'id', 'order_id', 'partner_id', 'date', 'timestamp']
    feature_columns = [col for col in numeric_columns if col not in exclude_columns]

    logger.info(f"Після фільтрації залишилось {len(feature_columns)} ознак для аналізу.")

    # Обробка категоріальних ознак
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [col for col in categorical_features if col not in exclude_columns]

    if categorical_features:
        logger.info(f"Знайдено {len(categorical_features)} категоріальних ознак")
        for feature in categorical_features:
            logger.info(f"Кодування категоріальної ознаки: {feature}")
            df[feature] = LabelEncoder().fit_transform(df[feature].astype(str))
        feature_columns.extend(categorical_features)

    # Обробка пропущених значень
    if df[feature_columns].isnull().sum().sum() > 0:
        logger.info("Заповнення пропущених значень...")
        numeric_features = df[feature_columns].select_dtypes(include=['number']).columns
        df[numeric_features] = SimpleImputer(strategy='median').fit_transform(df[numeric_features])

        # Якщо залишились пропущені значення в нечислових колонках
        non_numeric_features = [col for col in feature_columns if col not in numeric_features]
        if non_numeric_features and df[non_numeric_features].isnull().sum().sum() > 0:
            for col in non_numeric_features:
                df[col] = SimpleImputer(strategy='most_frequent').fit_transform(df[[col]])

    return df, feature_columns

def format_value(value):
    """Форматує числове значення для відображення"""
    if pd.isna(value):
        return "N/A"
    elif isinstance(value, (int, np.integer)):
        return f"{value:,d}"
    elif abs(value) < 0.0001 or abs(value) >= 10000:
        return f"{value:.4e}"
    else:
        return f"{value:.4f}"

# --- Методи статистичного аналізу з all_columns_statistics.py ---

def calculate_basic_stats(df, column_name, group_column='is_successful', group_names=['Неуспішні', 'Успішні']):
    """
    Розраховує базові статистичні показники для двох груп даних

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        group_column (str): Назва колонки для групування
        group_names (list): Назви груп

    Returns:
        pd.DataFrame: Датафрейм із статистичними показниками
    """
    # Розділення даних на групи
    group_0 = df[df[group_column] == 0][column_name].dropna()
    group_1 = df[df[group_column] == 1][column_name].dropna()

    # Пропускаємо, якщо даних недостатньо
    if len(group_0) < 2 or len(group_1) < 2:
        return None

    # Створення словника з даними для кожної групи
    data_dict = {
        group_names[0]: group_0,
        group_names[1]: group_1
    }

    # Створення датафрейму
    stats_df = pd.DataFrame()

    # Розрахунок базових статистик для кожної групи
    for group_name, data in data_dict.items():
        group_stats = {
            'Кількість': len(data),
            'Середнє': data.mean(),
            'Медіана': data.median(),
            'Стандартне відхилення': data.std(),
            'Коефіцієнт варіації': data.std() / data.mean() if data.mean() != 0 else np.nan,
            'Мінімум': data.min(),
            'Максимум': data.max(),
            'Квартиль 25%': data.quantile(0.25),
            'Квартиль 75%': data.quantile(0.75),
            'Коефіцієнт асиметрії': stats.skew(data),
            'Ексцес': stats.kurtosis(data)
        }

        # Додавання статистик до датафрейму
        if stats_df.empty:
            stats_df = pd.DataFrame(group_stats, index=[group_name])
        else:
            stats_df.loc[group_name] = group_stats

    return stats_df

def perform_statistical_tests(df, column_name, group_column='is_successful', alpha=0.05):
    """
    Проводить статистичні тести для порівняння двох груп даних

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column_name (str): Назва колонки для аналізу
        group_column (str): Назва колонки для групування
        alpha (float): Рівень значущості

    Returns:
        pd.DataFrame: Датафрейм із результатами тестів
        pd.DataFrame: Датафрейм із довірчими інтервалами
        float: p-значення t-тесту
        float: p-значення тесту Манна-Уітні
    """
    # Розділення даних на групи
    group_0 = df[df[group_column] == 0][column_name].dropna()
    group_1 = df[df[group_column] == 1][column_name].dropna()

    # Пропускаємо, якщо даних недостатньо
    if len(group_0) < 2 or len(group_1) < 2:
        return None, None, np.nan, np.nan

    # t-тест для незалежних вибірок
    try:
        t_stat, t_pvalue = stats.ttest_ind(group_0, group_1, equal_var=False)
    except:
        t_stat, t_pvalue = np.nan, np.nan

    # Тест Манна-Уітні
    try:
        mw_stat, mw_pvalue = stats.mannwhitneyu(group_0, group_1)
    except:
        mw_stat, mw_pvalue = np.nan, np.nan

    # Розрахунок довірчих інтервалів
    try:
        ci_0 = stats.t.interval(1-alpha, len(group_0)-1, loc=group_0.mean(), scale=stats.sem(group_0))
        ci_1 = stats.t.interval(1-alpha, len(group_1)-1, loc=group_1.mean(), scale=stats.sem(group_1))
    except:
        ci_0 = (np.nan, np.nan)
        ci_1 = (np.nan, np.nan)

    # Створення датафрейму із результатами
    tests_results = pd.DataFrame({
        'Тест': ['t-тест (Welch)', 'Тест Манна-Уітні'],
        'Статистика': [t_stat, mw_stat],
        'p-значення': [t_pvalue, mw_pvalue],
        'Значущість': [
            "Значуща різниця" if p < alpha and not np.isnan(p) else "Немає значущої різниці"
            for p in [t_pvalue, mw_pvalue]
        ]
    })

    # Результати довірчих інтервалів
    ci_results = pd.DataFrame({
        'Група': ['Неуспішні', 'Успішні'],
        'Середнє': [group_0.mean(), group_1.mean()],
        'Нижня межа CI': [ci_0[0], ci_1[0]],
        'Верхня межа CI': [ci_0[1], ci_1[1]]
    })

    return tests_results, ci_results, t_pvalue, mw_pvalue

def calculate_cohen_d(group_0_values, group_1_values):
    """
    Розраховує розмір ефекту d Коена

    Args:
        group_0_values (pd.Series): Значення для першої групи
        group_1_values (pd.Series): Значення для другої групи

    Returns:
        float: Значення d Коена
    """
    # Перевірка на достатню кількість даних
    if len(group_0_values) < 2 or len(group_1_values) < 2:
        return np.nan

    # Розрахунок середніх та стандартних відхилень
    mean_0 = group_0_values.mean()
    mean_1 = group_1_values.mean()
    std_0 = group_0_values.std()
    std_1 = group_1_values.std()

    # Перевірка на валідні значення
    if np.isnan(mean_0) or np.isnan(mean_1) or np.isnan(std_0) or np.isnan(std_1):
        return np.nan

    # Розрахунок об'єднаного стандартного відхилення
    pooled_std = np.sqrt((std_0**2 + std_1**2) / 2)

    # Уникнення ділення на нуль
    if pooled_std == 0:
        return np.nan

    # Розрахунок d Коена
    cohen_d = abs(mean_0 - mean_1) / pooled_std

    return cohen_d

def calculate_auc(df, column, target='is_successful'):
    """
    Розраховує AUC для окремої колонки як предиктора

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column (str): Назва колонки-предиктора
        target (str): Назва цільової змінної

    Returns:
        float: Значення AUC
    """
    # Видаляємо рядки з пропущеними значеннями
    valid_data = df[[column, target]].dropna()

    # Пропускаємо колонки з константними значеннями
    if len(valid_data[column].unique()) <= 1:
        return 0.5

    try:
        # Розрахунок AUC
        auc = roc_auc_score(valid_data[target], valid_data[column])

        # Якщо AUC < 0.5, це означає негативну кореляцію
        # Інвертуємо значення для отримання позитивної метрики
        return max(auc, 1 - auc)
    except:
        return 0.5  # У випадку помилки повертаємо нейтральне значення

def calculate_iv(df, column, target='is_successful', bins=10):
    """
    Розраховує Information Value для числової колонки

    Args:
        df (pd.DataFrame): Датафрейм з даними
        column (str): Назва колонки для аналізу
        target (str): Назва цільової змінної
        bins (int): Кількість бінів для дискретизації

    Returns:
        float: Значення IV
    """
    # Видаляємо рядки з пропущеними значеннями
    valid_data = df[[column, target]].dropna()

    # Перевірка на порожні значення
    if valid_data.empty or len(valid_data[column].unique()) <= 1:
        return 0

    # Створюємо копію для уникнення попереджень
    df_temp = valid_data.copy()

    try:
        # Дискретизація числової колонки
        df_temp['bin'] = pd.qcut(df_temp[column], bins, duplicates='drop')
    except:
        # Якщо дискретизація не вдалася, використовуємо унікальні значення
        df_temp['bin'] = df_temp[column]

    # Розрахунок WoE та IV
    grouped = df_temp.groupby('bin')[target].agg(['count', 'sum'])
    grouped['non_target'] = grouped['count'] - grouped['sum']

    # Уникаємо ділення на нуль
    if grouped['sum'].sum() == 0 or grouped['non_target'].sum() == 0:
        return 0

    grouped['target_rate'] = grouped['sum'] / grouped['sum'].sum()
    grouped['non_target_rate'] = grouped['non_target'] / grouped['non_target'].sum()

    # Заміна нульових значень на малі числа для уникнення log(0)
    grouped['target_rate'] = grouped['target_rate'].replace(0, 0.0001)
    grouped['non_target_rate'] = grouped['non_target_rate'].replace(0, 0.0001)

    grouped['woe'] = np.log(grouped['target_rate'] / grouped['non_target_rate'])
    grouped['iv'] = (grouped['target_rate'] - grouped['non_target_rate']) * grouped['woe']

    return grouped['iv'].sum()

# --- Методи машинного навчання з feature_importance_ranking.py ---

def calculate_mutual_information(X, y):
    """
    Обчислює Mutual Information між ознаками та цільовою змінною

    Args:
        X (pd.DataFrame): Набір ознак
        y (pd.Series): Цільова змінна

    Returns:
        pd.Series: Значення Mutual Information для кожної ознаки
    """
    logger.info("Обчислюємо Mutual Information...")
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42, discrete_features='auto')
        return pd.Series(mi_scores, index=X.columns)
    except Exception as e:
        logger.error(f"Помилка при обчисленні Mutual Information: {str(e)}")
        return pd.Series(np.nan, index=X.columns)

def calculate_f_statistics(X, y):
    """
    Обчислює ANOVA F-test

    Args:
        X (pd.DataFrame): Набір ознак
        y (pd.Series): Цільова змінна

    Returns:
        pd.Series: Значення F-test для кожної ознаки
    """
    logger.info("Обчислюємо ANOVA F-test...")
    try:
        f_scores, _ = f_classif(X, y)
        return pd.Series(f_scores, index=X.columns)
    except Exception as e:
        logger.error(f"Помилка при обчисленні ANOVA F-test: {str(e)}")
        return pd.Series(np.nan, index=X.columns)

def calculate_spearman_correlation(X, y):
    """
    Обчислює Spearman Correlation між ознаками та цільовою змінною

    Args:
        X (pd.DataFrame): Набір ознак
        y (pd.Series): Цільова змінна

    Returns:
        pd.Series: Значення Spearman Correlation для кожної ознаки
    """
    logger.info("Обчислюємо Spearman Correlation...")
    try:
        spearman_corr = X.corrwith(y, method="spearman").abs()
        return spearman_corr
    except Exception as e:
        logger.error(f"Помилка при обчисленні Spearman Correlation: {str(e)}")
        return pd.Series(np.nan, index=X.columns)

def calculate_logistic_regression(X, y):
    """
    Обчислює коефіцієнти логістичної регресії

    Args:
        X (pd.DataFrame): Набір ознак
        y (pd.Series): Цільова змінна

    Returns:
        pd.Series: Абсолютні значення коефіцієнтів для кожної ознаки
    """
    logger.info("Обчислюємо Logistic Regression Coefficients...")
    try:
        # Стандартизація даних
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Логістична регресія
        model_lr = LogisticRegression(max_iter=10000, random_state=42, solver='liblinear', class_weight='balanced')
        model_lr.fit(X_scaled, y)

        # Абсолютні значення коефіцієнтів
        abs_coefficients = np.abs(model_lr.coef_[0])
        return pd.Series(abs_coefficients, index=X.columns)
    except Exception as e:
        logger.error(f"Помилка при обчисленні Logistic Regression Coefficients: {str(e)}")
        return pd.Series(np.nan, index=X.columns)

def calculate_decision_tree(X, y, cv=5):
    """
    Обчислює важливість ознак за допомогою Decision Tree з крос-валідацією

    Args:
        X (pd.DataFrame): Набір ознак
        y (pd.Series): Цільова змінна
        cv (int): Кількість фолдів для крос-валідації

    Returns:
        pd.Series: Значення важливості для кожної ознаки
    """
    logger.info("Обчислюємо Decision Tree Feature Importance з крос-валідацією...")
    try:
        model_dt = DecisionTreeClassifier(random_state=42)
        dt_importances = []

        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        for train_idx, val_idx in cv_obj.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            model_dt.fit(X_train, y_train)
            dt_importances.append(model_dt.feature_importances_)

        dt_importance = np.mean(dt_importances, axis=0)
        return pd.Series(dt_importance, index=X.columns)
    except Exception as e:
        logger.error(f"Помилка при обчисленні Decision Tree Importance: {str(e)}")
        return pd.Series(np.nan, index=X.columns)

def calculate_random_forest(X, y, cv=5):
    """
    Обчислює важливість ознак за допомогою Random Forest з крос-валідацією

    Args:
        X (pd.DataFrame): Набір ознак
        y (pd.Series): Цільова змінна
        cv (int): Кількість фолдів для крос-валідації

    Returns:
        pd.Series: Значення важливості для кожної ознаки
    """
    logger.info("Обчислюємо Random Forest Importance з крос-валідацією...")
    try:
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        importances = []

        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        for train_idx, val_idx in cv_obj.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            model_rf.fit(X_train, y_train)
            importances.append(model_rf.feature_importances_)

        rf_importance = np.mean(importances, axis=0)
        return pd.Series(rf_importance, index=X.columns)
    except Exception as e:
        logger.error(f"Помилка при обчисленні Random Forest Importance: {str(e)}")
        return pd.Series(np.nan, index=X.columns)

def calculate_xgboost(X, y, cv=5):
    """
    Обчислює важливість ознак за допомогою XGBoost з крос-валідацією

    Args:
        X (pd.DataFrame): Набір ознак
        y (pd.Series): Цільова змінна
        cv (int): Кількість фолдів для крос-валідації

    Returns:
        pd.Series: Значення важливості для кожної ознаки
    """
    logger.info("Обчислюємо XGBoost Feature Importance з крос-валідацією...")
    try:
        model_xgb = xgb.XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1,
                                      scale_pos_weight=(len(y) - sum(y)) / sum(y))
        importances = []

        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        for train_idx, val_idx in cv_obj.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            model_xgb.fit(X_train, y_train)
            importances.append(model_xgb.feature_importances_)

        xgb_importance = np.mean(importances, axis=0)
        return pd.Series(xgb_importance, index=X.columns)
    except Exception as e:
        logger.error(f"Помилка при обчисленні XGBoost Importance: {str(e)}")
        return pd.Series(np.nan, index=X.columns)

# --- Функції для комбінованого рейтингу та візуалізації ---

def calculate_all_importance_metrics(df, features, target_column='is_successful'):
    """
    Розраховує всі метрики важливості ознак

    Args:
        df (pd.DataFrame): Датафрейм з даними
        features (list): Список ознак для аналізу
        target_column (str): Назва цільової колонки

    Returns:
        dict: Словник з результатами для кожної метрики
    """
    logger.info("Початок розрахунку всіх метрик важливості ознак...")

    X = df[features]
    y = df[target_column]

    # Словник для зберігання результатів
    results = {}

    # 1. Статистичні тести
    for column in features:
        # Розрахунок статистичних тестів для кожної колонки
        _, _, t_pvalue, mw_pvalue = perform_statistical_tests(df, column, target_column)

        # Збереження p-значень
        if 't_test' not in results:
            results['t_test'] = {}
        if 'mann_whitney' not in results:
            results['mann_whitney'] = {}

        results['t_test'][column] = t_pvalue
        results['mann_whitney'][column] = mw_pvalue

    # Перетворення на Series
    results['t_test'] = pd.Series(results['t_test'])
    results['mann_whitney'] = pd.Series(results['mann_whitney'])

    # 2. Метрики ефекту
    results['cohen_d'] = pd.Series({
        column: calculate_cohen_d(
            df[df[target_column] == 0][column].dropna(),
            df[df[target_column] == 1][column].dropna()
        ) for column in features
    })

    results['auc'] = pd.Series({
        column: calculate_auc(df, column, target_column) for column in features
    })

    results['iv'] = pd.Series({
        column: calculate_iv(df, column, target_column) for column in features
    })

    # 3. Методи машинного навчання
    results['mutual_info'] = calculate_mutual_information(X, y)
    results['f_statistic'] = calculate_f_statistics(X, y)
    results['spearman'] = calculate_spearman_correlation(X, y)
    results['logistic'] = calculate_logistic_regression(X, y)
    results['decision_tree'] = calculate_decision_tree(X, y)
    results['random_forest'] = calculate_random_forest(X, y)
    results['xgboost'] = calculate_xgboost(X, y)

    logger.info("Завершено розрахунок всіх метрик важливості ознак")
    return results

def calculate_combined_ranking(all_metrics_results):
    """
    Обчислює комбінований рейтинг ознак на основі всіх метрик

    Args:
        all_metrics_results (dict): Словник з результатами для кожної метрики

    Returns:
        pd.DataFrame: Датафрейм з комбінованим рейтингом
    """
    logger.info("Обчислення комбінованого рейтингу ознак...")

    # Створюємо DataFrame для зберігання результатів
    results_df = pd.DataFrame(index=all_metrics_results[list(all_metrics_results.keys())[0]].index)

    # Додаємо результати кожної метрики як окрему колонку
    for metric_name, metric_results in all_metrics_results.items():
        results_df[metric_name] = metric_results

    # Замінюємо NaN на значення, які не впливатимуть на рейтинг
    for metric in ['t_test', 'mann_whitney']:
        results_df[metric] = results_df[metric].fillna(1.0)  # p-значення, менше краще, max = 1.0

    for metric in ['cohen_d', 'auc', 'iv', 'mutual_info', 'f_statistic', 'spearman',
                   'logistic', 'decision_tree', 'random_forest', 'xgboost']:
        results_df[metric] = results_df[metric].fillna(0.0)  # решта метрик, більше краще, min = 0.0

    # Розраховуємо ранги для кожної метрики (1 = найкраща ознака)
    rankings_df = pd.DataFrame(index=results_df.index)

    # Для метрик, де менше значення краще (p-значення)
    for metric in ['t_test', 'mann_whitney']:
        rankings_df[f'{metric}_rank'] = results_df[metric].rank(method='average')

    # Для метрик, де більше значення краще
    for metric in ['cohen_d', 'auc', 'iv', 'mutual_info', 'f_statistic', 'spearman',
                   'logistic', 'decision_tree', 'random_forest', 'xgboost']:
        rankings_df[f'{metric}_rank'] = results_df[metric].rank(method='average', ascending=False)

    # Обчислюємо середній ранг
    rank_columns = [col for col in rankings_df.columns if col.endswith('_rank')]
    rankings_df['avg_rank'] = rankings_df[rank_columns].mean(axis=1)

    # Нормалізуємо до 100%
    min_avg_rank = rankings_df['avg_rank'].min()
    max_avg_rank = rankings_df['avg_rank'].max()
    rankings_df['importance_score'] = 100 - ((rankings_df['avg_rank'] - min_avg_rank) /
                                             (max_avg_rank - min_avg_rank) * 100)

    # Сортуємо за важливістю (більше = важливіше)
    rankings_df = rankings_df.sort_values('importance_score', ascending=False)

    # Додаємо оригінальні значення метрик для аналізу
    for metric in all_metrics_results.keys():
        rankings_df[metric] = results_df[metric]

    logger.info("Завершено обчислення комбінованого рейтингу ознак")
    return rankings_df

def plot_feature_importance(rankings_df, top_n=15, title='Важливість ознак', save_path=None):
    """
    Будує графік важливості ознак

    Args:
        rankings_df (pd.DataFrame): DataFrame з результатами рейтингу
        top_n (int): Кількість найважливіших ознак для відображення
        title (str): Заголовок графіка
        save_path (str): Шлях для збереження графіка, якщо None - не зберігати
    """
    plt.figure(figsize=(12, 8))

    # Вибираємо топ-N ознак
    plot_df = rankings_df.head(top_n).copy()

    # Додаємо українські назви для ознак
    plot_df['Feature_UA'] = plot_df.index.map(get_ua_feature_name)

    # Будуємо графік
    plt.barh(plot_df['Feature_UA'], plot_df['importance_score'], color='skyblue')
    plt.xlabel('Відносна важливість (%)')
    plt.ylabel('Ознака')
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()  # Найважливіша ознака зверху

    # Додаємо значення на графіку
    for i, v in enumerate(plot_df['importance_score']):
        plt.text(v + 1, i, f"{v:.1f}%", va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Графік збережено: {save_path}")

    plt.close()

def plot_metrics_comparison(rankings_df, feature, save_path=None):
    """
    Будує графік порівняння різних метрик для конкретної ознаки

    Args:
        rankings_df (pd.DataFrame): DataFrame з результатами рейтингу
        feature (str): Назва ознаки для аналізу
        save_path (str): Шлях для збереження графіка, якщо None - не зберігати
    """
    # Вибираємо метрики для відображення (без рангів)
    metrics_to_plot = [col for col in rankings_df.columns
                       if not col.endswith('_rank')
                       and col not in ['avg_rank', 'importance_score']]

    # Нормалізуємо значення для порівняння
    normalized_values = {}
    for metric in metrics_to_plot:
        # Для p-значень інвертуємо (1 - p)
        if metric in ['t_test', 'mann_whitney']:
            max_val = rankings_df[metric].max()
            normalized_values[metric] = 1 - (rankings_df.loc[feature, metric] / max_val)
        else:
            # Для решти метрик - звичайна нормалізація
            max_val = rankings_df[metric].max()
            if max_val > 0:
                normalized_values[metric] = rankings_df.loc[feature, metric] / max_val
            else:
                normalized_values[metric] = 0

    # Створення графіка
    plt.figure(figsize=(12, 6))
    metrics_names = list(normalized_values.keys())
    values = list(normalized_values.values())

    plt.bar(metrics_names, values, color='lightblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Нормалізована важливість')
    plt.title(f'Важливість ознаки "{feature}" за різними метриками')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Додаємо значення на графіку
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Графік метрик для {feature} збережено: {save_path}")

    plt.close()

def plot_heatmap(rankings_df, top_n=15, save_path=None):
    """
    Будує теплову карту важливості ознак за різними метриками

    Args:
        rankings_df (pd.DataFrame): DataFrame з результатами рейтингу
        top_n (int): Кількість найважливіших ознак для відображення
        save_path (str): Шлях для збереження графіка, якщо None - не зберігати
    """
    # Вибираємо топ-N ознак за загальним рейтингом
    top_features = rankings_df.head(top_n).index.tolist()

    # Вибираємо метрики для відображення (без рангів)
    metrics_to_plot = [col for col in rankings_df.columns
                       if not col.endswith('_rank')
                       and col not in ['avg_rank', 'importance_score']]

    # Створюємо новий DataFrame для теплової карти
    heatmap_df = rankings_df.loc[top_features, metrics_to_plot].copy()

    # Нормалізуємо значення для кожної метрики окремо
    for metric in metrics_to_plot:
        if metric in ['t_test', 'mann_whitney']:
            # Для p-значень: менше = краще, інвертуємо
            max_val = heatmap_df[metric].max()
            heatmap_df[metric] = 1 - (heatmap_df[metric] / max_val)
        else:
            # Для решти метрик: більше = краще
            max_val = heatmap_df[metric].max()
            if max_val > 0:
                heatmap_df[metric] = heatmap_df[metric] / max_val

    # Додаємо українські назви
    heatmap_df.index = [f"{idx} ({get_ua_feature_name(idx)})" for idx in heatmap_df.index]

    # Створення теплової карти
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_df, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5)
    plt.title('Важливість ознак за різними метриками (нормалізовано)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Теплову карту збережено: {save_path}")

    plt.close()

def print_importance_table(rankings_df, top_n=20):
    """
    Виводить таблицю важливості ознак

    Args:
        rankings_df (pd.DataFrame): DataFrame з результатами рейтингу
        top_n (int): Кількість найважливіших ознак для відображення
    """
    print("\n=== Загальний рейтинг важливості ознак ===")

    # Підготовка даних для таблиці
    display_df = rankings_df.head(top_n).copy()
    display_df.index.name = 'Feature'
    display_df.reset_index(inplace=True)

    # Додаємо українські назви
    display_df['Feature_UA'] = display_df['Feature'].apply(get_ua_feature_name)

    # Додаємо ранг
    display_df.insert(0, 'Rank', range(1, len(display_df) + 1))

    # Вибираємо лише потрібні колонки
    display_columns = ['Rank', 'Feature', 'Feature_UA', 'importance_score', 'avg_rank']

    # Додаємо вибрані метрики для відображення
    display_metrics = ['t_test', 'cohen_d', 'auc', 'iv', 'random_forest', 'xgboost']
    display_columns.extend(display_metrics)

    # Створюємо копію для відображення
    table_df = display_df[display_columns].copy()

    # Перейменовуємо колонки для зручності
    column_names = {
        'Rank': 'Ранг',
        'Feature': 'Ознака',
        'Feature_UA': 'Назва українською',
        'importance_score': 'Важливість (%)',
        'avg_rank': 'Середній ранг',
        't_test': 'p-значення t-тесту',
        'cohen_d': 'd Коена',
        'auc': 'AUC',
        'iv': 'IV',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost'
    }
    table_df = table_df.rename(columns=column_names)

    # Форматуємо значення - спеціальний підхід для p-значень
    formatted_table = table_df.copy()

    # Перетворюємо p-значення в науковий формат
    if 'p-значення t-тесту' in formatted_table.columns:
        formatted_table['p-значення t-тесту'] = formatted_table['p-значення t-тесту'].apply(
            lambda x: f"{x:.2e}" if not pd.isna(x) else "N/A")

    # Інші колонки форматуємо звичайним чином
    for col in formatted_table.columns:
        if col not in ['Ранг', 'Ознака', 'Назва українською', 'p-значення t-тесту']:
            formatted_table[col] = formatted_table[col].apply(
                lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

    # Виведення таблиці з використанням форматованих значень
    print(tabulate(formatted_table, headers='keys', tablefmt='fancy_grid', showindex=False))

def print_method_rankings(rankings_df, method, top_n=10):
    """
    Виводить рейтинг ознак за окремим методом

    Args:
        rankings_df (pd.DataFrame): DataFrame з результатами рейтингу
        method (str): Назва методу
        top_n (int): Кількість найважливіших ознак для відображення
    """
    # Вибираємо правильний порядок сортування
    ascending = True if method in ['t_test', 'mann_whitney'] else False

    # Створюємо копію та сортуємо за вказаним методом
    method_df = rankings_df.sort_values(by=method, ascending=ascending).head(top_n).copy()
    method_df.index.name = 'Feature'
    method_df.reset_index(inplace=True)

    # Додаємо українські назви та ранг
    method_df['Feature_UA'] = method_df['Feature'].apply(get_ua_feature_name)
    method_df.insert(0, 'Rank', range(1, len(method_df) + 1))

    # Визначення назви методу для відображення
    method_names = {
        't_test': 'p-значення t-тесту',
        'mann_whitney': 'p-значення тесту Манна-Уітні',
        'cohen_d': 'Розмір ефекту d Коена',
        'auc': 'Area Under Curve (AUC)',
        'iv': 'Information Value (IV)',
        'mutual_info': 'Mutual Information',
        'f_statistic': 'ANOVA F-тест',
        'spearman': 'Кореляція Спірмана',
        'logistic': 'Логістична регресія',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost'
    }

    method_title = method_names.get(method, method)
    print(f"\n=== Рейтинг за методом: {method_title} ===")

    # Вибираємо колонки для відображення
    display_columns = ['Rank', 'Feature', 'Feature_UA', method]
    table_df = method_df[display_columns].copy()

    # Перейменовуємо колонки
    column_names = {
        'Rank': 'Ранг',
        'Feature': 'Ознака',
        'Feature_UA': 'Назва українською',
        method: method_title
    }
    table_df = table_df.rename(columns=column_names)

    # Виведення таблиці
    # Використовуємо різний формат для p-значень та інших метрик
    if method in ['t_test', 'mann_whitney']:
        # Для p-значень використовуємо науковий формат з більшою точністю
        print(tabulate(table_df, headers='keys', tablefmt='fancy_grid', showindex=False,
                       floatfmt=".2e"))
    else:
        # Для інших метрик використовуємо звичайний формат
        print(tabulate(table_df, headers='keys', tablefmt='fancy_grid', showindex=False,
                       floatfmt=".4f"))

def main():
    """
    Головна функція програми
    """
    logger.info("Початок роботи програми комплексного аналізу важливості ознак")

    # Параметри
    data_file = 'cleaned_result.csv'
    target_column = 'is_successful'

    try:
        # Завантаження даних
        logger.info(f"Завантаження даних з {data_file}...")
        df, features = load_data(data_file, target_column)

        # Додаткова інформація про цільову змінну
        logger.info(f"Розподіл цільової змінної: {df[target_column].value_counts().to_dict()}")

        # Аналіз кореляцій між ознаками
        logger.info("Аналіз кореляцій між ознаками...")
        correlation_matrix = df[features].corr().abs()
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(upper.columns[i], upper.columns[j]) for i in range(len(upper.columns))
                           for j in range(len(upper.columns)) if i < j and upper.iloc[i, j] > 0.8]

        if high_corr_pairs:
            logger.warning(f"Виявлено {len(high_corr_pairs)} пар ознак з високою кореляцією (>0.8):")
            for pair in high_corr_pairs[:5]:
                logger.warning(f"  {pair[0]} <-> {pair[1]}: {correlation_matrix.loc[pair[0], pair[1]]:.3f}")
            if len(high_corr_pairs) > 5:
                logger.warning(f"  ... та {len(high_corr_pairs)-5} інших пар")

        # Розрахунок всіх метрик важливості ознак
        logger.info(f"Початок аналізу {len(features)} ознак...")
        all_metrics = calculate_all_importance_metrics(df, features, target_column)

        # Обчислення комбінованого рейтингу
        rankings_df = calculate_combined_ranking(all_metrics)

        # Збереження результатів
        rankings_path = f"{results_dir}/feature_importance_rankings.csv"
        rankings_df.to_csv(rankings_path)
        logger.info(f"Збережено результати рейтингу важливості ознак: {rankings_path}")

        # Виведення загального рейтингу
        print_importance_table(rankings_df, top_n=20)

        # Виведення рейтингів за окремими методами
        methods = [
            't_test', 'mann_whitney', 'cohen_d', 'auc', 'iv',
            'mutual_info', 'f_statistic', 'spearman',
            'logistic', 'decision_tree', 'random_forest', 'xgboost'
        ]

        for method in methods:
            print_method_rankings(rankings_df, method, top_n=10)

        # Створення візуалізацій

        # 1. Загальний рейтинг
        plot_feature_importance(
            rankings_df,
            title='Загальний рейтинг важливості ознак',
            save_path=f"{results_dir}/overall_importance.png"
        )

        # 2. Теплова карта
        plot_heatmap(
            rankings_df,
            top_n=15,
            save_path=f"{results_dir}/importance_heatmap.png"
        )

        # 3. Графіки для топ-5 ознак
        top_features = rankings_df.head(5).index.tolist()
        for feature in top_features:
            plot_metrics_comparison(
                rankings_df,
                feature,
                save_path=f"{results_dir}/{feature}_metrics_comparison.png"
            )

        # 4. Графіки для кожного методу
        for method in methods:
            method_df = rankings_df.sort_values(
                by=method,
                ascending=(method in ['t_test', 'mann_whitney'])
            ).copy()

            method_names = {
                't_test': 'p-значення t-тесту',
                'mann_whitney': 'p-значення тесту Манна-Уітні',
                'cohen_d': 'Розмір ефекту d Коена',
                'auc': 'AUC',
                'iv': 'IV',
                'mutual_info': 'Mutual Information',
                'f_statistic': 'ANOVA F-test',
                'spearman': 'Кореляція Спірмана',
                'logistic': 'Логістична регресія',
                'decision_tree': 'Decision Tree',
                'random_forest': 'Random Forest',
                'xgboost': 'XGBoost'
            }

            plot_feature_importance(
                method_df,
                title=f'Важливість ознак за методом: {method_names.get(method, method)}',
                save_path=f"{results_dir}/{method}_importance.png"
            )

        logger.info(f"Аналіз завершено. Всі результати збережено в директорії: {results_dir}")

    except Exception as e:
        logger.error(f"Помилка: {str(e)}", exc_info=True)
        return

if __name__ == "__main__":
    main()
