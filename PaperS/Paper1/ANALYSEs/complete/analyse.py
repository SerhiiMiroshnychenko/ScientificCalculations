# -*- coding: utf-8 -*-
"""
Аналіз значущості ознак для прогнозування успішності B2B-замовлень
з використанням п'яти різних методів оцінки

1. AUC (Area Under the Curve)
2. Mutual Information
3. Distance Correlation
4. Логістична регресія
5. Decision Tree
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import dcor  # потрібно встановити: pip install dcor
from scipy import stats
import warnings
import re

warnings.filterwarnings('ignore')

# Шлях до файлу з обробленими даними
file_path = 'preprocessed_data.csv'

# Завантаження даних
print("Завантаження даних...")
data = pd.read_csv(file_path)
print(f"Завантажено набір даних розміром {data.shape}")

# Перевірка наявності цільової змінної
if 'is_successful' not in data.columns:
    raise ValueError("У наборі даних відсутня цільова змінна 'is_successful'")

# Відокремлення цільової змінної від ознак
X = data.drop('is_successful', axis=1)
y = data['is_successful']

print(f"Аналізуємо {X.shape[1]} ознак для їх впливу на успішність замовлень")

# Перевірка на відсутність текстових даних
if X.select_dtypes(include=['object']).shape[1] > 0:
    print("Увага: виявлено текстові ознаки. Для аналізу будуть використані лише числові ознаки.")
    X = X.select_dtypes(exclude=['object'])

print(f"Після фільтрації залишилось {X.shape[1]} числових ознак")


# Функція для ідентифікації та групування циклічних ознак
def identify_cyclical_features(feature_names):
    """
    Ідентифікує і групує пари циклічних ознак (sin/cos)
    Повертає словник вигляду {'базова_назва': ['базова_назва_sin', 'базова_назва_cos']}
    """
    cyclical_pairs = {}
    sin_features = [f for f in feature_names if f.endswith('_sin')]
    
    for sin_feature in sin_features:
        base_name = sin_feature[:-4]  # Видаляємо '_sin'
        cos_feature = f"{base_name}_cos"
        
        if cos_feature in feature_names:
            # Створюємо базову назву без _sin/_cos
            cyclical_pairs[base_name] = [sin_feature, cos_feature]
    
    return cyclical_pairs


# Функція для створення DataFrame з обробленими ознаками (об'єднання циклічних)
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
        # Для кожної пари створюємо нову ознаку з іменем base_name
        # Використовуємо евклідову норму (корінь з суми квадратів) як міру важливості
        X_processed[base_name] = np.sqrt(X[sin_feature]**2 + X[cos_feature]**2)
    
    return X_processed


# Обробляємо ознаки, об'єднуючи циклічні
X_processed = preprocess_features_for_analysis(X)
print(f"Після обробки циклічних ознак маємо {X_processed.shape[1]} ознак")


# Функція для обчислення та впорядкування значущості ознак
def create_feature_importance_df(importance_values, feature_names, method_name):
    """
    Створює DataFrame з рейтингом значущості ознак та зберігає його у CSV
    """
    importance_df = pd.DataFrame({
        'Ознака': feature_names,
        'Значущість': importance_values
    })

    importance_df = importance_df.sort_values('Значущість', ascending=False)

    # Збереження результатів у CSV файл
    output_file = f'importance_{method_name}.csv'
    importance_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Результати методу {method_name} збережено у файл {output_file}")

    return importance_df


# Функція для візуалізації результатів
def plot_feature_importance(importance_df, method_name):
    """
    Створює візуалізацію всіх ознак
    """
    plt.figure(figsize=(14, max(10, len(importance_df) * 0.3)))  # Динамічно налаштовуємо висоту

    # Створюємо горизонтальну гістограму для всіх ознак
    sns.barplot(x='Значущість', y='Ознака', data=importance_df)

    plt.title(f'Значущість ознак за методом {method_name}')
    plt.tight_layout()

    # Збереження візуалізації
    output_file = f'importance_{method_name}_plot.png'
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Візуалізацію збережено у файл {output_file}")


#######################################
# 1. AUC (Area Under the Curve)
#######################################
print("\n1. Обчислення значущості ознак за методом AUC...")


def calculate_auc_importance(X, y):
    """
    Обчислює значущість ознак на основі AUC для кожної ознаки окремо
    """
    importances = []
    feature_names = X.columns

    for feature in feature_names:
        auc = roc_auc_score(y, X[feature])
        # Якщо AUC < 0.5, ознака має обернений вплив, беремо (1 - AUC)
        auc_corrected = max(auc, 1 - auc)
        importances.append(auc_corrected)

    return importances, feature_names


auc_importances, features = calculate_auc_importance(X_processed, y)
auc_importance_df = create_feature_importance_df(auc_importances, features, 'AUC')
plot_feature_importance(auc_importance_df, 'AUC')

#######################################
# 2. Mutual Information
#######################################
print("\n2. Обчислення значущості ознак за методом Mutual Information...")

mi_importances = mutual_info_classif(X_processed, y, random_state=42)
mi_importance_df = create_feature_importance_df(mi_importances, X_processed.columns, 'MutualInformation')
plot_feature_importance(mi_importance_df, 'Mutual Information')

#######################################
# 3. Distance Correlation
#######################################
print("\n3. Обчислення значущості ознак за методом Distance Correlation...")


def calculate_dcor_importance(X, y):
    """
    Обчислює значущість ознак на основі кореляції відстаней
    """
    importances = []
    feature_names = X.columns

    y_array = y.values.reshape(-1, 1)  # Перетворення y в 2D масив для dcor

    for feature in feature_names:
        x_array = X[feature].values.reshape(-1, 1)
        # Обчислення кореляції відстаней
        dc = dcor.distance_correlation(x_array, y_array)
        importances.append(dc)

    return importances, feature_names


dcor_importances, features = calculate_dcor_importance(X_processed, y)
dcor_importance_df = create_feature_importance_df(dcor_importances, features, 'DistanceCorrelation')
plot_feature_importance(dcor_importance_df, 'Distance Correlation')

#######################################
# 4. Логістична регресія
#######################################
print("\n4. Обчислення значущості ознак за допомогою Логістичної регресії...")

# Масштабування даних для логістичної регресії
scaler = StandardScaler()
X_processed_scaled = scaler.fit_transform(X_processed)

# Навчання логістичної регресії
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
log_reg.fit(X_processed_scaled, y)

# Отримання абсолютних значень коефіцієнтів як міри важливості
log_reg_importances = np.abs(log_reg.coef_[0])

# Створення DataFrame з результатами
log_reg_importance_df = create_feature_importance_df(log_reg_importances, X_processed.columns, 'LogisticRegression')
plot_feature_importance(log_reg_importance_df, 'Логістична регресія')


# Додатково: створення таблиці з коефіцієнтами, odds ratio та p-values
def calculate_logistic_regression_stats(X_scaled, y, feature_names):
    """
    Обчислює детальну статистику для коефіцієнтів логістичної регресії
    використовуючи statsmodels для отримання p-значень
    """
    import statsmodels.api as sm

    # Додаємо константу для перехоплення
    X_with_const = sm.add_constant(X_scaled)

    # Навчання моделі за допомогою statsmodels
    logit_model = sm.Logit(y, X_with_const)
    try:
        result = logit_model.fit(disp=0)  # disp=0 відключає виведення інформації про збіжність

        # Отримання коефіцієнтів, p-значень і довірчих інтервалів
        coef = result.params[1:]  # Пропускаємо константу
        p_values = result.pvalues[1:]  # Пропускаємо константу
        odds_ratio = np.exp(coef)

        # Створення результуючого DataFrame
        result_df = pd.DataFrame({
            'Ознака': feature_names,
            'Коефіцієнт': coef,
            'Odds_Ratio': odds_ratio,
            'p_value': p_values
        })

    except Exception as e:
        print(f"Помилка при навчанні logit моделі: {e}")
        print("Створення спрощеної таблиці без p-значень...")

        # Використовуємо sklearn LogisticRegression як запасний варіант
        log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
        log_reg.fit(X_scaled, y)

        coef = log_reg.coef_[0]
        odds_ratio = np.exp(coef)

        # Створення результуючого DataFrame без p-значень
        result_df = pd.DataFrame({
            'Ознака': feature_names,
            'Коефіцієнт': coef,
            'Odds_Ratio': odds_ratio
        })

    # Сортування за абсолютним значенням коефіцієнтів
    result_df = result_df.sort_values('Коефіцієнт', key=abs, ascending=False)

    # Збереження результатів
    output_file = 'logistic_regression_detailed.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Детальні результати логістичної регресії збережено у файл {output_file}")

    return result_df


# Обчислення детальної статистики для логістичної регресії
log_reg_detailed = calculate_logistic_regression_stats(X_processed_scaled, y, X_processed.columns)

#######################################
# 5. Decision Tree
#######################################
print("\n5. Обчислення значущості ознак за допомогою Decision Tree...")

# Навчання дерева рішень
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_processed, y)

# Отримання важливостей ознак
dt_importances = dt.feature_importances_

# Створення DataFrame з результатами
dt_importance_df = create_feature_importance_df(dt_importances, X_processed.columns, 'DecisionTree')
plot_feature_importance(dt_importance_df, 'Decision Tree')

#######################################
# Порівняння результатів різних методів
#######################################
print("\nСтворення зведеної таблиці результатів...")


# Функція для отримання рангу ознаки
def get_feature_rank(feature, importance_df):
    """
    Повертає ранг ознаки в DataFrame з важливостями
    """
    # Додаємо колонку з рангами після сортування
    df_with_ranks = importance_df.reset_index(drop=True)
    df_with_ranks['Ранг'] = df_with_ranks.index + 1

    # Отримуємо ранг для заданої ознаки
    return int(df_with_ranks[df_with_ranks['Ознака'] == feature]['Ранг'].values[0])


# Створюємо зведену таблицю з рангами для кожного методу
def create_summary_table(feature_names, importance_dfs, method_names):
    """
    Створює зведену таблицю з рангами ознак за різними методами
    """
    summary_data = []

    for feature in feature_names:
        feature_data = {'Ознака': feature}

        # Додаємо ранги та значення для кожного методу
        for i, method in enumerate(method_names):
            df = importance_dfs[i]
            rank = get_feature_rank(feature, df)
            value = float(df[df['Ознака'] == feature]['Значущість'])

            feature_data[f'{method}_ранг'] = rank
            feature_data[f'{method}_значення'] = value

        # Обчислюємо середній ранг
        rank_columns = [feature_data[f'{method}_ранг'] for method in method_names]
        feature_data['Середній_ранг'] = np.mean(rank_columns)

        summary_data.append(feature_data)

    # Створюємо DataFrame та сортуємо за середнім рангом
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Середній_ранг')

    # Збереження зведеної таблиці
    output_file = 'feature_importance_summary.csv'
    summary_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Зведену таблицю збережено у файл {output_file}")

    return summary_df


# Створення зведеної таблиці
importance_dfs = [auc_importance_df, mi_importance_df, dcor_importance_df,
                  log_reg_importance_df, dt_importance_df]
method_names = ['AUC', 'MI', 'dCor', 'LogReg', 'DecTree']
summary_df = create_summary_table(X_processed.columns, importance_dfs, method_names)

# Візуалізація всіх ознак за середнім рангом
plt.figure(figsize=(14, max(10, len(summary_df) * 0.3)))  # Динамічно налаштовуємо висоту
sns.barplot(x='Середній_ранг', y='Ознака', data=summary_df)
plt.title('Середній ранг значущості ознак за всіма методами')
plt.tight_layout()

# Збереження візуалізації середнього рангу
output_file = 'feature_importance_average_rank.png'
plt.savefig(output_file, dpi=300)
plt.close()
print(f"Візуалізацію середнього рангу збережено у файл {output_file}")

# Створення стовпчастої діаграми для порівняння всіх ознак за різними методами
comparison_data = []
for feature in X_processed.columns:
    for i, method in enumerate(method_names):
        df = importance_dfs[i]
        value = float(df[df['Ознака'] == feature]['Значущість'])
        
        # Масштабуємо значення, щоб вони були порівнянні
        if method == 'AUC':
            # AUC значення між 0.5 і 1, масштабуємо до [0, 1]
            scaled_value = (value - 0.5) * 2 if value > 0.5 else 0
        else:
            # Нормалізуємо відносно максимального значення для методу
            max_value = df['Значущість'].max()
            scaled_value = value / max_value if max_value > 0 else 0

        comparison_data.append({
            'Ознака': feature,
            'Метод': method,
            'Відносна_значущість': scaled_value
        })

comparison_df = pd.DataFrame(comparison_data)

# Візуалізуємо топ-15 ознак для кращої читабельності
top_features = summary_df.head(15)['Ознака'].tolist()
top_comparison_df = comparison_df[comparison_df['Ознака'].isin(top_features)]

# Візуалізація порівняння методів для топ-15 ознак
plt.figure(figsize=(16, 10))
sns.barplot(x='Ознака', y='Відносна_значущість', hue='Метод', data=top_comparison_df)
plt.title('Порівняння відносної значущості топ-15 ознак за різними методами')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Збереження порівняльної діаграми для топ-15
output_file = 'feature_importance_comparison_top15.png'
plt.savefig(output_file, dpi=300)
plt.close()
print(f"Порівняльну діаграму для топ-15 ознак збережено у файл {output_file}")

# Створюємо інтерактивну HTML-таблицю з результатами
try:
    html_table = summary_df.to_html(index=False)
    with open('feature_importance_summary.html', 'w', encoding='utf-8') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Результати аналізу значущості ознак</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tr:hover {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Результати аналізу значущості ознак</h1>
            <p>Таблиця містить ранги та значення значущості для різних методів аналізу.</p>
            {html_table}
        </body>
        </html>
        """)
    print("Створено HTML-таблицю з результатами: feature_importance_summary.html")
except Exception as e:
    print(f"Помилка при створенні HTML-таблиці: {e}")

print("\nАналіз значущості ознак завершено. Результати збережено у відповідні файли.")
