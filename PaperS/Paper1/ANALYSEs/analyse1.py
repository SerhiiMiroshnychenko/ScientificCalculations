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
def plot_feature_importance(importance_df, method_name, top_n=20):
    """
    Створює візуалізацію топ-N найважливіших ознак
    """
    plt.figure(figsize=(12, 8))

    # Вибираємо топ-N ознак
    plot_df = importance_df.head(top_n)

    # Створюємо горизонтальну гістограму
    sns.barplot(x='Значущість', y='Ознака', data=plot_df)

    plt.title(f'Топ-{top_n} найважливіших ознак за методом {method_name}')
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


auc_importances, features = calculate_auc_importance(X, y)
auc_importance_df = create_feature_importance_df(auc_importances, features, 'AUC')
plot_feature_importance(auc_importance_df, 'AUC')

#######################################
# 2. Mutual Information
#######################################
print("\n2. Обчислення значущості ознак за методом Mutual Information...")

mi_importances = mutual_info_classif(X, y, random_state=42)
mi_importance_df = create_feature_importance_df(mi_importances, X.columns, 'MutualInformation')
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


dcor_importances, features = calculate_dcor_importance(X, y)
dcor_importance_df = create_feature_importance_df(dcor_importances, features, 'DistanceCorrelation')
plot_feature_importance(dcor_importance_df, 'Distance Correlation')

#######################################
# 4. Логістична регресія
#######################################
print("\n4. Обчислення значущості ознак за допомогою Логістичної регресії...")

# Масштабування даних для логістичної регресії
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Навчання логістичної регресії
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
log_reg.fit(X_scaled, y)

# Отримання абсолютних значень коефіцієнтів як міри важливості
log_reg_importances = np.abs(log_reg.coef_[0])

# Створення DataFrame з результатами
log_reg_importance_df = create_feature_importance_df(log_reg_importances, X.columns, 'LogisticRegression')
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
log_reg_detailed = calculate_logistic_regression_stats(X_scaled, y, X.columns)

#######################################
# 5. Decision Tree
#######################################
print("\n5. Обчислення значущості ознак за допомогою Decision Tree...")

# Навчання дерева рішень
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X, y)

# Отримання важливостей ознак
dt_importances = dt.feature_importances_

# Створення DataFrame з результатами
dt_importance_df = create_feature_importance_df(dt_importances, X.columns, 'DecisionTree')
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
summary_df = create_summary_table(X.columns, importance_dfs, method_names)

# Візуалізація топ-10 ознак за середнім рангом
top_features = summary_df.head(10)['Ознака'].tolist()

plt.figure(figsize=(15, 10))

# Створення стовпчастої діаграми для порівняння топ-10 ознак за різними методами
comparison_data = []
for feature in top_features:
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

# Візуалізація порівняння методів
plt.figure(figsize=(16, 10))
sns.barplot(x='Ознака', y='Відносна_значущість', hue='Метод', data=comparison_df)
plt.title('Порівняння відносної значущості топ-10 ознак за різними методами')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Збереження порівняльної діаграми
output_file = 'feature_importance_comparison.png'
plt.savefig(output_file, dpi=300)
plt.close()
print(f"Порівняльну діаграму збережено у файл {output_file}")

print("\nАналіз значущості ознак завершено. Результати збережено у відповідні файли.")