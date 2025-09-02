import pandas as pd
from scipy.stats import normaltest, anderson
import numpy as np
from sklearn.preprocessing import RobustScaler


data = pd.read_csv('cleanest_data.csv')
print(data)
print(data.head(10))
print(data.tail(10))
print(data.columns)
print(data.info())
print(data.describe())
print(data.nunique())
print(data['is_successful'].value_counts(normalize=True))
print(data.skew(numeric_only=True))
print(data.isnull().sum())

# ДОСЛІДЖЕННЯ РОЗПОДІЛУ
# Вибираємо тільки числові колонки
numeric_cols = data.select_dtypes(include=[np.number]).columns

# Створюємо результуючу таблицю
results = []

for col in numeric_cols:
    col_data = data[col]

    # Тест Д'Агостіно-Кільмейра
    stat_dagostino, p_dagostino = normaltest(col_data)

    # Тест Андерсона-Дарлінга
    result_anderson = anderson(col_data)
    stat_anderson = result_anderson.statistic
    critical_anderson = result_anderson.critical_values[2]  # поріг для рівня значущості 5%

    # Оцінка нормальності
    is_normal_dagostino = p_dagostino > 0.05
    is_normal_anderson = stat_anderson < critical_anderson

    results.append({
        'column': col,
        'dagostino_stat': stat_dagostino,
        'dagostino_p': p_dagostino,
        'dagostino_is_normal': is_normal_dagostino,
        'anderson_stat': stat_anderson,
        'anderson_critical_5%': critical_anderson,
        'anderson_is_normal': is_normal_anderson
    })

# Перетворюємо в DataFrame для красивого вигляду
normality_results = pd.DataFrame(results)
#
pd.set_option('display.max_rows', None)  # показувати всі рядки
pd.set_option('display.max_columns', None)  # показувати всі колонки
pd.set_option('display.width', None)  # не обмежувати ширину виводу
pd.set_option('display.max_colwidth', None)  # повна ширина для кожної колонки

# Виводимо
print(normality_results.sort_values(by='dagostino_p'))

# ДОСЛІДЖЕННЯ ЧИСЛОВИХ СТАТИСТИК
# 1. Виділяємо тільки числові колонки
numeric_columns = data.select_dtypes(include=['number']).columns

# 2. Створюємо список для зберігання результатів
stats_list = []

# 3. Обходимо всі числові колонки
for col in numeric_columns:
    col_min = data[col].min()
    col_max = data[col].max()
    col_mean = data[col].mean()
    col_median = data[col].median()
    # mode() може повертати кілька значень, беремо перше
    col_mode = data[col].mode().iloc[0] if not data[col].mode().empty else None

    stats_list.append({
        'column': col,
        'min': col_min,
        'max': col_max,
        'mean': col_mean,
        'median': col_median,
        'mode': col_mode
    })

# 4. Створюємо фінальний DataFrame
stats_df = pd.DataFrame(stats_list)

# 5. Виводимо результат
print('\n\nЧислові статистики:')
print(stats_df)

# МАСШТАБУВАННЯ
# Виділяємо тільки числові колонки
numeric_columns = data.select_dtypes(include=['number']).columns.tolist()

# Ініціалізуємо RobustScaler
scaler = RobustScaler()

# Масштабуємо тільки числові дані
scaled_array = scaler.fit_transform(data[numeric_columns])

# Створюємо DataFrame з масштабованими даними, використовуючи оригінальні назви колонок
scaled_df = pd.DataFrame(scaled_array, columns=numeric_columns)

# Замінюємо оригінальні числові колонки на масштабовані
data[numeric_columns] = scaled_df

# ПЕРЕВІРКА ЧИСЛОВИХ СТАТИСТИК
# 1. Виділяємо тільки числові колонки
numeric_columns = data.select_dtypes(include=['number']).columns
#
# 2. Створюємо список для зберігання результатів
stats_list = []

# 3. Обходимо всі числові колонки
for col in numeric_columns:
    col_min = data[col].min()
    col_max = data[col].max()
    col_mean = data[col].mean()
    col_median = data[col].median()
    # mode() може повертати кілька значень, беремо перше
    col_mode = data[col].mode().iloc[0] if not data[col].mode().empty else None

    stats_list.append({
        'column': col,
        'min': col_min,
        'max': col_max,
        'mean': col_mean,
        'median': col_median,
        'mode': col_mode
    })

# 4. Створюємо фінальний DataFrame
stats_df = pd.DataFrame(stats_list)

# 5. Виводимо результат
print('\n\nПеревірка масштабування:')
print(stats_df)

for col in ['day_of_week', 'month', 'salesperson', 'source']:
    print(f"Розподіл для {col}:")
    print(pd.crosstab(data[col], data['is_successful'], normalize='index'))

from scipy.stats import chi2_contingency
for col in ['day_of_week', 'month', 'salesperson', 'source']:
    contingency_table = pd.crosstab(data[col], data['is_successful'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"{col}: chi2 = {chi2}, p-value = {p}")

for col in numeric_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)][col]
    print(f"{col}: кількість викидів = {len(outliers)}")