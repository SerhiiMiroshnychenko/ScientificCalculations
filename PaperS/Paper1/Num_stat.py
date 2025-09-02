import pandas as pd

data = pd.read_csv('data_scaled_replaced.csv')


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

pd.set_option('display.max_rows', None)  # показувати всі рядки
pd.set_option('display.max_columns', None)  # показувати всі колонки
pd.set_option('display.width', None)  # не обмежувати ширину виводу
pd.set_option('display.max_colwidth', None)  # повна ширина для кожної колонки

# 5. Виводимо результат
print(stats_df)


