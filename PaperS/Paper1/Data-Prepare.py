import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

pd.set_option('display.max_columns', None)  # показувати всі колонки
pd.set_option('display.width', None)  # не обмежувати ширину виводу
pd.set_option('display.max_colwidth', None)  # повна ширина для кожної колонки


# Оновлена функція для циклічного кодування
def cyclical_encoding(data, column, max_val):
    """
    Створює циклічне кодування для порядкових циклічних змінних.
    """
    # Визначаємо коди категорій використовуючи рекомендований підхід
    if isinstance(data[column].dtype, pd.CategoricalDtype):
        codes = data[column].cat.codes
    else:
        codes = pd.Categorical(data[column]).codes

    # Обчислюємо синус і косинус
    sin_vals = np.sin(2 * np.pi * codes / max_val)
    cos_vals = np.cos(2 * np.pi * codes / max_val)

    # Створюємо нові стовпці
    result_df = data.copy()
    result_df[f'{column}_sin'] = sin_vals
    result_df[f'{column}_cos'] = cos_vals

    return result_df


# Функція для Target Encoding з крос-валідацією (без змін)
def target_encode_cv(data, column, target, n_folds=5):
    """
    Виконує Target Encoding з крос-валідацією для уникнення витоку даних.
    """
    result = pd.Series(index=data.index, dtype='float64')
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Загальне середнє значення для обробки нових категорій
    global_mean = data[target].mean()

    for train_idx, test_idx in kf.split(data):
        # Обчислюємо цільові середні на тренувальному наборі
        target_means = data.iloc[train_idx].groupby(column)[target].mean()

        # Застосовуємо їх до тестового набору
        temp_map = data.iloc[test_idx][column].map(target_means)
        result.iloc[test_idx] = temp_map.fillna(global_mean)

    return result


# Завантаження даних
data = pd.read_csv('cleanest_data.csv')
print('-' * 50)
print(data.head())
print(data.info())

# # Циклічне кодування для дня тижня та місяця
# data = cyclical_encoding(data, 'day_of_week', 7)
# data = cyclical_encoding(data, 'month', 12)
#
# # Target Encoding для salesperson та source
# data['salesperson_encoded'] = target_encode_cv(data, 'salesperson', 'is_successful')
# data['source_encoded'] = target_encode_cv(data, 'source', 'is_successful')
#
# print("Кодування завершено успішно.")
# print(f"Форма закодованих даних: {data.shape}")
# print("\nРозподіл цільової змінної:")
# print(data['is_successful'].value_counts(normalize=True))
# print("\nПеревірка нових закодованих колонок:")
# for col in ['day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
#             'salesperson_encoded', 'source_encoded']:
#     if col in data.columns:
#         print(f"{col}: min={data[col].min()}, max={data[col].max()}, mean={data[col].mean()}")

# ДОСЛІДЖЕННЯ ЧИСЛОВИХ СТАТИСТИК
# 1. Виділяємо тільки числові колонки
numeric_columns = data.select_dtypes(include=['number']).columns

# 3. Заміна від'ємних значень на 0 у числових стовпцях
for col in numeric_columns:
    data[col] = data[col].apply(lambda x: max(0, x))  # Застосовуємо функцію max(0, x) до кожного елемента стовпця

# 1. Виключення даних за 2025 рік
data['create_date'] = pd.to_datetime(data['create_date'])  # Перетворення стовпця 'create_date' у формат datetime
data = data[data['create_date'].dt.year < 2025].copy()  # Фільтруємо рядки, залишаючи лише ті, де рік менше 2025

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
print('\n')

# 2. Додавання стовпця 'create_date_months'
min_date = data['create_date'].min()  # Знаходимо найпершу дату
data.loc[:, 'create_date_months'] = ((data['create_date'] - min_date).dt.days / 30).astype(
    int)  # Обчислюємо різницю в місяцях, перетворюємо у int

# Нормалізація partner_success_rate
data['partner_success_rate'] = data['partner_success_rate'] / 100

# Список колонок, які потрібно масштабувати
columns_to_scale = [
    'order_amount',
    'order_messages',
    'order_changes',
    'partner_total_orders',
    'partner_order_age_days',
    'partner_avg_amount',
    'partner_success_avg_amount',
    'partner_fail_avg_amount',
    'partner_total_messages',
    'partner_success_avg_messages',
    'partner_fail_avg_messages',
    'partner_avg_changes',
    'partner_success_avg_changes',
    'partner_fail_avg_changes',
    'order_lines_count',
    'discount_total'
]

# 2. Ініціалізуємо RobustScaler
scaler = RobustScaler()

# 3. Масштабуємо тільки числові дані
scaled_array = scaler.fit_transform(data[columns_to_scale])

# 4. Перетворюємо назад у DataFrame, додаючи _scaled
scaled_df = pd.DataFrame(scaled_array, columns=[col + '_scaled' for col in columns_to_scale])

# 5. Об'єднуємо масштабовані колонки з оригінальними немасштабованими
data_scaled = pd.concat([data.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

# 6. Готово!
print("Масштабування завершено. Фінальний датафрейм має форму:", data_scaled.shape)

# Виводимо перші кілька рядків оброблених даних для перевірки
print('-' * 50)
print(data_scaled.head())
print(data_scaled.info())
data = data_scaled

# Циклічне кодування для дня тижня та місяця
data = cyclical_encoding(data, 'day_of_week', 7)
data = cyclical_encoding(data, 'month', 12)

# Target Encoding для salesperson та source
data['salesperson_encoded'] = target_encode_cv(data, 'salesperson', 'is_successful')
data['source_encoded'] = target_encode_cv(data, 'source', 'is_successful')

print("Кодування завершено успішно.")
print(f"Форма закодованих даних: {data.shape}")
print("\nРозподіл цільової змінної:")
print(data['is_successful'].value_counts(normalize=True))
print("\nПеревірка нових закодованих колонок:")
for col in ['day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
            'salesperson_encoded', 'source_encoded']:
    if col in data.columns:
        print(f"{col}: min={data[col].min()}, max={data[col].max()}, mean={data[col].mean()}")

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
print('\n')
