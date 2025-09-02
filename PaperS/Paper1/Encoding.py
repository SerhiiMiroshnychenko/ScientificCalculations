import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


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


# Зчитуємо дані
data = pd.read_csv('cleanest_data.csv')

# Циклічне кодування для дня тижня та місяця
data = cyclical_encoding(data, 'day_of_week', 7)
data = cyclical_encoding(data, 'month', 12)

# Target Encoding для salesperson та source
data['salesperson_encoded'] = target_encode_cv(data, 'salesperson', 'is_successful')
data['source_encoded'] = target_encode_cv(data, 'source', 'is_successful')

# Зберігаємо закодовані дані
data.to_csv('encoded_data.csv', index=False)

print("Кодування завершено успішно.")
print(f"Форма закодованих даних: {data.shape}")
print("\nРозподіл цільової змінної:")
print(data['is_successful'].value_counts(normalize=True))
print("\nПеревірка нових закодованих колонок:")
for col in ['day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
            'salesperson_encoded', 'source_encoded']:
    if col in data.columns:
        print(f"{col}: min={data[col].min()}, max={data[col].max()}, mean={data[col].mean()}")