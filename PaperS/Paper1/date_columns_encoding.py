import pandas as pd
from feature_engine.creation import CyclicalFeatures

data = pd.read_csv('cleanest_data.csv')

print(data.columns)

# Дослідження колонки 'day_of_week'
print("\nРозподіл за днями тижня:")
print(data['day_of_week'].value_counts().sort_index())

# Дослідження колонки 'month'
print("\nРозподіл за місяцями:")
print(data['month'].value_counts().sort_index())

# Дослідження колонки 'quarter'
print("\nРозподіл за кварталами:")
print(data['quarter'].value_counts().sort_index())

# Дослідження колонки 'hour_of_day'
print("\nРозподіл за годинами:")
print(data['hour_of_day'].value_counts().sort_index())

for col in ['day_of_week', 'month']:
    print(f"\nРозподіл для {col}:")
    print(pd.crosstab(data[col], data['is_successful'], normalize='index'))

# Створення мапінгів для днів тижня та місяців
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Застосування мапінгів до відповідних колонок
data['day_of_week'] = data['day_of_week'].map(day_mapping)
data['month'] = data['month'].map(month_mapping)

# Ініціалізуй енкодер для циклічних ознак
cyclical_encoder = CyclicalFeatures(variables=['day_of_week', 'month', 'quarter', 'hour_of_day'],
                                     drop_original=True)

# Навчай та трансформуй дані
data_encoded = cyclical_encoder.fit_transform(data)

# Виведи перші кілька рядків оновлених даних
pd.set_option('display.max_columns', None)
print(data_encoded.head())