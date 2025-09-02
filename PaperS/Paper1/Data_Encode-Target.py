import pandas as pd
from category_encoders import TargetEncoder

# Завантажуємо дані (переконайтеся, що шлях до файлу правильний)
data = pd.read_csv('cleanest_data.csv')

pd.set_option('display.max_rows', None)

# 1. Унікальні значення
print("Унікальні значення в стовпці 'salesperson':")
print(data['salesperson'].unique())
print("\nУнікальні значення в стовпці 'source':")
print(data['source'].unique())

# 2. Розподіл унікальних значень
print("\nРозподіл значень в стовпці 'salesperson':")
print(data['salesperson'].value_counts())
print("\nРозподіл значень в стовпці 'source':")
print(data['source'].value_counts())

# 3. Розподіл 'salesperson' за 'is_successful'
salesperson_success = data.groupby('salesperson')['is_successful'].value_counts(normalize=True).unstack(fill_value=0)
print("\nРозподіл 'is_successful' за 'salesperson':")
print(salesperson_success)

# 4. Розподіл 'source' за 'is_successful'
source_success = data.groupby('source')['is_successful'].value_counts(normalize=True).unstack(fill_value=0)
print("\nРозподіл 'is_successful' за 'source':")
print(source_success)

# Визначаємо цільову змінну та категоріальні стовпці
y = data['is_successful']
salesperson_col = 'salesperson'
source_col = 'source'

# --- Обробка стовпця 'salesperson' ---
print("Унікальні значення до обробки 'salesperson':", data[salesperson_col].nunique())

# Ініціалізуємо TargetEncoder зі згладжуванням (параметр 'smoothing')
encoder_salesperson = TargetEncoder(cols=[salesperson_col], smoothing=10) # Можна налаштувати значення smoothing

# Застосовуємо кодування
data['salesperson_encoded'] = encoder_salesperson.fit_transform(data[salesperson_col], y)

print("Унікальні значення після обробки 'salesperson':", data['salesperson_encoded'].nunique())
print("\nПерші 5 значень 'salesperson' та їх закодовані значення:")
print(data[[salesperson_col, 'salesperson_encoded']].head())

# --- Обробка стовпця 'source' ---
print("\nУнікальні значення до обробки 'source':", data[source_col].nunique())

# Групуємо рідкісні значення в 'source' перед кодуванням
threshold = 5 # Визначаємо поріг для рідкісних значень
value_counts = data[source_col].value_counts()
rare_values = value_counts[value_counts <= threshold].index
data[source_col] = data[source_col].replace(rare_values, 'Rare_Source')

print("Унікальні значення після групування рідкісних в 'source':", data[source_col].nunique())

# Ініціалізуємо TargetEncoder для 'source'
encoder_source = TargetEncoder(cols=[source_col], smoothing=10) # Можна налаштувати значення smoothing

# Застосовуємо кодування
data['source_encoded'] = encoder_source.fit_transform(data[source_col], y)

print("Унікальні значення після обробки 'source':", data['source_encoded'].nunique())
print("\nПерші 5 значень 'source' та їх закодовані значення:")
print(data[[source_col, 'source_encoded']].head())

# Тепер DataFrame 'data' містить нові закодовані стовпці 'salesperson_encoded' та 'source_encoded'
# які можна використовувати для подальшого навчання моделі.