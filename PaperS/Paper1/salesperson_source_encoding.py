import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('cleanest_data.csv')

# Визначаємо цільову змінну та категоріальні стовпці
y = data['is_successful']
salesperson_col = 'salesperson'
source_col = 'source'

# Копіюємо дані, щоб не змінювати оригінал
encoded_data = data.copy()

# Ініціалізуємо LabelEncoder для кожного стовпця
le_salesperson = LabelEncoder()
le_source = LabelEncoder()

# Кодуємо категоріальні змінні
encoded_data[salesperson_col] = le_salesperson.fit_transform(data[salesperson_col])
encoded_data[source_col] = le_source.fit_transform(data[source_col])

# Зберігаємо закодовані дані (опціонально)
encoded_data.to_csv('encoded_data.csv', index=False)

# Виводимо перші рядки для перевірки
print("Label Encoded Data:")
print(encoded_data[[salesperson_col, source_col]].head())