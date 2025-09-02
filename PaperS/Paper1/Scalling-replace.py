import pandas as pd
from sklearn.preprocessing import RobustScaler

# Завантажуємо дані
data = pd.read_csv('cleanest_data.csv')

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

# Готово!
print("Масштабування завершено. Оригінальні числові колонки замінено на масштабовані.")
print("Фінальний датафрейм має форму:", data.shape)

# Зберігаємо оновлений DataFrame в новий файл
data.to_csv('data_scaled_replaced.csv', index=False)