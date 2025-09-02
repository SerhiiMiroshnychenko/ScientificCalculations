import pandas as pd
from sklearn.preprocessing import RobustScaler

data = pd.read_csv('cleanest_data.csv')

# 1. Виділяємо тільки числові колонки
numeric_columns = data.select_dtypes(include=['number']).columns.tolist()

# 2. Ініціалізуємо RobustScaler
scaler = RobustScaler()

# 3. Масштабуємо тільки числові дані
scaled_array = scaler.fit_transform(data[numeric_columns])

# 4. Перетворюємо назад у DataFrame, додаючи _scaled
scaled_df = pd.DataFrame(scaled_array, columns=[col + '_scaled' for col in numeric_columns])

# 5. Об'єднуємо масштабовані колонки з оригінальними немасштабованими
data_scaled = pd.concat([data.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

# 6. Готово!
print("Масштабування завершено. Фінальний датафрейм має форму:", data_scaled.shape)

# Якщо хочеш зберегти в файл для аналізу
data_scaled.to_csv('data_scaled.csv', index=False)
