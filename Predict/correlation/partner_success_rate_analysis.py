import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# Створюємо директорію для збереження графіків
results_dir = f"partner_success_rate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(results_dir, exist_ok=True)

# Встановлюємо українську локаль для графіків
import locale
try:
    locale.setlocale(locale.LC_ALL, 'uk_UA.UTF-8')
except:
    print("Українська локаль не знайдена, використовуємо стандартну")

# Завантаження даних
print("Завантажуємо дані з CSV-файлу...")
df = pd.read_csv("cleaned_result.csv")
print(f"Завантажено {df.shape[0]} рядків та {df.shape[1]} стовпців")
print(f"Розподіл класів: {df['is_successful'].value_counts().to_dict()}")

# Перевіряємо наявність пропущених значень
missing_values = df['partner_success_rate'].isnull().sum()
if missing_values > 0:
    print(f"Виявлено {missing_values} пропущених значень в partner_success_rate, заповнюємо медіаною")
    df['partner_success_rate'] = df['partner_success_rate'].fillna(df['partner_success_rate'].median())

# Налаштовуємо загальний стиль графіків
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# 3. Гістограма з нормалізацією
print("Створюємо вдосконалену гістограму з нормалізацією...")

# Аналізуємо розподіл значень partner_success_rate
unique_counts = df['partner_success_rate'].value_counts().sort_index()
print(f"Кількість унікальних значень partner_success_rate: {len(unique_counts)}")
print(f"Мінімальне значення: {df['partner_success_rate'].min()}, Максимальне значення: {df['partner_success_rate'].max()}")

# Створюємо більш детальні категорії на основі квантилів
num_bins = 10  # Використовуємо 10 рівних інтервалів для більш детального аналізу
bins = np.linspace(0, df['partner_success_rate'].max(), num_bins+1)
labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]

# Створюємо категорії
df['success_rate_category'] = pd.cut(df['partner_success_rate'], bins=bins, labels=labels, include_lowest=True)

# Обчислюємо статистику для кожної категорії
category_stats = df.groupby('success_rate_category').agg(
    total_count=('is_successful', 'count'),
    success_count=('is_successful', 'sum'),
    success_rate=('is_successful', lambda x: x.sum() / len(x) * 100)
).reset_index()

# Додаємо відсоток від загальної кількості
category_stats['percentage_of_total'] = category_stats['total_count'] / len(df) * 100

# Сортуємо за категоріями для правильного відображення
category_stats = category_stats.sort_values('success_rate_category')

print("Статистика за категоріями:")
print(category_stats)

# Додатковий аналіз: Створення гістограми розподілу значень partner_success_rate
plt.figure(figsize=(12, 6))
plt.hist(df['partner_success_rate'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Розподіл значень partner_success_rate')
plt.xlabel('Значення partner_success_rate')
plt.ylabel('Кількість замовлень')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{results_dir}/3b_distribution_of_partner_success_rate.png", dpi=300)
plt.close()

print("Аналіз завершено. Результати збережено в директорію:", results_dir)
