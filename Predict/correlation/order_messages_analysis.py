import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# Створюємо директорію для збереження графіків
results_dir = f"order_messages_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
missing_values = df['order_messages'].isnull().sum()
if missing_values > 0:
    print(f"Виявлено {missing_values} пропущених значень в order_messages, заповнюємо нулями")
    df['order_messages'] = df['order_messages'].fillna(0)

# Налаштовуємо загальний стиль графіків
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. Віоліновий графік замість коробкового
print("Створюємо віоліновий графік...")
plt.figure(figsize=(10, 6))

# Створюємо віоліновий графік
ax = sns.violinplot(x='is_successful', y='order_messages', data=df,
                    inner='box', cut=0, density_norm='width')

# Налаштування графіка
y_max = min(df['order_messages'].quantile(0.95), 25)  # Обмежуємо до 95-го перцентиля або максимум 25
ax.set_ylim(0, y_max)
ax.set_title('Розподіл кількості повідомлень за успішністю замовлення\n(віоліновий графік)')
ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
ax.set_ylabel('Кількість повідомлень')

# Додаємо медіану та середнє значення як текст
for i, success in enumerate([0, 1]):
    subset = df[df['is_successful'] == success]['order_messages']
    median = subset.median()
    mean = subset.mean()
    plt.text(i, median - 2, f'Медіана: {median:.1f}', ha='center')
    plt.text(i, mean + 2, f'Середнє: {mean:.1f}', ha='center')

# Додаємо інформацію про викиди
plt.text(0.5, y_max*0.9, f"Макс. кількість повідомлень: {df['order_messages'].max():.0f}",
         ha='center', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{results_dir}/1_violin_plot_order_messages.png", dpi=300)
plt.close()

# 2. Щільність розподілу (Density Plot)
print("Створюємо графік щільності розподілу...")

# Визначаємо межу для 95-го перцентиля
p95 = df['order_messages'].quantile(0.95)
max_val = df['order_messages'].max()

# Створюємо один графік
plt.figure(figsize=(12, 8))
ax = plt.gca()

# Малюємо графік щільності для кожної категорії
for i, (success, color, label) in enumerate([(0, 'forestgreen', 'Невдалі замовлення'),
                                             (1, 'crimson', 'Успішні замовлення')]):
    subset = df[df['is_successful'] == success]['order_messages']
    mean_val = subset.mean()
    median_val = subset.median()

    # Малюємо графік щільності
    sns.kdeplot(data=subset, ax=ax, color=color, fill=True, alpha=0.5, label=f"{label}")

    # Додаємо вертикальні лінії
    plt.axvline(x=mean_val, color=color, linestyle='--', alpha=0.7)
    plt.axvline(x=median_val, color=color, linestyle=':', alpha=0.7)

    # Додаємо текст у верхній правий кут
    plt.text(0.98, 0.85 - i*0.1,
             f"{label}:\nСереднє: {mean_val:.1f}\nМедіана: {median_val:.1f}",
             transform=ax.transAxes,  # Координати відносно графіка (0-1)
             color=color, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Додаємо інформацію про максимальне значення
plt.text(0.98, 0.65, f"Макс. кількість повідомлень: {max_val:.0f}",
         transform=ax.transAxes, ha='right', va='top',
         bbox=dict(facecolor='white', alpha=0.8))

# Налаштування графіка
plt.xlim(0, p95*1.1)  # Показуємо до 95-го перцентиля + 10%
plt.ylim(0, plt.ylim()[1] * 1.1)  # Додаємо 10% простору зверху
plt.title('Щільність розподілу кількості повідомлень за успішністю замовлення\n(до 95-го перцентиля)')
plt.xlabel('Кількість повідомлень')
plt.ylabel('Щільність')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f"{results_dir}/2_density_order_messages_by_success.png", dpi=300)
plt.close()

# 3. Гістограма з нормалізацією
print("Створюємо гістограму з нормалізацією...")

# Визначаємо межі для групування значень order_messages
p95 = df['order_messages'].quantile(0.95)  # 95й перцентиль для кращого масштабування
max_messages = min(df['order_messages'].max(), p95)
bins = np.arange(0, max_messages + 5, 5)  # Групуємо по 5 повідомлень

plt.figure(figsize=(14, 7))

# Створення гістограми
df['message_bins'] = pd.cut(df['order_messages'], bins)
hist = df.groupby(['message_bins', 'is_successful'], observed=True).size().unstack()
hist_norm = hist.div(hist.sum(axis=1), axis=0).fillna(0)

# Побудова графіка
hist_norm.plot(kind='bar', stacked=True, color=['#FF6B6B', '#4ECDC4'])

# Налаштування графіка
plt.title('Відсоток успішних та невдалих замовлень за кількістю повідомлень')
plt.xlabel('Кількість повідомлень')
plt.ylabel('Частка від загальної кількості')
plt.xticks(rotation=45)
plt.legend(['Невдалі замовлення', 'Успішні замовлення'])

# Додаємо підписи відсотків для успішних замовлень
for i, p in enumerate(plt.gca().patches[len(bins)-1:]):  # Тільки для успішних замовлень
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    if height > 0.05:  # Показуємо підпис тільки якщо стовпчик достатньо великий
        plt.text(x+width/2, y+height/2, f'{height:.1%}', ha='center', color='white')

plt.tight_layout()
plt.savefig(f"{results_dir}/3_histogram_normalized_success_by_messages.png", dpi=300)
plt.close()

# 4. Логістична крива ймовірності
print("Створюємо логістичну криву ймовірності...")

# Визначаємо межу для 95-го перцентиля
p95 = df['order_messages'].quantile(0.95)

# Підготовка даних
X = df[['order_messages']].values
y = df['is_successful'].values

# Масштабування для кращих результатів
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Навчання логістичної регресії
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Створення даних для кривої
X_range = np.linspace(0, p95, 100).reshape(-1, 1)  # Обмежуємо до 95-го перцентиля
X_range_scaled = scaler.transform(X_range)
y_proba = model.predict_proba(X_range_scaled)[:, 1]

# Побудова графіку
plt.figure(figsize=(10, 6))

# Додаємо розсіяні точки з альфа прозорістю (вибірка тільки в межах 95-го перцентиля)
X_filtered = X[X <= p95]
y_filtered = y[X.flatten() <= p95]
sample_size = min(5000, len(X_filtered))

if len(X_filtered) > 0:  # Перевіряємо, що є дані після фільтрації
    sampled_indices = np.random.choice(len(X_filtered), size=min(sample_size, len(X_filtered)), replace=False)
    plt.scatter(X_filtered[sampled_indices], y_filtered[sampled_indices],
                alpha=0.3, c=y_filtered[sampled_indices], cmap='coolwarm', edgecolors='none')

# Додаємо криву логістичної регресії
plt.plot(X_range, y_proba, color='blue', linewidth=3)
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Поріг класифікації (0.5)')

# Знаходимо точку перетину кривої з порогом 0.5
threshold_idx = np.abs(y_proba - 0.5).argmin()
threshold_x = X_range[threshold_idx][0]
plt.axvline(x=threshold_x, color='green', linestyle='--', alpha=0.7)
plt.text(threshold_x + 1, 0.3, f'Точка перетину: {threshold_x:.1f} повідомлень', color='green')

# Налаштування графіка
plt.title('Логістична регресія: ймовірність успішності замовлення за кількістю повідомлень\n(до 95-го перцентиля)')
plt.xlabel('Кількість повідомлень')
plt.ylabel('Ймовірність успішного замовлення')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, p95)  # Повертаємо як було (до 95-го перцентиля)
plt.ylim(-0.05, 1.05)  # Додаємо відступ від нуля знизу та зверху

# Додаємо інформацію про коефіцієнти моделі
coef = model.coef_[0][0]
intercept = model.intercept_[0]
plt.text(0.02, 0.12, f'Коефіцієнт моделі: {coef:.4f}\nЗміщення: {intercept:.4f}',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

# Додаємо інформацію про 95-й перцентиль
plt.text(0.98, 0.12, f"95-й перцентиль: {p95:.1f} повідомлень",
         transform=plt.gca().transAxes, ha='right',
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{results_dir}/4_logistic_regression_curve.png", dpi=300)
plt.close()

print(f"Всі графіки успішно збережено в директорію: {results_dir}")
