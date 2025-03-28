#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
from datetime import datetime
from matplotlib import scale as mscale
from matplotlib.transforms import Transform
import matplotlib.transforms as mtransforms

# Створюємо директорію для збереження графіків
results_dir = f"order_amount_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
missing_values = df['order_amount'].isnull().sum()
if missing_values > 0:
    print(f"Виявлено {missing_values} пропущених значень в order_amount, заповнюємо медіаною")
    df['order_amount'] = df['order_amount'].fillna(df['order_amount'].median())

# Налаштовуємо загальний стиль графіків
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. Віоліновий графік замість коробкового
print("Створюємо віоліновий графік...")
plt.figure(figsize=(10, 6))

# Визначаємо межу для 95-го перцентиля
p95 = df['order_amount'].quantile(0.95)

# Фільтруємо дані за 95-м перцентилем для кращої візуалізації
df_filtered = df[df['order_amount'] <= p95]

# Створюємо віоліновий графік
ax = sns.violinplot(x='is_successful', y='order_amount', data=df_filtered,
                    inner='box', cut=0, density_norm='width')

# Налаштування графіка
ax.set_title('Розподіл суми замовлення за успішністю замовлення\n(віоліновий графік)')
ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
ax.set_ylabel('Сума замовлення (USD)')

# Додаємо медіану та середнє значення як текст
for i, success in enumerate([0, 1]):
    subset = df[df['is_successful'] == success]['order_amount']
    median = subset.median()
    mean = subset.mean()
    plt.text(i, median - p95*0.1, f'Медіана: {median:.1f}', ha='center')
    plt.text(i, median + p95*0.1, f'Середнє: {mean:.1f}', ha='center')

# Додаємо інформацію про викиди
plt.text(0.5, p95*0.9, f"95-й перцентиль: {p95:.0f} USD\nМакс. сума: {df['order_amount'].max():.0f} USD",
         ha='center', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{results_dir}/1_violin_plot_order_amount.png", dpi=300)
plt.close()

# 2. Щільність розподілу (Density Plot)
print("Створюємо графік щільності розподілу...")

# Створюємо один графік
plt.figure(figsize=(12, 8))
ax = plt.gca()

# Малюємо графік щільності для кожної категорії
for i, (success, color, label) in enumerate([(0, 'forestgreen', 'Невдалі замовлення'),
                                             (1, 'crimson', 'Успішні замовлення')]):
    subset = df[df['is_successful'] == success]['order_amount']
    subset_filtered = subset[subset <= p95]  # Фільтруємо для кращої візуалізації
    mean_val = subset.mean()
    median_val = subset.median()

    # Малюємо графік щільності
    sns.kdeplot(data=subset_filtered, ax=ax, color=color, fill=True, alpha=0.5, label=f"{label}")

    # Додаємо вертикальні лінії
    plt.axvline(x=mean_val, color=color, linestyle='--', alpha=0.7)
    plt.axvline(x=median_val, color=color, linestyle=':', alpha=0.7)

    # Додаємо текст у верхній правий кут
    plt.text(0.98, 0.85 - i*0.1,
             f"{label}:\nСереднє: {mean_val:.1f} USD\nМедіана: {median_val:.1f} USD",
             transform=ax.transAxes,  # Координати відносно графіка (0-1)
             color=color, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Додаємо інформацію про максимальне значення
plt.text(0.98, 0.65, f"95-й перцентиль: {p95:.0f} USD\nМакс. сума: {df['order_amount'].max():.0f} USD",
         transform=ax.transAxes, ha='right', va='top',
         bbox=dict(facecolor='white', alpha=0.8))

# Налаштування графіка
plt.xlim(0, p95*1.1)  # Показуємо до 95-го перцентиля + 10%
plt.ylim(0, plt.ylim()[1] * 1.1)  # Додаємо 10% простору зверху
plt.title('Щільність розподілу суми замовлення за успішністю замовлення\n(до 95-го перцентиля)')
plt.xlabel('Сума замовлення (USD)')
plt.ylabel('Щільність')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f"{results_dir}/2_density_order_amount_by_success.png", dpi=300)
plt.close()

# 2.1 Додатковий графік щільності з обмеженням до 8000 USD
print("Створюємо обрізаний графік щільності розподілу (до 8000 USD)...")

# Створюємо один графік
plt.figure(figsize=(12, 8))
ax = plt.gca()

# Малюємо графік щільності для кожної категорії
for i, (success, color, label) in enumerate([(0, 'forestgreen', 'Невдалі замовлення'),
                                             (1, 'crimson', 'Успішні замовлення')]):
    subset = df[df['is_successful'] == success]['order_amount']
    # Обмежуємо до 8000 USD для кращої візуалізації
    subset_filtered = subset[subset <= 8000]
    mean_val = subset.mean()
    median_val = subset.median()

    # Малюємо графік щільності
    sns.kdeplot(data=subset_filtered, ax=ax, color=color, fill=True, alpha=0.5, label=f"{label}")

    # Додаємо вертикальні лінії (лише якщо вони менше 8000)
    if mean_val <= 8000:
        plt.axvline(x=mean_val, color=color, linestyle='--', alpha=0.7)
    if median_val <= 8000:
        plt.axvline(x=median_val, color=color, linestyle=':', alpha=0.7)

    # Додаємо текст у верхній правий кут
    plt.text(0.98, 0.85 - i*0.1,
             f"{label}:\nСереднє: {mean_val:.1f} USD\nМедіана: {median_val:.1f} USD",
             transform=ax.transAxes,  # Координати відносно графіка (0-1)
             color=color, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Налаштування графіка
plt.xlim(0, 8000)  # Обмежуємо до 8000 USD
plt.ylim(0, plt.ylim()[1] * 1.1)  # Додаємо 10% простору зверху
plt.title('Щільність розподілу суми замовлення за успішністю замовлення\n(обмежено до 8000 USD)')
plt.xlabel('Сума замовлення (USD)')
plt.ylabel('Щільність')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f"{results_dir}/2_density_order_amount_by_success_cutoff_8000.png", dpi=300)
plt.close()

# 3. Вірогідність успіху залежно від суми замовлення (бінарне групування)
print("Створюємо графік вірогідності успіху залежно від суми замовлення...")

# Використовуємо біни для групування даних за сумою замовлення
bins = 20
df_filtered = df[df['order_amount'] <= p95].copy()
df_filtered['amount_bin'] = pd.cut(df_filtered['order_amount'], bins=bins)

# Розраховуємо вірогідність успіху та середину біна
success_rates = df_filtered.groupby('amount_bin', observed=False)['is_successful'].mean().reset_index()
bin_centers = df_filtered.groupby('amount_bin', observed=False)['order_amount'].mean().reset_index()
bin_counts = df_filtered.groupby('amount_bin', observed=False).size().reset_index(name='count')

# Об'єднуємо результати
success_rates = pd.merge(success_rates, bin_centers, on='amount_bin')
success_rates = pd.merge(success_rates, bin_counts, on='amount_bin')

# Видаляємо рядки з NaN для безпечності
success_rates = success_rates.dropna(subset=['order_amount', 'is_successful'])

# Розмір графіка
plt.figure(figsize=(14, 8))

# Головний графік - вірогідність успіху
ax1 = plt.gca()
ax1.set_xlabel('Сума замовлення')
ax1.set_ylabel('Вірогідність успіху', color='blue')

# Налаштування розміру точок залежно від кількості спостережень
sizes = 30 * (success_rates['count'] / success_rates['count'].max()) + 20

# Візуалізуємо точки
scatter = ax1.scatter(success_rates['order_amount'], success_rates['is_successful'],
                      s=sizes, alpha=0.7, c='blue', edgecolors='darkblue')

# Додаємо лінійну регресію
if len(success_rates) >= 4:
    z = np.polyfit(success_rates['order_amount'], success_rates['is_successful'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(success_rates['order_amount'].min(), success_rates['order_amount'].max(), 100)
    ax1.plot(x_trend, p(x_trend), 'r--', linewidth=2,
             label=f'Лінійний тренд: y = {z[0]:.6f}x + {z[1]:.2f}')
    ax1.legend(loc='upper left')

# Додаємо гістограму кількості замовлень як другу вісь
ax2 = ax1.twinx()
ax2.set_ylabel('Кількість замовлень', color='gray')
ax2.bar(success_rates['order_amount'], success_rates['count'],
        alpha=0.2, color='gray', width=p95/bins)
ax2.set_ylim(0, success_rates['count'].max() * 1.2)
ax2.tick_params(axis='y', colors='gray')

# Налаштування основного графіку
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)
ax1.set_xlim(0, p95)

plt.title('Вірогідність успіху залежно від суми замовлення')
plt.tight_layout()
plt.savefig(f"{results_dir}/3_success_probability_by_amount.png", dpi=300)
plt.close()

# 4. Логістична регресія та ROC-крива
print("Будуємо логістичну регресію та ROC-криву...")

# Беремо тільки order_amount як незалежну змінну для моделі
X = df[['order_amount']].values
y = df['is_successful'].values

# Розділяємо дані на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Стандартизуємо дані
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Навчаємо логістичну регресію
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Отримуємо прогнози вірогідностей
y_score = model.predict_proba(X_test_scaled)[:, 1]

# Обчислюємо ROC-криву та AUC
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Створюємо візуалізацію ROC-кривої
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC крива (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Випадкова модель')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Специфічність)')
plt.ylabel('True Positive Rate (Чутливість)')
plt.title('ROC-крива для прогнозування успішності замовлення за сумою')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{results_dir}/4_logistic_regression_roc.png", dpi=300)
plt.close()

# 5. Гістограма з нормалізацією - відсоток успішності залежно від суми
print("Створюємо гістограму з нормалізацією...")

# Визначаємо межі для групування значень order_amount
max_amount = min(df['order_amount'].max(), p95)
bin_width = max_amount / 10  # 10 бінів
bins = np.arange(0, max_amount + bin_width, bin_width)

plt.figure(figsize=(14, 7))

# Створення гістограми
df['amount_bins'] = pd.cut(df['order_amount'], bins)
hist = df.groupby(['amount_bins', 'is_successful'], observed=True).size().unstack()
hist_norm = hist.div(hist.sum(axis=1), axis=0).fillna(0)

# Побудова графіка
hist_norm.plot(kind='bar', stacked=True, color=['#FF6B6B', '#4ECDC4'])

# Налаштування графіка
plt.title('Відсоток успішних та невдалих замовлень за сумою замовлення')
plt.xlabel('Сума замовлення (USD)')
plt.ylabel('Частка від загальної кількості')
plt.xticks(rotation=45, fontsize=10)  # Зменшено розмір шрифту для підписів осі X
plt.legend(['Невдалі замовлення', 'Успішні замовлення'])

# Додаємо підписи відсотків для успішних замовлень з меншим розміром шрифту
for i, p in enumerate(plt.gca().patches[len(bins)-1:]):  # Тільки для успішних замовлень
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    if height > 0.05:  # Показуємо підпис тільки якщо стовпчик достатньо великий
        plt.text(x+width/2, y+height/2, f'{height:.1%}', ha='center', fontsize=8, color='white')  # Зменшено розмір шрифту

plt.tight_layout()
plt.savefig(f"{results_dir}/5_histogram_normalized_success_by_amount.png", dpi=300)
plt.close()

# Створюємо власну функцію трансформації для нелінійної шкали
class CustomScaleTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, threshold=1e6, max_value=4.2e6):
        super().__init__()
        self.threshold = threshold  # поріг в 1 мільйон
        self.max_value = max_value  # максимальне значення на графіку (трохи більше 4 млн)

    def transform_non_affine(self, a):
        # Копіюємо масив, щоб не змінювати оригінал
        a = np.array(a, copy=True)

        # Застосовуємо трансформацію:
        # - Значення від 0 до threshold (1 млн) залишаються в діапазоні [0, 0.5]
        # - Значення від threshold до max_value (4.2 млн) масштабуються до діапазону [0.5, 1]
        mask = a <= self.threshold
        a[mask] = 0.5 * a[mask] / self.threshold
        a[~mask] = 0.5 + 0.5 * (a[~mask] - self.threshold) / (self.max_value - self.threshold)

        return a

    def inverted(self):
        return CustomScaleInverseTransform(self.threshold, self.max_value)

class CustomScaleInverseTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, threshold, max_value):
        super().__init__()
        self.threshold = threshold
        self.max_value = max_value

    def transform_non_affine(self, a):
        a = np.array(a, copy=True)

        # Обернена трансформація
        mask = a <= 0.5
        a[mask] = a[mask] * 2 * self.threshold
        a[~mask] = self.threshold + (a[~mask] - 0.5) * 2 * (self.max_value - self.threshold)

        return a

    def inverted(self):
        return CustomScaleTransform(self.threshold, self.max_value)

class CustomScale(mscale.ScaleBase):
    name = 'custom'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.threshold = kwargs.get('threshold', 1e6)
        self.max_value = kwargs.get('max_value', 4.2e6)

    def get_transform(self):
        return CustomScaleTransform(self.threshold, self.max_value)

    def set_default_locators_and_formatters(self, axis):
        # Встановлюємо розташування поділок на осі Y
        axis.set_major_locator(plt.FixedLocator([0, 0.25*1e6, 0.5*1e6, 0.75*1e6, 1e6, 2e6, 3e6, 4e6]))
        axis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}' if x < 1e6 else f'{x/1e6:.0f}'))

# Реєструємо нашу кастомну шкалу
mscale.register_scale(CustomScale)

# 6. Scatter plot з користувацькою шкалою
print("Створюємо scatter plot з користувацькою шкалою...")

# Використаємо альтернативний підхід із SymmetricalLogScale для нерівномірної шкали
plt.figure(figsize=(10, 6))
sns.stripplot(x='is_successful', y='order_amount', data=df, jitter=True, alpha=0.5)
plt.title('Scatter plot з jitter для суми замовлення за успішністю')
plt.xlabel('Успішність')
plt.ylabel('Сума замовлення (млн)')
plt.xticks([0, 1], ['Неуспішні', 'Успішні'])

try:
    # Спробуємо застосувати нашу кастомну шкалу
    plt.yscale('custom', threshold=1e6, max_value=4.2e6)
except Exception as e:
    print(f"Помилка при застосуванні кастомної шкали: {e}")
    print("Використовуємо альтернативний підхід із symlog...")
    # Якщо не вдалося, використовуємо вбудовану symlog шкалу
    plt.yscale('symlog', linthresh=1e6, linscale=0.5)
    plt.ylim(0, 4.2e6)

plt.grid(True, alpha=0.3)

# Додаємо горизонтальні лінії для візуального орієнтиру
plt.axhline(y=1e6, color='red', linestyle='--', alpha=0.5)
plt.text(1.05, 1.1e6, '1 млн (поділ шкали)', color='red', alpha=0.7)

# Зберігаємо графік у файл
current_date = datetime.now().strftime('%Y%m%d')
plt.savefig(f"{results_dir}/scatter_plot_custom_scale_{current_date}.png", dpi=300, bbox_inches='tight')
plt.close()

# Виводимо підсумкову інформацію
print("\nСтатистика для сум замовлень:")
print(f"Успішні замовлення (n={df[df['is_successful'] == 1].shape[0]}):")
print(f"  Середнє: {df[df['is_successful'] == 1]['order_amount'].mean():.2f} USD")
print(f"  Медіана: {df[df['is_successful'] == 1]['order_amount'].median():.2f} USD")
print(f"  Стандартне відхилення: {df[df['is_successful'] == 1]['order_amount'].std():.2f} USD")
print(f"  Мінімум: {df[df['is_successful'] == 1]['order_amount'].min():.2f} USD")
print(f"  Максимум: {df[df['is_successful'] == 1]['order_amount'].max():.2f} USD")

print(f"\nНеуспішні замовлення (n={df[df['is_successful'] == 0].shape[0]}):")
print(f"  Середнє: {df[df['is_successful'] == 0]['order_amount'].mean():.2f} USD")
print(f"  Медіана: {df[df['is_successful'] == 0]['order_amount'].median():.2f} USD")
print(f"  Стандартне відхилення: {df[df['is_successful'] == 0]['order_amount'].std():.2f} USD")
print(f"  Мінімум: {df[df['is_successful'] == 0]['order_amount'].min():.2f} USD")
print(f"  Максимум: {df[df['is_successful'] == 0]['order_amount'].max():.2f} USD")

# Обчислюємо та виводимо коефіцієнти моделі
print(f"\nКоефіцієнти логістичної регресії:")
print(f"Перехват (Intercept): {model.intercept_[0]:.4f}")
print(f"Коефіцієнт для order_amount: {model.coef_[0][0]:.4f}")

# Інтерпретація
if model.coef_[0][0] > 0:
    print("\nІнтерпретація: Збільшення суми замовлення ПІДВИЩУЄ вірогідність успішного замовлення.")
else:
    print("\nІнтерпретація: Збільшення суми замовлення ЗНИЖУЄ вірогідність успішного замовлення.")

# Обчислюємо точність моделі
accuracy = model.score(X_test_scaled, y_test)
print(f"Точність моделі (accuracy): {accuracy:.4f}")

print(f"\nУсі візуалізації збережено в директорії: {results_dir}")
print("Для аналізу впливу суми замовлення на успішність замовлення ви можете переглянути створені графіки.")
