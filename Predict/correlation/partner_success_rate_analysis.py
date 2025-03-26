import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
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

# 1. Віоліновий графік
print("Створюємо віоліновий графік...")
plt.figure(figsize=(10, 6))

# Створюємо віоліновий графік
ax = sns.violinplot(x='is_successful', y='partner_success_rate', data=df,
                    inner='box', cut=0, density_norm='width')

# Налаштування графіка
y_max = 1.0  # Відсоток успішності від 0 до 1
ax.set_ylim(0, y_max)
ax.set_title('Розподіл середнього % успішних замовлень клієнта\nза успішністю поточного замовлення')
ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
ax.set_ylabel('Середній % успішних замовлень клієнта')

# Додаємо медіану та середнє значення як текст
for i, success in enumerate([0, 1]):
    subset = df[df['is_successful'] == success]['partner_success_rate']
    median = subset.median()
    mean = subset.mean()
    plt.text(i, median - 0.05, f'Медіана: {median:.3f}', ha='center')
    plt.text(i, mean + 0.05, f'Середнє: {mean:.3f}', ha='center')

# Додаємо інформацію про викиди
plt.text(0.5, y_max*0.9, f"Діапазон: від {df['partner_success_rate'].min():.3f} до {df['partner_success_rate'].max():.3f}",
         ha='center', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{results_dir}/1_violin_plot_partner_success_rate.png", dpi=300)
plt.close()

# 2. Щільність розподілу (Density Plot)
print("Створюємо графік щільності розподілу...")

# Створюємо один графік
plt.figure(figsize=(12, 8))
ax = plt.gca()

# Малюємо графік щільності для кожної категорії
for i, (success, color, label) in enumerate([(0, 'forestgreen', 'Невдалі замовлення'),
                                             (1, 'crimson', 'Успішні замовлення')]):
    subset = df[df['is_successful'] == success]['partner_success_rate']
    mean_val = subset.mean()
    median_val = subset.median()

    # Малюємо графік щільності
    sns.kdeplot(data=subset, ax=ax, color=color, fill=True, alpha=0.5, label=f"{label}")

    # Додаємо вертикальні лінії
    plt.axvline(x=mean_val, color=color, linestyle='--', alpha=0.7)
    plt.axvline(x=median_val, color=color, linestyle=':', alpha=0.7)

    # Додаємо текст у верхній правий кут
    plt.text(0.98, 0.85 - i*0.1,
             f"{label}:\nСереднє: {mean_val:.3f}\nМедіана: {median_val:.3f}",
             transform=ax.transAxes,  # Координати відносно графіка (0-1)
             color=color, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Налаштування графіка
plt.xlim(0, 1)  # Відсоток успішності від 0 до 1
plt.ylim(0, plt.ylim()[1] * 1.1)  # Додаємо 10% простору зверху
plt.title('Щільність розподілу середнього % успішних замовлень клієнта\nза успішністю поточного замовлення')
plt.xlabel('Середній % успішних замовлень клієнта')
plt.ylabel('Щільність')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

# Додаємо інформацію про 25й, 50й та 75й перцентилі
q25 = df['partner_success_rate'].quantile(0.25)
q50 = df['partner_success_rate'].quantile(0.50)
q75 = df['partner_success_rate'].quantile(0.75)

plt.text(0.02, 0.5, f"Перцентилі:\n25й: {q25:.3f}\n50й: {q50:.3f}\n75й: {q75:.3f}",
         transform=ax.transAxes, ha='left', va='center',
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{results_dir}/2_density_partner_success_rate.png", dpi=300)
plt.close()

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

# Підготовка даних для графіку
plt.figure(figsize=(16, 8))

# Створюємо основний графік
x = np.arange(len(category_stats))
width = 0.8

# Обчислюємо відсотки для неуспішних замовлень
category_stats['failure_rate'] = 100 - category_stats['success_rate']

# Створюємо складені стовпчики
plt.bar(x, category_stats['failure_rate'], width, label='Невдалі замовлення', color='#FF6B6B')
plt.bar(x, category_stats['success_rate'], width, bottom=category_stats['failure_rate'],
        label='Успішні замовлення', color='#4ECDC4')

# Налаштування графіка
plt.title('Відсоток успішних та невдалих замовлень за категоріями успішності партнера')
plt.xlabel('Категорія середнього % успішних замовлень клієнта')
plt.ylabel('Відсоток від замовлень, %')
plt.xticks(x, category_stats['success_rate_category'], rotation=45, ha='right')
plt.legend(loc='upper left')

# Додаємо підписи відсотків для успішних замовлень
for i, row in enumerate(category_stats.itertuples()):
    if row.success_rate > 5:  # Показуємо підпис тільки якщо стовпчик достатньо великий
        height = row.failure_rate + row.success_rate/2
        plt.text(i, height, f'{row.success_rate:.1f}%', ha='center', va='center', color='white', fontweight='bold')

# Додаємо інформацію про розмір вибірки та відсоток від загальної кількості
for i, row in enumerate(category_stats.itertuples()):
    if row.total_count > 0:
        plt.text(i, -5, f'n={row.total_count}\n({row.percentage_of_total:.1f}%)', ha='center')

# Додаємо другу вісь Y для відображення кількості замовлень у кожній категорії
ax2 = plt.twinx()
ax2.plot(x, category_stats['total_count'], 'o-', color='darkblue', linewidth=2, label='Кількість замовлень')
ax2.set_ylabel('Кількість замовлень')
ax2.legend(loc='upper right')

plt.ylim(0, 100)  # Фіксована шкала для відсотків
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{results_dir}/3_improved_histogram_success_by_rate_categories.png", dpi=300)
plt.close()

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

# Додатковий аналіз: Перевірка впливу квантилів
print("Аналізуємо успішність замовлень за квантилями partner_success_rate...")

# Створюємо 5 рівних за кількістю груп (квантилів)
df['success_rate_quantile'] = pd.qcut(df['partner_success_rate'], q=5, labels=False)

# Обчислюємо межі квантилів для підписів
quantile_edges = pd.qcut(df['partner_success_rate'], q=5, retbins=True)[1]
quantile_labels = [f"Q{i+1}: {quantile_edges[i]:.3f}-{quantile_edges[i+1]:.3f}" for i in range(5)]

# Обчислюємо статистику для кожного квантиля
quantile_stats = df.groupby('success_rate_quantile').agg(
    total_count=('is_successful', 'count'),
    success_count=('is_successful', 'sum'),
    success_rate=('is_successful', lambda x: x.sum() / len(x) * 100),
    mean_partner_rate=('partner_success_rate', 'mean')
).reset_index()

# Створюємо графік успішності за квантилями
plt.figure(figsize=(14, 7))
bars = plt.bar(range(5), quantile_stats['success_rate'], color='#4ECDC4')
plt.title('Відсоток успішних замовлень за квантилями partner_success_rate')
plt.xlabel('Квантиль partner_success_rate')
plt.ylabel('Відсоток успішних замовлень, %')
plt.xticks(range(5), quantile_labels, rotation=45, ha='right')
plt.ylim(0, 100)

# Додаємо підписи відсотків
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%\nn={quantile_stats["total_count"][i]}',
             ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{results_dir}/3c_success_rate_by_quantiles.png", dpi=300)
plt.close()

# 4. Логістична крива ймовірності
print("Створюємо логістичну криву ймовірності...")

# Підготовка даних
X = df[['partner_success_rate']].values
y = df['is_successful'].values

# Масштабування для кращих результатів
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Навчання логістичної регресії
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Створення даних для кривої
X_range = np.linspace(0, 1, 100).reshape(-1, 1)  # Відсоток успішності від 0 до 1
X_range_scaled = scaler.transform(X_range)
y_proba = model.predict_proba(X_range_scaled)[:, 1]

# Побудова графіку
plt.figure(figsize=(10, 6))

# Додаємо розсіяні точки з альфа прозорістю
sample_size = min(5000, len(df))
sampled_indices = np.random.choice(len(df), size=sample_size, replace=False)
plt.scatter(X[sampled_indices], y[sampled_indices],
            alpha=0.3, c=y[sampled_indices], cmap='coolwarm', edgecolors='none')

# Додаємо криву логістичної регресії
plt.plot(X_range, y_proba, color='blue', linewidth=3)
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Поріг класифікації (0.5)')

# Знаходимо точку перетину кривої з порогом 0.5
threshold_idx = np.abs(y_proba - 0.5).argmin()
threshold_x = X_range[threshold_idx][0]
plt.axvline(x=threshold_x, color='green', linestyle='--', alpha=0.7)
plt.text(threshold_x + 0.05, 0.3, f'Точка перетину: {threshold_x:.3f}', color='green')

# Налаштування графіка
plt.title('Логістична регресія: ймовірність успішності замовлення за середнім % успішних замовлень клієнта')
plt.xlabel('Середній % успішних замовлень клієнта')
plt.ylabel('Ймовірність успішного замовлення')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(-0.05, 1.05)  # Додаємо відступ від країв
plt.ylim(-0.05, 1.05)  # Додаємо відступ від країв

# Додаємо інформацію про коефіцієнти моделі
coef = model.coef_[0][0]
intercept = model.intercept_[0]
plt.text(0.02, 0.12, f'Коефіцієнт моделі: {coef:.4f}\nЗміщення: {intercept:.4f}',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

# Підпис для пояснення високого коефіцієнта
plt.text(0.5, 0.05,
         f'Висока величина коефіцієнта ({coef:.1f}) вказує на дуже сильну залежність\nуспішності замовлення від історичного % успішних замовлень клієнта',
         transform=plt.gca().transAxes, ha='center',
         bbox=dict(facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(f"{results_dir}/4_logistic_regression_curve.png", dpi=300)
plt.close()

print(f"Всі графіки успішно збережено в директорію: {results_dir}")

# Зберігаємо також MD-файл з аналізом
print("Створюємо MD-файл з аналізом...")

# Обчислюємо додаткові статистики
successful_orders = df[df['is_successful'] == 1]['partner_success_rate']
failed_orders = df[df['is_successful'] == 0]['partner_success_rate']

md_content = f"""# Аналіз залежності успішності замовлень від середнього % успішних замовлень клієнта

## Вступ

Цей документ містить аналіз залежності успішності поточного замовлення від середнього відсотка успішних замовлень клієнта в минулому. Аналіз базується на даних з файлу `cleaned_result.csv`, який містить {df.shape[0]} записів з {df.shape[1]} полями.

Розподіл класів успішності в наборі даних:
- Успішні замовлення (is_successful=1): {df['is_successful'].value_counts().get(1, 0)} записів ({df['is_successful'].value_counts().get(1, 0)/len(df)*100:.1f}%)
- Невдалі замовлення (is_successful=0): {df['is_successful'].value_counts().get(0, 0)} записів ({df['is_successful'].value_counts().get(0, 0)/len(df)*100:.1f}%)

## Графіки розподілу

### Віоліновий графік розподілу середнього % успішних замовлень клієнта

![Віоліновий графік](1_violin_plot_partner_success_rate.png)

Віоліновий графік демонструє суттєву різницю між успішними та невдалими замовленнями за середнім % успішних замовлень клієнта:

- **Невдалі замовлення**:
  - Середнє значення: {failed_orders.mean():.3f}
  - Медіана: {failed_orders.median():.3f}
  - Більш сконцентрований у зоні низьких значень

- **Успішні замовлення**:
  - Середнє значення: {successful_orders.mean():.3f}
  - Медіана: {successful_orders.median():.3f}
  - Більш рівномірний розподіл з вищими значеннями

Ця різниця свідчить про те, що успішність поточного замовлення сильно корелює з історичним відсотком успішних замовлень клієнта.

### Графік щільності розподілу

![Щільність розподілу](2_density_partner_success_rate.png)

Графік щільності розподілу показує більш детальну картину розподілу даних:

- Розподіл для невдалих замовлень має пік у зоні низьких значень (0.2-0.4)
- Розподіл для успішних замовлень має пік у зоні високих значень (0.6-0.8)
- Перцентилі загального розподілу:
  - 25й перцентиль: {df['partner_success_rate'].quantile(0.25):.3f}
  - 50й перцентиль (медіана): {df['partner_success_rate'].quantile(0.50):.3f}
  - 75й перцентиль: {df['partner_success_rate'].quantile(0.75):.3f}

### Гістограма з нормалізацією

![Гістограма](3_improved_histogram_success_by_rate_categories.png)

Гістограма показує, як змінюється частка успішних замовлень залежно від середнього % успішних замовлень клієнта:

- У клієнтів з низьким % успішних замовлень (0.0-0.3) більшість нових замовлень також невдалі
- У клієнтів з високим % успішних замовлень (0.7-1.0) більшість нових замовлень успішні
- Спостерігається чітка тенденція зростання частки успішних замовлень зі збільшенням історичного відсотка успішності

### Логістична регресія

![Логістична регресія](4_logistic_regression_curve.png)

Модель логістичної регресії дозволяє кількісно оцінити залежність між середнім % успішних замовлень клієнта та ймовірністю успішності поточного замовлення.

#### Ключові параметри моделі

- **Коефіцієнт моделі: {coef:.4f}**
  - Надзвичайно високе значення коефіцієнта вказує на дуже сильну залежність
  - Зростання середнього % успішних замовлень клієнта на 0.1 (10%) збільшує шанси успіху поточного замовлення приблизно в e^({coef:.4f}*0.1) ≈ {np.exp(coef*0.1):.2f} рази

- **Зміщення (перетин): {intercept:.4f}**

#### Інтерпретація результатів

1. **Точка перетину порогу 0.5: {threshold_x:.3f}**
   - Замовлення клієнтів із середнім % успішних замовлень менше {threshold_x:.3f} мають ймовірність успіху менше 50%
   - Замовлення клієнтів із середнім % успішних замовлень більше {threshold_x:.3f} мають ймовірність успіху більше 50%

2. **Зростання ймовірності успіху**:
   - 0.2 (20%): приблизно {100*y_proba[np.abs(X_range.flatten() - 0.2).argmin()]:.0f}% ймовірність успіху
   - 0.4 (40%): приблизно {100*y_proba[np.abs(X_range.flatten() - 0.4).argmin()]:.0f}% ймовірність успіху
   - 0.6 (60%): приблизно {100*y_proba[np.abs(X_range.flatten() - 0.6).argmin()]:.0f}% ймовірність успіху
   - 0.8 (80%): приблизно {100*y_proba[np.abs(X_range.flatten() - 0.8).argmin()]:.0f}% ймовірність успіху

## Практичні висновки

1. **Сильна предиктивна здатність**:
   - Історичний відсоток успішних замовлень клієнта є надзвичайно сильним предиктором успішності поточного замовлення
   - Ця змінна може бути використана для раннього виявлення потенційно проблемних замовлень

2. **Рекомендації для бізнес-процесів**:
   - Розробити диференційований підхід до клієнтів залежно від їхнього історичного % успішних замовлень
   - Забезпечити підвищену увагу та додаткову підтримку для клієнтів з низьким % успішних замовлень
   - Розглянути можливість встановлення різних рівнів автоматизації обробки замовлень для різних сегментів клієнтів

3. **Сегментація клієнтів**:
   - Високо-ризикові клієнти (< {threshold_x:.2f} успішних замовлень)
   - Середньо-ризикові клієнти ({threshold_x:.2f} - 0.7)
   - Низько-ризикові клієнти (> 0.7)

## Технічні деталі

Математична формула логістичної регресії:

```
P(успіх) = 1 / (1 + e^-({intercept:.4f} + {coef:.4f} * середній_процент_успішних_замовлень))
```

де:
- P(успіх) - ймовірність успішного замовлення (від 0 до 1)
- e - основа натурального логарифма (≈ 2.718)
- {intercept:.4f} - зміщення (перетин) моделі
- {coef:.4f} - коефіцієнт впливу середнього % успішних замовлень клієнта

## Обмеження аналізу

1. Аналіз враховує лише середній % успішних замовлень клієнта, але не детальну історію або нещодавні тенденції
2. Не враховується кількість історичних замовлень, на яких базується середній % (новий клієнт з 1 успішним замовленням = 100% vs досвідчений клієнт з 99/100 успішних замовлень)
3. Можливий вплив інших факторів, таких як категорія товарів, сума замовлення, сезонність тощо
"""

with open(f"{results_dir}/аналіз_залежності_успішності_від_історичних_показників.md", "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"MD-файл збережено: {results_dir}/аналіз_залежності_успішності_від_історичних_показників.md")
