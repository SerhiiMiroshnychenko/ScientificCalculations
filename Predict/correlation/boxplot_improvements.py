import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from datetime import datetime
import os

# Створюємо директорію для збереження графіків
results_dir = f"boxplot_improvements_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(results_dir, exist_ok=True)

# Завантаження даних
print("Завантажуємо дані з CSV-файлу...")
df = pd.read_csv("cleaned_result.csv")
print(f"Завантажено {df.shape[0]} рядків та {df.shape[1]} стовпців")

# Перевіряємо наявність пропущених значень
missing_values = df['order_messages'].isnull().sum()
if missing_values > 0:
    print(f"Виявлено {missing_values} пропущених значень в order_messages, заповнюємо нулями")
    df['order_messages'] = df['order_messages'].fillna(0)

# Налаштовуємо загальний стиль графіків
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Варіант 1: Обмеження діапазону осі Y
print("Створюємо графік з обмеженим діапазоном осі Y...")
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='is_successful', y='order_messages', data=df)

# Встановлюємо максимальне значення для осі Y до 95-го перцентиля
y_max = min(df['order_messages'].quantile(0.95), 50)
ax.set_ylim(0, y_max)

# Додаємо текст про викиди
plt.text(0.5, y_max*0.9, f"Викиди: макс. {df['order_messages'].max():.0f} повідомлень",
         ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Налаштування графіка
ax.set_title('Розподіл кількості повідомлень за успішністю замовлення\n(обмежений діапазон)')
ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
ax.set_ylabel('Кількість повідомлень')

# Додаємо медіану та середнє значення як текст
for i, success in enumerate([0, 1]):
    subset = df[df['is_successful'] == success]['order_messages']
    median = subset.median()
    mean = subset.mean()
    plt.text(i, median + 1, f'Медіана: {median:.1f}', ha='center')
    plt.text(i, mean + 3, f'Середнє: {mean:.1f}', ha='center')

plt.tight_layout()
plt.savefig(f"{results_dir}/1_boxplot_limited_range.png", dpi=300)
plt.close()

# Варіант 2: Використання двох графіків: основний та вставка для викидів
print("Створюємо графік з основним та додатковим зображенням...")
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# Основний графік з обмеженим діапазоном
ax1 = plt.subplot(gs[0])
sns.boxplot(x='is_successful', y='order_messages', data=df, ax=ax1)
ax1.set_ylim(0, 30)  # Обмежуємо для кращої видимості основної частини даних
ax1.set_title('Розподіл кількості повідомлень (основний діапазон)')
ax1.set_xlabel('')  # Прибираємо підпис осі X для основного графіка

# Графік з повним діапазоном для відображення викидів
ax2 = plt.subplot(gs[1])
sns.boxplot(x='is_successful', y='order_messages', data=df, ax=ax2)
ax2.set_title('Повний діапазон з викидами')
ax2.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
ax2.set_ylabel('Кількість повідомлень')

# Додаємо медіану та середнє значення як текст
for i, success in enumerate([0, 1]):
    subset = df[df['is_successful'] == success]['order_messages']
    median = subset.median()
    mean = subset.mean()
    ax1.text(i, median + 1, f'Медіана: {median:.1f}', ha='center')
    ax1.text(i, mean + 3, f'Середнє: {mean:.1f}', ha='center')

# Налаштування загального вигляду
plt.tight_layout()
plt.savefig(f"{results_dir}/2_boxplot_with_subplot.png", dpi=300)
plt.close()

# Варіант 3: Використання віолінового графіка (Violin Plot) замість коробкового
print("Створюємо віоліновий графік...")
plt.figure(figsize=(10, 6))

# Створюємо віоліновий графік
ax = sns.violinplot(x='is_successful', y='order_messages', data=df,
                    inner='box', cut=0, density_norm='width')

# Налаштування графіка
ax.set_ylim(0, min(df['order_messages'].quantile(0.95), 50))
ax.set_title('Розподіл кількості повідомлень за успішністю замовлення\n(віоліновий графік)')
ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
ax.set_ylabel('Кількість повідомлень')

# Додаємо медіану та середнє значення як текст
for i, success in enumerate([0, 1]):
    subset = df[df['is_successful'] == success]['order_messages']
    median = subset.median()
    mean = subset.mean()
    plt.text(i, median + 1, f'Медіана: {median:.1f}', ha='center')
    plt.text(i, mean + 3, f'Середнє: {mean:.1f}', ha='center')

plt.tight_layout()
plt.savefig(f"{results_dir}/3_violin_plot.png", dpi=300)
plt.close()

# Варіант 4: Використання swarm plot для показу розподілу точок
print("Створюємо графік з розподілом точок (swarm plot)...")
plt.figure(figsize=(12, 6))

# Створюємо випадкову вибірку даних для кращої наочності
sample_size = min(2000, len(df))
df_sample = df.sample(sample_size, random_state=42)

# Основний графік - коробковий
ax = sns.boxplot(x='is_successful', y='order_messages', data=df_sample,
                 width=0.4, fliersize=0)

# Додаємо swarm plot для відображення розподілу точок
sns.swarmplot(x='is_successful', y='order_messages', data=df_sample,
              size=1, color='black', alpha=0.5)

# Обмежуємо вісь Y
ax.set_ylim(0, 30)
ax.set_title('Розподіл кількості повідомлень за успішністю замовлення\n(з розподілом точок)')
ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
ax.set_ylabel('Кількість повідомлень')

# Додаємо медіану та середнє значення як текст
for i, success in enumerate([0, 1]):
    subset = df_sample[df_sample['is_successful'] == success]['order_messages']
    median = subset.median()
    mean = subset.mean()
    plt.text(i, median + 1, f'Медіана: {median:.1f}', ha='center')
    plt.text(i, mean + 3, f'Середнє: {mean:.1f}', ha='center')

plt.tight_layout()
plt.savefig(f"{results_dir}/4_boxplot_with_swarm.png", dpi=300)
plt.close()

# Варіант 5: Використання strip plot з jitter ефектом
print("Створюємо графік з jitter ефектом (strip plot)...")
plt.figure(figsize=(10, 6))

# Обмежуємо максимальне значення для кращої видимості
df_plot = df.copy()
upper_limit = df['order_messages'].quantile(0.95)
df_plot['order_messages_capped'] = df_plot['order_messages'].clip(upper=upper_limit)

# Основний графік - коробковий
ax = sns.boxplot(x='is_successful', y='order_messages_capped', data=df_plot,
                 width=0.4, fliersize=0)

# Додаємо strip plot для відображення розподілу точок
sns.stripplot(x='is_successful', y='order_messages_capped', data=df_plot.sample(5000),
              jitter=True, size=2.5, alpha=0.3)

# Налаштування графіка
ax.set_title('Розподіл кількості повідомлень за успішністю замовлення\n(з jitter ефектом, обмежено 95-им перцентилем)')
ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
ax.set_ylabel('Кількість повідомлень')

# Додаємо примітку про обмеження
plt.text(0.5, upper_limit*0.95, f"Обмежено до {upper_limit:.0f} повідомлень\n(95й перцентиль)",
         ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Додаємо медіану та середнє значення як текст
for i, success in enumerate([0, 1]):
    subset = df_plot[df_plot['is_successful'] == success]['order_messages_capped']
    median = subset.median()
    mean = subset.mean()
    plt.text(i, median + 1, f'Медіана: {median:.1f}', ha='center')
    plt.text(i, mean + 3, f'Середнє: {mean:.1f}', ha='center')

plt.tight_layout()
plt.savefig(f"{results_dir}/5_boxplot_with_stripplot.png", dpi=300)
plt.close()

print(f"Всі графіки успішно збережено в директорію: {results_dir}")
