import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import logging
import os

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Створення директорії для графіків, якщо вона не існує
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    logger.info(f"Створено директорію '{plots_dir}' для збереження графіків")
else:
    logger.info(f"Директорія '{plots_dir}' вже існує")

# Завантаження даних
try:
    df = pd.read_csv('cleanest_data.csv')
    logger.info(f"Завантажено {len(df)} рядків з файлу cleanest_data.csv")
except Exception as e:
    logger.error(f"Помилка при завантаженні файлу: {e}")
    exit(1)

# Перетворення create_date у datetime
try:
    df['create_date'] = pd.to_datetime(df['create_date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    logger.info("Колонка create_date перетворена у формат datetime")
except Exception as e:
    logger.warning(f"Помилка при перетворенні create_date: {e}")
    exit(1)

# Виключення даних за 2025 рік
df_filtered = df[df['create_date'].dt.year < 2025]
excluded_count = len(df) - len(df_filtered)
if excluded_count > 0:
    logger.info(f"Виключено {excluded_count} рядків з даними за 2025 рік")
else:
    logger.info("Даних за 2025 рік не виявлено")

# Створення колонки create_date_months
min_date = df_filtered['create_date'].min()
df_filtered['create_date_months'] = ((df_filtered['create_date'] - min_date).dt.days / 30.44).round(2)
logger.info(f"Додано числову колонку 'create_date_months': місяці від {min_date}")

# 1. Графік середньої успішності за місяцями
plt.figure(figsize=(10, 6))
success_by_month = df_filtered.groupby(df_filtered['create_date_months'].round()).agg({'is_successful': 'mean'}).reset_index()
plt.plot(success_by_month['create_date_months'], success_by_month['is_successful'], 'o-', alpha=0.7)
plt.title('Середня успішність замовлень за місяцями', fontsize=14)
plt.xlabel('Місяці від початку даних')
plt.ylabel('Частка успішних замовлень')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_average_success_by_month.png'), dpi=300)
logger.info("Графік збережено як '01_average_success_by_month.png'")
plt.close()

# 2. Графік з ковзним середнім для згладжування
plt.figure(figsize=(10, 6))
# Сортуємо дані за часом
sorted_data = df_filtered.sort_values('create_date_months')
# Створюємо бінінг даних за місяцями
bins = pd.cut(sorted_data['create_date_months'], bins=min(50, int(sorted_data['create_date_months'].max()) + 1))
success_by_bin = sorted_data.groupby(bins)['is_successful'].mean().reset_index()
# Перетворюємо біни в числові значення для графіка
success_by_bin['month_value'] = success_by_bin['create_date_months'].apply(lambda x: x.mid)

plt.plot(success_by_bin['month_value'], success_by_bin['is_successful'], 'o-', linewidth=2)
plt.title('Успішність замовлень (ковзне середнє)', fontsize=14)
plt.xlabel('Місяці від початку даних')
plt.ylabel('Частка успішних замовлень')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '02_moving_average_success.png'), dpi=300)
logger.info("Графік збережено як '02_moving_average_success.png'")
plt.close()

# 3. Щільність розподілу замовлень за часом із поділом на успішні/неуспішні
plt.figure(figsize=(10, 6))
# Додаємо коментар про ефект границь на графіку щільності
"""
Примітка: Зниження щільності до нуля на початку та в кінці графіка є методологічним ефектом
оцінки щільності ядра (KDE), а не обов'язково реальним зменшенням кількості замовлень.
Це "ефект границь", який виникає тому що алгоритм KDE має менше даних на краях періоду 
спостереження. Для точного розуміння зміни кількості замовлень потрібно дивитися на 
гістограму кількості замовлень за місяцями.
"""
sns.kdeplot(data=df_filtered[df_filtered['is_successful'] == 1], x='create_date_months',
            label='Успішні замовлення', alpha=0.6)
sns.kdeplot(data=df_filtered[df_filtered['is_successful'] == 0], x='create_date_months',
            label='Неуспішні замовлення', alpha=0.6)
plt.title('Щільність розподілу замовлень за часом', fontsize=14)
plt.xlabel('Місяці від початку даних')
plt.ylabel('Щільність')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '03_density_distribution.png'), dpi=300)
logger.info("Графік збережено як '03_density_distribution.png'")
plt.close()

# 4. Співвідношення успішних/неуспішних замовлень за квартилями часу
plt.figure(figsize=(10, 6))
quartiles = pd.qcut(df_filtered['create_date_months'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
success_by_quartile = df_filtered.groupby(quartiles)['is_successful'].value_counts(normalize=True).unstack().fillna(0)
success_by_quartile.plot(kind='bar', stacked=True)
plt.title('Співвідношення успішних/неуспішних замовлень за періодами', fontsize=14)
plt.xlabel('Квартиль періоду даних')
plt.ylabel('Частка')
plt.legend(['Неуспішні', 'Успішні'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '04_quartile_success_ratio.png'), dpi=300)
logger.info("Графік збережено як '04_quartile_success_ratio.png'")
plt.close()

# 5. Тренд успішності з часом (регресійний аналіз)
plt.figure(figsize=(10, 6))
sns.regplot(x='create_date_months', y='is_successful', data=df_filtered, scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
plt.title('Тренд успішності замовлень з часом', fontsize=14)
plt.xlabel('Місяці від початку даних')
plt.ylabel('Ймовірність успіху (1=успіх, 0=невдача)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '05_success_trend.png'), dpi=300)
logger.info("Графік тренду збережено як '05_success_trend.png'")
plt.close()

# 6. Додатковий графік: кількість замовлень за місяцями
plt.figure(figsize=(10, 6))
orders_by_month = df_filtered.groupby(df_filtered['create_date_months'].round()).size().reset_index(name='count')
plt.bar(orders_by_month['create_date_months'], orders_by_month['count'], alpha=0.7, width=0.8)
plt.title('Кількість замовлень за місяцями', fontsize=14)
plt.xlabel('Місяці від початку даних')
plt.ylabel('Кількість замовлень')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '06_orders_count_by_month.png'), dpi=300)
logger.info("Графік збережено як '06_orders_count_by_month.png'")
plt.close()

# 7. Додатковий графік: Успішність замовлень за роками з додаванням відсотків на стовпцях
plt.figure(figsize=(10, 6))
df_filtered['year'] = df_filtered['create_date'].dt.year
success_by_year = df_filtered.groupby('year')['is_successful'].agg(['mean', 'count']).reset_index()
ax = plt.bar(success_by_year['year'], success_by_year['mean'], alpha=0.7)

# Додаємо підписи з кількістю замовлень та відсотками успішності
for i, v in enumerate(success_by_year['mean']):
    # Додаємо кількість замовлень
    plt.text(i, v + 0.02, f"{success_by_year['count'][i]:,}", ha='center')
    # Додаємо відсоток успішності
    plt.text(i, v - 0.03, f"{v:.1%}", ha='top', color='black', fontweight='bold')

plt.title('Успішність замовлень за роками', fontsize=14)
plt.xlabel('Рік')
plt.ylabel('Частка успішних замовлень')
plt.ylim(0, max(success_by_year['mean']) * 1.2)  # Трохи більший діапазон для підписів
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '07_success_by_year.png'), dpi=300)
logger.info("Графік успішності за роками збережено як '07_success_by_year.png'")
plt.close()

# Виведення стислої статистики
print("\nСтатистика успішності замовлень за періодами:")
for period, data in df_filtered.groupby(pd.cut(df_filtered['create_date_months'], 4)):
    success_rate = data['is_successful'].mean()
    orders_count = len(data)
    print(f"Період {period}: {success_rate:.2%} успішних замовлень (всього {orders_count})")

# Статистика за роками
print("\nСтатистика за роками:")
yearly_stats = df_filtered.groupby(df_filtered['create_date'].dt.year)['is_successful'].agg(['mean', 'count'])
for year, stats in yearly_stats.iterrows():
    print(f"Рік {year}: {stats['mean']:.2%} успішних замовлень (всього {stats['count']})")

# Додаємо інформацію про ефект границь у текстовий файл
boundary_effect_info = """
ПРИМІТКА ПРО ЕФЕКТ ГРАНИЦЬ НА ГРАФІКУ ЩІЛЬНОСТІ РОЗПОДІЛУ:
Зниження щільності до нуля на початку та в кінці графіка щільності розподілу є 
методологічним ефектом оцінки щільності ядра (KDE), а не обов'язково реальним 
зменшенням кількості замовлень. Це "ефект границь", який виникає тому що алгоритм 
KDE має менше даних на краях періоду спостереження. Для точного розуміння зміни 
кількості замовлень потрібно дивитися на гістограму кількості замовлень за місяцями.
"""

# Збереження аналітичної інформації в текстовий файл
with open(os.path.join(plots_dir, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Аналіз успішності замовлень\n")
    f.write(f"=========================\n\n")
    f.write(f"Період даних: з {min_date.strftime('%Y-%m-%d')} по {df_filtered['create_date'].max().strftime('%Y-%m-%d')}\n")
    f.write(f"Загальна кількість замовлень: {len(df_filtered)}\n")
    f.write(f"Виключено даних за 2025 рік: {excluded_count}\n")
    f.write(f"Середня успішність: {df_filtered['is_successful'].mean():.2%}\n")
    f.write(f"Тривалість періоду: {df_filtered['create_date_months'].max():.1f} місяців\n\n")

    f.write("Статистика за періодами:\n")
    for period, data in df_filtered.groupby(pd.cut(df_filtered['create_date_months'], 4)):
        success_rate = data['is_successful'].mean()
        orders_count = len(data)
        f.write(f"- Період {period}: {success_rate:.2%} успішних замовлень (всього {orders_count})\n")

    f.write("\nСтатистика за роками:\n")
    for year, stats in yearly_stats.iterrows():
        f.write(f"- Рік {year}: {stats['mean']:.2%} успішних замовлень (всього {stats['count']})\n")

    # Додаємо пояснення про ефект границь
    f.write(f"\n{boundary_effect_info}")

logger.info(f"Аналіз успішно завершено. Всі графіки збережено в директорії '{plots_dir}'")
