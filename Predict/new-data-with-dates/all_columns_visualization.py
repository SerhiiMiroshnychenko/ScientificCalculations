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

# Створюємо головну директорію для збереження графіків
results_dir = f"all_columns_analysis"
os.makedirs(results_dir, exist_ok=True)

# Встановлюємо українську локаль для графіків
import locale

try:
    locale.setlocale(locale.LC_ALL, 'uk_UA.UTF-8')
except:
    print("Українська локаль не знайдена, використовуємо стандартну")

# Завантаження даних
print("Завантажуємо дані з CSV-файлу...")
df = pd.read_csv("cleanest_data.csv")
print(f"Завантажено {df.shape[0]} рядків та {df.shape[1]} стовпців")
print(f"Розподіл класів: {df['is_successful'].value_counts().to_dict()}")

# Налаштовуємо загальний стиль графіків
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})


# Функція для створення директорії для колонки
def ensure_column_dir(column_name):
    """
    Створює директорію для збереження графіків конкретної колонки

    Args:
        column_name (str): Назва колонки

    Returns:
        str: Шлях до директорії
    """
    # Замінюємо недопустимі символи у назві колонки
    safe_column_name = "".join([c if c.isalnum() or c in ['-', '_'] else '_' for c in column_name])
    column_dir = os.path.join(results_dir, safe_column_name)
    os.makedirs(column_dir, exist_ok=True)
    return column_dir


# Функція для обробки пропущених та від'ємних значень
def preprocess_column(df, column_name):
    if df[column_name].dtype in [np.int64, np.float64]:
        # Обробка від'ємних значень для числових колонок
        negative_values = (df[column_name] < 0).sum()
        if negative_values > 0:
            print(f"Виявлено {negative_values} від'ємних значень в {column_name}, замінюємо на нулі")
            df.loc[df[column_name] < 0, column_name] = 0
            print(f"Після обробки від'ємних значень: мін = {df[column_name].min()}, макс = {df[column_name].max()}")

    # Обробка пропущених значень для будь-яких типів колонок
    missing_values = df[column_name].isnull().sum()
    if missing_values > 0:
        print(f"Виявлено {missing_values} пропущених значень в {column_name}")
        if df[column_name].dtype in [np.int64, np.float64]:
            print(f"Заповнюємо медіаною для числової колонки {column_name}")
            df[column_name] = df[column_name].fillna(df[column_name].median())
        else:
            print(f"Заповнюємо найчастішим значенням для категоріальної колонки {column_name}")
            df[column_name] = df[column_name].fillna(df[column_name].mode()[0])

    return df


# Визначаємо функцію для створення віолінового графіка
def create_violin_plot(df, column_name):
    if df[column_name].dtype not in [np.int64, np.float64]:
        print(f"Колонка {column_name} не є числовою. Пропускаємо віоліновий графік.")
        return

    print(f"Створюємо віоліновий графік для {column_name}...")
    plt.figure(figsize=(10, 6))

    # Визначаємо межу для 95-го перцентиля
    p95 = df[column_name].quantile(0.95)

    # Фільтруємо дані за 95-м перцентилем для кращої візуалізації
    df_filtered = df[df[column_name] <= p95]

    # Створюємо віоліновий графік
    ax = sns.violinplot(x='is_successful', y=column_name, data=df_filtered,
                        inner='box', cut=0, density_norm='width')

    # Налаштування графіка
    ax.set_title(f'Розподіл {column_name} за успішністю замовлення\n(віоліновий графік)')
    ax.set_xlabel('Успішність замовлення (0 - невдале, 1 - успішне)')
    ax.set_ylabel(column_name)

    # Додаємо медіану та середнє значення як текст
    for i, success in enumerate([0, 1]):
        subset = df[df['is_successful'] == success][column_name]
        median = subset.median()
        mean = subset.mean()
        plt.text(i, median - p95 * 0.1, f'Медіана: {median:.1f}', ha='center')
        plt.text(i, median + p95 * 0.1, f'Середнє: {mean:.1f}', ha='center')

    # Додаємо інформацію про викиди
    plt.text(0.5, p95 * 0.9, f"95-й перцентиль: {p95:.0f}\nМакс. значення: {df[column_name].max():.0f}",
             ha='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    # Створюємо директорію для колонки
    column_dir = ensure_column_dir(column_name)
    plt.savefig(f"{column_dir}/violin_plot_{column_name}.png", dpi=300)
    plt.close()


# Визначаємо функцію для створення графіка щільності розподілу
def create_density_plot(df, column_name):
    if df[column_name].dtype not in [np.int64, np.float64]:
        print(f"Колонка {column_name} не є числовою. Пропускаємо графік щільності.")
        return

    print(f"Створюємо графік щільності розподілу для {column_name}...")

    # Створюємо один графік
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Визначаємо межу для 95-го перцентиля
    p95 = df[column_name].quantile(0.95)

    # Малюємо графік щільності для кожної категорії
    for i, (success, color, label) in enumerate([(0, 'forestgreen', 'Невдалі замовлення'),
                                                 (1, 'crimson', 'Успішні замовлення')]):
        subset = df[df['is_successful'] == success][column_name]
        subset_filtered = subset[subset <= p95]  # Фільтруємо для кращої візуалізації

        if len(subset_filtered) > 0:
            mean_val = subset.mean()
            median_val = subset.median()

            # Малюємо графік щільності
            sns.kdeplot(data=subset_filtered, ax=ax, color=color, fill=True, alpha=0.5, label=f"{label}")

            # Додаємо вертикальні лінії
            plt.axvline(x=mean_val, color=color, linestyle='--', alpha=0.7)
            plt.axvline(x=median_val, color=color, linestyle=':', alpha=0.7)

            # Додаємо текст у верхній правий кут
            plt.text(0.98, 0.85 - i * 0.1,
                     f"{label}:\nСереднє: {mean_val:.1f}\nМедіана: {median_val:.1f}",
                     transform=ax.transAxes,  # Координати відносно графіка (0-1)
                     color=color, ha='right', va='top',
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Додаємо інформацію про максимальне значення
    plt.text(0.98, 0.65, f"95-й перцентиль: {p95:.0f}\nМакс. значення: {df[column_name].max():.0f}",
             transform=ax.transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8))

    # Налаштування графіка
    plt.xlim(0, p95 * 1.1)  # Показуємо до 95-го перцентиля + 10%
    plt.ylim(0, plt.ylim()[1] * 1.1)  # Додаємо 10% простору зверху
    plt.title(f'Щільність розподілу {column_name} за успішністю замовлення\n(до 95-го перцентиля)')
    plt.xlabel(column_name)
    plt.ylabel('Щільність')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    plt.tight_layout()
    # Створюємо директорію для колонки
    column_dir = ensure_column_dir(column_name)
    plt.savefig(f"{column_dir}/density_{column_name}_by_success.png", dpi=300)
    plt.close()


# Визначаємо функцію для створення графіка вірогідності успіху залежно від значення колонки
# Визначаємо функцію для створення графіка вірогідності успіху залежно від значення колонки
def create_success_rate_plot(df, column_name):
    if df[column_name].dtype not in [np.int64, np.float64]:
        print(f"Колонка {column_name} не є числовою. Пропускаємо графік вірогідності успіху.")
        return

    print(f"Створюємо графік вірогідності успіху залежно від {column_name}...")

    # Визначаємо межу для 95-го перцентиля
    p95 = df[column_name].quantile(0.95)

    # Використовуємо біни для групування даних за значенням колонки
    bins = 20
    df_filtered = df[df[column_name] <= p95].copy()

    # Перевіряємо, чи є достатньо даних і варіація
    if df_filtered[column_name].nunique() < 3 or len(df_filtered) < 10:
        print(f"Недостатньо унікальних значень або записів для {column_name}. Пропускаємо графік вірогідності успіху.")
        return

    df_filtered[f'{column_name}_bin'] = pd.cut(df_filtered[column_name], bins=bins)

    # Розраховуємо вірогідність успіху та середину біна
    success_rates = df_filtered.groupby(f'{column_name}_bin', observed=False)['is_successful'].mean().reset_index()
    bin_centers = df_filtered.groupby(f'{column_name}_bin', observed=False)[column_name].mean().reset_index()
    bin_counts = df_filtered.groupby(f'{column_name}_bin', observed=False).size().reset_index(name='count')

    # Об'єднуємо результати
    success_rates = pd.merge(success_rates, bin_centers, on=f'{column_name}_bin')
    success_rates = pd.merge(success_rates, bin_counts, on=f'{column_name}_bin')

    # Видаляємо рядки з NaN або inf значеннями
    success_rates = success_rates.replace([np.inf, -np.inf], np.nan).dropna()

    # Перевіряємо, чи залишилося достатньо даних
    if len(success_rates) < 3:
        print(f"Після обробки залишилося недостатньо даних для {column_name}. Пропускаємо графік вірогідності успіху.")
        return

    # Створюємо графік
    plt.figure(figsize=(14, 8))

    # Основний графік: вірогідність успіху
    ax1 = plt.gca()
    ax1.scatter(success_rates[column_name], success_rates['is_successful'],
                s=success_rates['count'] / 10, alpha=0.7, color='crimson')

    # Додаємо лінію тренду, тільки якщо є варіація в даних
    if success_rates[column_name].nunique() > 1 and success_rates['is_successful'].var() > 0:
        try:
            z = np.polyfit(success_rates[column_name], success_rates['is_successful'], 1)
            p = np.poly1d(z)
            plt.plot(success_rates[column_name], p(success_rates[column_name]), "r--", alpha=0.7,
                     label=f"Тренд: y={z[0]:.6f}x+{z[1]:.4f}")

            # Додаємо коефіцієнт кореляції
            corr = df_filtered[[column_name, 'is_successful']].corr().iloc[0, 1]
            plt.text(0.02, 0.95, f"Кореляція: {corr:.3f}", transform=ax1.transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))
        except np.linalg.LinAlgError:
            print(f"Не вдалося обчислити лінію тренду для {column_name}.")
            plt.text(0.02, 0.95, "Не вдалося обчислити лінію тренду", transform=ax1.transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))
    else:
        print(f"Недостатня варіація в даних для {column_name} для обчислення лінії тренду.")
        plt.text(0.02, 0.95, "Недостатня варіація для лінії тренду", transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

    # Додаємо другу вісь Y для гістограми
    ax2 = ax1.twinx()

    # Обчислюємо ширину стовпців гістограми
    bin_width = (success_rates[column_name].max() - success_rates[column_name].min()) / (len(success_rates) + 1)
    if bin_width <= 0:
        bin_width = p95 / bins  # Використовуємо запасний варіант, якщо розрахунок не вдався

    ax2.bar(success_rates[column_name], success_rates['count'], alpha=0.2, color='blue', width=bin_width)
    ax2.set_ylabel('Кількість замовлень в біні', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Налаштування графіка
    ax1.set_xlabel(column_name)
    ax1.set_ylabel('Вірогідність успіху замовлення', color='crimson')
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title(f'Вірогідність успіху замовлення залежно від {column_name}\n(до 95-го перцентиля)')

    plt.legend(loc='upper center')
    plt.tight_layout()
    # Створюємо директорію для колонки
    column_dir = ensure_column_dir(column_name)
    plt.savefig(f"{column_dir}/success_rate_{column_name}.png", dpi=300)
    plt.close()


# Визначаємо функцію для створення гістограми для категоріальних колонок
def create_category_histogram(df, column_name):
    if df[column_name].dtype in [np.int64, np.float64] and df[column_name].nunique() > 10:
        print(f"Колонка {column_name} має забагато унікальних значень для категоріальної гістограми.")
        return

    print(f"Створюємо гістограму для категоріальної колонки {column_name}...")

    # Підрахунок успішних/неуспішних замовлень для кожної категорії
    category_counts = df.groupby([column_name, 'is_successful']).size().unstack(fill_value=0)

    # Якщо колонка має менше 20 унікальних значень, показуємо всі категорії
    # В іншому випадку, показуємо тільки 20 найчастіших категорій
    if df[column_name].nunique() <= 20:
        categories_to_show = category_counts
    else:
        top_categories = df[column_name].value_counts().nlargest(20).index
        categories_to_show = category_counts.loc[top_categories]

    # Створюємо гістограму
    plt.figure(figsize=(14, 8))
    categories_to_show.plot(kind='bar', stacked=True, color=['forestgreen', 'crimson'])

    # Налаштування графіка
    plt.title(f'Розподіл {column_name} за успішністю замовлення')
    plt.xlabel(column_name)
    plt.ylabel('Кількість замовлень')
    plt.legend(['Неуспішні', 'Успішні'])
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Додаємо значення над стовпцями
    for i, (index, row) in enumerate(categories_to_show.iterrows()):
        total = row.sum()
        success_rate = row[1] / total if total > 0 else 0
        plt.text(i, total + 0.5, f"{success_rate:.1%}", ha='center')

    plt.tight_layout()
    # Створюємо директорію для колонки
    column_dir = ensure_column_dir(column_name)
    plt.savefig(f"{column_dir}/category_histogram_{column_name}.png", dpi=300)
    plt.close()


# Визначаємо функцію для створення віолінового графіка з логарифмічною шкалою
def create_log_violin_plot(df, column_name):
    if df[column_name].dtype not in [np.int64, np.float64]:
        print(f"Колонка {column_name} не є числовою. Пропускаємо віоліновий графік з логарифмічною шкалою.")
        return

    # Перевіряємо, чи є нульові або від'ємні значення
    if (df[column_name] <= 0).any():
        print(
            f"Колонка {column_name} містить нульові або від'ємні значення, які неможливо відобразити в логарифмічній шкалі.")
        print(f"Замінюємо їх на мінімальне позитивне значення для відображення.")
        min_positive = df[df[column_name] > 0][column_name].min()
        df_plot = df.copy()
        df_plot.loc[df_plot[column_name] <= 0, column_name] = min_positive / 10
    else:
        df_plot = df

    print(f"Створюємо violin plot з логарифмічною шкалою для {column_name}...")
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='is_successful', y=column_name, data=df_plot, inner='box', palette='Set3')
    plt.yscale('log')
    plt.title(f'Violin Plot для {column_name} з логарифмічною шкалою')
    plt.xlabel('Успішність')
    plt.ylabel(f'{column_name} (лог. шкала)')
    plt.xticks([0, 1], ['Неуспішні', 'Успішні'])

    # Додаємо статистичні показники як текст
    for i, success in enumerate([0, 1]):
        subset = df[df['is_successful'] == success][column_name]
        if len(subset) > 0:
            median = subset.median()
            mean = subset.mean()

            # Перевіряємо, що значення не нульові для відображення в логарифмічній шкалі
            if median > 0 and mean > 0:
                plt.text(i, median / 1.5, f'Медіана: {median:.1f}', ha='center', fontsize=9)
                plt.text(i, median * 1.5, f'Середнє: {mean:.1f}', ha='center', fontsize=9)

    # Зберігаємо графік
    # Створюємо директорію для колонки
    column_dir = ensure_column_dir(column_name)
    plt.savefig(f"{column_dir}/log_violin_plot_{column_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Функція для створення часових графіків
def create_time_series_plots(df, date_column):
    """
    Створює спеціалізовані графіки для часових даних

    Args:
        df (pd.DataFrame): Датафрейм з даними
        date_column (str): Назва колонки з датами
    """
    print(f"Створюємо часові графіки для колонки {date_column}...")

    # Переконуємося, що колонка дат має правильний формат
    try:
        df[date_column] = pd.to_datetime(df[date_column])
    except:
        print(f"Помилка перетворення колонки {date_column} у формат datetime. Пропускаємо часові графіки.")
        return

    # Створюємо директорію для колонки
    column_dir = ensure_column_dir(date_column)

    # 1. Графік кількості замовлень по днях
    plt.figure(figsize=(14, 8))

    # Створюємо новий датафрейм з кількістю замовлень по днях
    df_daily = df.copy()
    df_daily['date'] = df_daily[date_column].dt.date
    daily_orders = df_daily.groupby(['date', 'is_successful']).size().unstack(fill_value=0)

    if 1 not in daily_orders.columns:
        daily_orders[1] = 0
    if 0 not in daily_orders.columns:
        daily_orders[0] = 0

    # Обчислюємо загальну кількість замовлень
    daily_orders['total'] = daily_orders[0] + daily_orders[1]

    # Сортуємо за датою
    daily_orders = daily_orders.sort_index()

    # Малюємо графік
    ax = daily_orders[[0, 1]].plot(kind='bar', stacked=True, figsize=(14, 8),
                                   color=['forestgreen', 'crimson'])

    # Налаштування графіка
    plt.title('Кількість замовлень по днях')
    plt.xlabel('Дата')
    plt.ylabel('Кількість замовлень')
    plt.legend(['Неуспішні', 'Успішні'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Обертаємо мітки дат для кращої читабельності
    plt.xticks(rotation=45, ha='right')

    # Форматуємо мітки осі X, щоб відображалися тільки кожні N днів
    n_days = max(1, len(daily_orders) // 20)  # Виводимо не більше 20 міток
    for i, tick in enumerate(ax.xaxis.get_ticklabels()):
        if i % n_days != 0:
            tick.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{column_dir}/daily_orders.png", dpi=300)
    plt.close()

    # 2. Графік відсотка успішних замовлень по днях
    plt.figure(figsize=(14, 8))

    # Обчислюємо відсоток успішних замовлень
    success_rate = (daily_orders[1] / daily_orders['total'] * 100).fillna(0)

    # Малюємо графік
    ax = success_rate.plot(kind='line', marker='o', figsize=(14, 8), color='blue')

    # Додаємо середню лінію
    mean_rate = success_rate.mean()
    plt.axhline(y=mean_rate, color='r', linestyle='--', label=f'Середній % успіху: {mean_rate:.1f}%')

    # Налаштування графіка
    plt.title('Відсоток успішних замовлень по днях')
    plt.xlabel('Дата')
    plt.ylabel('Відсоток успішних замовлень (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Обертаємо мітки дат для кращої читабельності
    plt.xticks(rotation=45, ha='right')

    # Форматуємо мітки осі X, щоб відображалися тільки кожні N днів
    for i, tick in enumerate(ax.xaxis.get_ticklabels()):
        if i % n_days != 0:
            tick.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{column_dir}/daily_success_rate.png", dpi=300)
    plt.close()

    # 3. Теплова карта кількості замовлень по годинам і дням тижня
    plt.figure(figsize=(12, 8))

    # Створюємо новий датафрейм з годинами та днями тижня
    df_heatmap = df.copy()
    df_heatmap['hour'] = df_heatmap[date_column].dt.hour
    df_heatmap['day_of_week'] = df_heatmap[date_column].dt.day_name()

    # Перевпорядковуємо дні тижня
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Створюємо зведену таблицю
    heatmap_data = pd.pivot_table(df_heatmap, values='is_successful',
                                  index='day_of_week', columns='hour',
                                  aggfunc='count', fill_value=0)

    # Перевпорядковуємо індекси для правильного відображення днів тижня
    if not heatmap_data.empty:
        heatmap_data = heatmap_data.reindex(days_order, fill_value=0)

    # Малюємо теплову карту
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='g')

    # Налаштування графіка
    plt.title('Розподіл замовлень по годинам і дням тижня')
    plt.xlabel('Година дня')
    plt.ylabel('День тижня')

    plt.tight_layout()
    plt.savefig(f"{column_dir}/hour_day_heatmap.png", dpi=300)
    plt.close()

    # 4. Розподіл успішності замовлень по місяцях
    plt.figure(figsize=(12, 6))

    # Створюємо новий датафрейм з місяцями
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly[date_column].dt.month_name()
    df_monthly['month_num'] = df_monthly[date_column].dt.month

    # Групуємо дані по місяцях та успішності
    monthly_data = df_monthly.groupby(['month_num', 'month', 'is_successful']).size().unstack(fill_value=0)

    if 1 not in monthly_data.columns:
        monthly_data[1] = 0
    if 0 not in monthly_data.columns:
        monthly_data[0] = 0

    # Обчислюємо загальну кількість замовлень та відсоток успішних
    monthly_data['total'] = monthly_data[0] + monthly_data[1]
    monthly_data['success_rate'] = (monthly_data[1] / monthly_data['total'] * 100).fillna(0)

    # Сортуємо за номером місяця
    monthly_data = monthly_data.sort_index()

    # Створюємо подвійний графік: стовпчиковий для кількості та лінійний для %
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Стовпчиковий графік кількості замовлень
    bars = ax1.bar(monthly_data.index.get_level_values('month'), monthly_data['total'],
                   color='lightblue', alpha=0.7)
    ax1.set_xlabel('Місяць')
    ax1.set_ylabel('Кількість замовлень', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Додаємо значення над стовпцями
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.0f}', ha='center', va='bottom')

    # Створюємо другу вісь Y для відсотка успішності
    ax2 = ax1.twinx()
    ax2.plot(monthly_data.index.get_level_values('month'), monthly_data['success_rate'],
             'ro-', linewidth=2, markersize=8)
    ax2.set_ylabel('Відсоток успішних замовлень (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Додаємо значення поруч з точками
    for i, value in enumerate(monthly_data['success_rate']):
        ax2.text(i, value + 2, f'{value:.1f}%', color='red', ha='center')

    # Налаштування графіка
    plt.title('Кількість замовлень та відсоток успішних по місяцях')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{column_dir}/monthly_distribution.png", dpi=300)
    plt.close()

    # 5. Графік тренду середньої вартості замовлень у часі (якщо є колонка order_amount)
    if 'order_amount' in df.columns:
        plt.figure(figsize=(14, 8))

        # Створюємо новий датафрейм з датами
        df_amount = df.copy()
        df_amount['date'] = df_amount[date_column].dt.date

        # Групуємо дані по датах та обчислюємо середню вартість
        amount_by_date = df_amount.groupby(['date', 'is_successful'])['order_amount'].mean().unstack(fill_value=0)

        if 1 not in amount_by_date.columns:
            amount_by_date[1] = 0
        if 0 not in amount_by_date.columns:
            amount_by_date[0] = 0

        # Сортуємо за датою
        amount_by_date = amount_by_date.sort_index()

        # Малюємо графік
        ax = amount_by_date.plot(marker='o', figsize=(14, 8))

        # Налаштування графіка
        plt.title('Середня вартість замовлень по днях')
        plt.xlabel('Дата')
        plt.ylabel('Середня вартість замовлення')
        plt.legend(['Неуспішні', 'Успішні'])
        plt.grid(True, linestyle='--', alpha=0.7)

        # Обертаємо мітки дат для кращої читабельності
        plt.xticks(rotation=45, ha='right')

        # Форматуємо мітки осі X, щоб відображалися тільки кожні N днів
        for i, tick in enumerate(ax.xaxis.get_ticklabels()):
            if i % n_days != 0:
                tick.set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{column_dir}/avg_amount_trend.png", dpi=300)
        plt.close()

# Обробка та візуалізація всіх колонок
numerical_columns = df.select_dtypes(include=[np.int64, np.float64]).columns
categorical_columns = df.select_dtypes(exclude=[np.int64, np.float64]).columns

print("\nПочинаємо обробку та візуалізацію всіх колонок...")

# Обробляємо всі колонки
for column in df.columns:
    if column == 'is_successful':  # Пропускаємо колонку цільової змінної
        continue

    print(f"\nОбробка колонки: {column}")
    df = preprocess_column(df, column)

    # Створюємо відповідні графіки залежно від типу даних
    if column == 'create_date':
        # Особлива обробка для колонки дати
        create_time_series_plots(df, column)
    elif column in numerical_columns:
        create_violin_plot(df, column)
        create_log_violin_plot(df, column)
        create_density_plot(df, column)
        create_success_rate_plot(df, column)
    else:
        create_category_histogram(df, column)

print(f"\nВізуалізація всіх колонок завершена. Результати збережено в директорії {results_dir}")

# Виводимо підсумкову статистику
print("\nПідсумкова статистика:")
print(f"Всього замовлень: {df.shape[0]}")
print(f"Успішних замовлень: {df[df['is_successful'] == 1].shape[0]} ({df['is_successful'].mean():.1%})")
print(f"Неуспішних замовлень: {df[df['is_successful'] == 0].shape[0]} ({1 - df['is_successful'].mean():.1%})")
