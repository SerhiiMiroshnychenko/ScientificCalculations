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

    # Особлива обробка для днів тижня
    if column_name == 'day_of_week':
        # Порядок днів тижня від Понеділка до Неділі
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Перевпорядковуємо індекси для днів тижня
        categories_to_show = category_counts.reindex(days_order)
    # Особлива обробка для місяців
    elif column_name == 'month':
        # Порядок місяців від Січня до Грудня
        months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December']
        # Перевпорядковуємо індекси для місяців
        categories_to_show = category_counts.reindex(months_order)
    else:
        # Якщо колонка має менше 20 унікальних значень, показуємо всі категорії
        # В іншому випадку, показуємо тільки 20 найчастіших категорій
        if df[column_name].nunique() <= 20:
            categories_to_show = category_counts
        else:
            top_categories = df[column_name].value_counts().nlargest(20).index
            categories_to_show = category_counts.loc[top_categories]

    # Створюємо гістограму
    plt.figure(figsize=(14, 8))
    categories_to_show.plot(kind='bar', stacked=True, color=['blue', 'orange'])

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

    # 1. Графік кількості замовлень по місяцях
    plt.figure(figsize=(14, 8))

    # Створюємо новий датафрейм з групуванням по місяцях замість днів
    df_monthly = df.copy()
    # Додаємо фільтр, щоб виключити 2025 рік
    df_monthly = df_monthly[df_monthly[date_column].dt.year < 2025]
    df_monthly['year_month'] = df_monthly[date_column].dt.strftime('%Y-%m')
    monthly_orders = df_monthly.groupby(['year_month', 'is_successful']).size().unstack(fill_value=0)

    if 1 not in monthly_orders.columns:
        monthly_orders[1] = 0
    if 0 not in monthly_orders.columns:
        monthly_orders[0] = 0

    # Обчислюємо загальну кількість замовлень
    monthly_orders['total'] = monthly_orders[0] + monthly_orders[1]

    # Сортуємо за датою
    monthly_orders = monthly_orders.sort_index()

    # Створюємо графік
    fig, ax = plt.subplots(figsize=(14, 8))

    # Малюємо стовпці для обох категорій
    ax.bar(range(len(monthly_orders)), monthly_orders[0], width=0.8,
           color='blue', alpha=0.7, label='Неуспішні')
    ax.bar(range(len(monthly_orders)), monthly_orders[1], width=0.8,
           bottom=monthly_orders[0], color='orange', alpha=0.7, label='Успішні')

    # Робимо красиві мітки на осі X з відображенням років
    plt.xticks(range(len(monthly_orders)), monthly_orders.index, rotation=45, ha='right')

    # Відображаємо тільки кожну N-ту мітку, щоб уникнути перекриття
    n = max(1, len(monthly_orders) // 12)  # Відображаємо приблизно 12 міток
    for i, label in enumerate(ax.get_xticklabels()):
        if i % n != 0:
            label.set_visible(False)

    # Форматуємо мітки на осі Y для відображення тисяч
    import matplotlib.ticker as ticker
    formatter = ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    ax.yaxis.set_major_formatter(formatter)

    # Додаємо заголовок і підписи до осей
    plt.title('Кількість замовлень по місяцях', fontsize=14)
    plt.xlabel('Рік-Місяць', fontsize=12)
    plt.ylabel('Кількість замовлень', fontsize=12)

    # Додаємо сітку для кращого сприйняття
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.grid(axis='x', alpha=0.3)

    # Додаємо легенду
    plt.legend(fontsize=10)

    # Додаємо ефекти для виділення тренду
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{column_dir}/monthly_orders.png", dpi=300)
    plt.close()

    # 2. Графік відсотка успішних замовлень по місяцях
    plt.figure(figsize=(14, 8))

    # Обчислюємо відсоток успішних замовлень
    success_rate = (monthly_orders[1] / monthly_orders['total'] * 100).fillna(0)

    # Створюємо датафрейм для щоденних даних
    df_daily_success = df.copy()
    # Додаємо фільтр, щоб виключити 2025 рік
    df_daily_success = df_daily_success[df_daily_success[date_column].dt.year < 2025]
    df_daily_success['date'] = df_daily_success[date_column].dt.date

    # Групуємо щоденні дані
    daily_success = df_daily_success.groupby(['date', 'is_successful']).size().unstack(fill_value=0)
    if 1 not in daily_success.columns:
        daily_success[1] = 0
    if 0 not in daily_success.columns:
        daily_success[0] = 0

    daily_success['total'] = daily_success[0] + daily_success[1]
    daily_success['success_rate'] = (daily_success[1] / daily_success['total'] * 100).fillna(0)

    # Створюємо графік
    fig, ax = plt.subplots(figsize=(14, 8))

    # Малюємо щоденні дані напівпрозорими точками
    ax.scatter(daily_success.index, daily_success['success_rate'],
               color='blue', alpha=0.1, s=10, label='Щоденні дані')

    # Підготовка дат для місячних даних
    month_dates = []
    # Конвертуємо рядки 'YYYY-MM' в об'єкти дати (беремо середину місяця)
    for ym in success_rate.index:
        year, month = map(int, ym.split('-'))
        month_dates.append(pd.Timestamp(year=year, month=month, day=15))

    # Малюємо місячну агрегацію як окремі точки з лінією
    ax.plot(month_dates, success_rate,
            'o-', color='crimson', linewidth=2, markersize=6, label='Місячна агрегація')

    # Форматуємо дати на осі X
    import matplotlib.dates as mdates
    years = mdates.YearLocator()   # кожен рік
    months = mdates.MonthLocator()  # кожен місяць
    years_fmt = mdates.DateFormatter('%Y')

    # Форматування осі X
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    # Встановлюємо діапазон осі Y від 0 до 100 відсотків
    ax.set_ylim(0, 100)

    # Форматуємо мітки на осі Y для відображення відсотків
    formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.0f}%')
    ax.yaxis.set_major_formatter(formatter)

    # Додаємо середній рівень успішності
    mean_rate = daily_success['success_rate'].mean()
    ax.axhline(y=mean_rate, color='blue', linestyle='--', alpha=0.7,
               label=f'Середній % успіху: {mean_rate:.1f}%')

    # Додаємо заголовок і підписи до осей
    plt.title('Відсоток успішних замовлень по місяцях', fontsize=14)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Відсоток успішних замовлень', fontsize=12)

    # Додаємо сітку для кращого сприйняття
    plt.grid(True, linestyle='--', alpha=0.7)

    # Додаємо легенду
    plt.legend(fontsize=10, loc='upper right')

    # Додаємо ефекти для виділення тренду
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{column_dir}/monthly_success_rate.png", dpi=300)
    plt.close()

    # Також додаємо стовпчиковий графік кількості замовлень по рокам
    plt.figure(figsize=(12, 8))

    # Додаємо стовпець для року
    df_yearly = df.copy()
    # Додаємо фільтр, щоб виключити 2025 рік
    df_yearly = df_yearly[df_yearly[date_column].dt.year < 2025]
    df_yearly['year'] = df_yearly[date_column].dt.year

    # Групуємо дані по роках та успішності
    yearly_orders = df_yearly.groupby(['year', 'is_successful']).size().unstack(fill_value=0)

    if 1 not in yearly_orders.columns:
        yearly_orders[1] = 0
    if 0 not in yearly_orders.columns:
        yearly_orders[0] = 0

    # Обчислюємо загальну кількість та відсоток
    yearly_orders['total'] = yearly_orders[0] + yearly_orders[1]
    yearly_orders['success_rate'] = (yearly_orders[1] / yearly_orders['total'] * 100).fillna(0)

    # Рисуємо стовпчиковий графік по рокам
    x = np.arange(len(yearly_orders.index))
    width = 0.8

    fig, ax = plt.subplots(figsize=(12, 8))

    # Рисуємо стовпці для неуспішних і успішних замовлень
    p1 = ax.bar(x, yearly_orders[0], width, label='Неуспішні', color='blue', alpha=0.7)
    p2 = ax.bar(x, yearly_orders[1], width, bottom=yearly_orders[0],
                label='Успішні', color='orange', alpha=0.7)

    # Додаємо підписи та налаштування
    ax.set_xlabel('Рік', fontsize=12)
    ax.set_ylabel('Кількість замовлень', fontsize=12)
    ax.set_title('Розподіл замовлень по роках', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(yearly_orders.index)

    # Форматуємо числа з розділювачами тисяч
    formatter = ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    ax.yaxis.set_major_formatter(formatter)

    # Додаємо сітку та легенду
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    # Додаємо підписи з відсотком успішності на стовпці
    for i, year in enumerate(yearly_orders.index):
        total = yearly_orders.loc[year, 'total']
        rate = yearly_orders.loc[year, 'success_rate']
        ax.text(i, yearly_orders.loc[year, 'total'] + 0.02 * ax.get_ylim()[1],
                f'{rate:.1f}%',
                ha='center', va='bottom',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    plt.tight_layout()
    plt.savefig(f"{column_dir}/yearly_orders.png", dpi=300)
    plt.close()

    # 3. Теплова карта кількості замовлень по годинам і дням тижня
    plt.figure(figsize=(12, 8))

    # Створюємо новий датафрейм з годинами та днями тижня
    df_heatmap = df.copy()
    # Додаємо фільтр, щоб виключити 2025 рік
    df_heatmap = df_heatmap[df_heatmap[date_column].dt.year < 2025]
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
    # Додаємо фільтр, щоб виключити 2025 рік
    df_monthly = df_monthly[df_monthly[date_column].dt.year < 2025]
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

    # Встановлюємо шкалу від 0 до 100% для осі Y2
    ax2.set_ylim(0, 100)

    # Форматуємо мітки на осі Y для відображення відсотків
    formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.0f}%')
    ax2.yaxis.set_major_formatter(formatter)

    # Додаємо заголовок і підписи до осей
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
        # Додаємо фільтр, щоб виключити 2025 рік
        df_amount = df_amount[df_amount[date_column].dt.year < 2025]
        df_amount['date'] = df_amount[date_column].dt.date
        df_amount['year_month'] = df_amount[date_column].dt.strftime('%Y-%m')

        # Групуємо дані помісячно (замість щоденно) для кращої візуалізації трендів
        monthly_amount = df_amount.groupby(['year_month', 'is_successful'])['order_amount'].mean().unstack(fill_value=0)

        if 1 not in monthly_amount.columns:
            monthly_amount[1] = 0
        if 0 not in monthly_amount.columns:
            monthly_amount[0] = 0

        # Сортуємо за датою
        monthly_amount = monthly_amount.sort_index()

        # Створюємо графік
        fig, ax = plt.subplots(figsize=(14, 8))

        # Малюємо лінії для кожної категорії
        ax.plot(range(len(monthly_amount)), monthly_amount[0], 'o-', color='blue', linewidth=2, label='Неуспішні')
        ax.plot(range(len(monthly_amount)), monthly_amount[1], 'o-', color='orange', linewidth=2, label='Успішні')

        # Робимо красиві мітки на осі X з відображенням років
        plt.xticks(range(len(monthly_amount)), monthly_amount.index, rotation=45, ha='right')

        # Відображаємо тільки кожну N-ту мітку, щоб уникнути перекриття
        n = max(1, len(monthly_amount) // 12)  # Відображаємо приблизно 12 міток
        for i, label in enumerate(ax.get_xticklabels()):
            if i % n != 0:
                label.set_visible(False)

        # Форматуємо мітки на осі Y для відображення тисяч
        import matplotlib.ticker as ticker
        formatter = ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
        ax.yaxis.set_major_formatter(formatter)

        # Додаємо заголовок і підписи до осей
        plt.title('Середня вартість замовлень по місяцях', fontsize=14)
        plt.xlabel('Рік-Місяць', fontsize=12)
        plt.ylabel('Середня вартість замовлення, грн', fontsize=12)

        # Додаємо сітку для кращого сприйняття
        plt.grid(True, linestyle='--', alpha=0.7)

        # Додаємо легенду
        plt.legend(fontsize=10)

        # Додаємо ефекти для виділення тренду
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Видаляємо підписи для важливих точок
        # Нічого не додаємо замість цього

        plt.tight_layout()
        plt.savefig(f"{column_dir}/avg_amount_trend.png", dpi=300)
        plt.close()

        # Додатковий графік - порівняння розподілу вартості по роках
        plt.figure(figsize=(12, 8))

        # Додаємо стовпець для року
        df_amount['year'] = df_amount[date_column].dt.year

        # Групуємо дані по роках для статистики
        yearly_stats = df_amount.groupby(['year', 'is_successful'])['order_amount'].agg(['mean', 'median']).unstack()

        # Рисуємо графік з групованими стовпцями для середніх значень
        means = yearly_stats['mean']
        x = np.arange(len(means.index))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        # Рисуємо стовпці для неуспішних і успішних замовлень
        if 0 in means.columns:
            rects1 = ax.bar(x - width/2, means[0], width, label='Неуспішні', color='blue', alpha=0.7)
        else:
            rects1 = ax.bar(x - width/2, [0] * len(means.index), width, label='Неуспішні', color='blue', alpha=0.7)

        if 1 in means.columns:
            rects2 = ax.bar(x + width/2, means[1], width, label='Успішні', color='orange', alpha=0.7)
        else:
            rects2 = ax.bar(x + width/2, [0] * len(means.index), width, label='Успішні', color='orange', alpha=0.7)

        # Додаємо підписи, назву і легенду
        ax.set_xlabel('Рік', fontsize=12)
        ax.set_ylabel('Середня вартість замовлення, грн', fontsize=12)
        ax.set_title('Порівняння середньої вартості замовлень по роках')
        ax.set_xticks(x)
        ax.set_xticklabels(means.index)

        # Форматуємо числа з розділювачами тисяч
        formatter = ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
        ax.yaxis.set_major_formatter(formatter)

        # Додаємо сітку та легенду
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

        # Додаємо значення над стовпцями
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate('{:,.0f}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 пікселі вертикальний зсув
                                textcoords="offset points",
                                ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.savefig(f"{column_dir}/yearly_avg_amount.png", dpi=300)
        plt.close()

    # Додаємо графік щоденної кількості замовлень по місяцях для порівняння з оригінальним
    plt.figure(figsize=(14, 8))

    # Створюємо новий датафрейм з щоденними даними
    df_daily = df.copy()
    # Додаємо фільтр, щоб виключити 2025 рік
    df_daily = df_daily[df_daily[date_column].dt.year < 2025]
    df_daily['date'] = df_daily[date_column].dt.date
    df_daily['year_month'] = df_daily[date_column].dt.strftime('%Y-%m')

    # Групуємо дані по днях
    daily_orders = df_daily.groupby(['date', 'is_successful']).size().unstack(fill_value=0)

    if 1 not in daily_orders.columns:
        daily_orders[1] = 0
    if 0 not in daily_orders.columns:
        daily_orders[0] = 0

    # Обчислюємо загальну кількість замовлень
    daily_orders['total'] = daily_orders[0] + daily_orders[1]

    # Сортуємо за датою
    daily_orders = daily_orders.sort_index()

    # Створюємо графік
    fig, ax = plt.subplots(figsize=(14, 8))

    # Визначаємо ширину вікна для ковзного середнього
    window = 30

    # Прибираємо заповнені області і залишаємо тільки лінії тренду
    if len(daily_orders) > window:
        # Малюємо лінії тренду (ковзне середнє за 30 днів)
        ax.plot(daily_orders.index, daily_orders[0].rolling(window=window).mean(),
                color='blue', linewidth=3, label='Тренд неуспішних')
        ax.plot(daily_orders.index, daily_orders[1].rolling(window=window).mean(),
                color='orange', linewidth=3, label='Тренд успішних')
        ax.plot(daily_orders.index, daily_orders['total'].rolling(window=window).mean(),
                color='green', linewidth=3, label='Тренд загальних')

    # Форматуємо дати на осі X
    # Створюємо локатор для позначок по роках
    import matplotlib.dates as mdates
    years = mdates.YearLocator()   # кожен рік
    months = mdates.MonthLocator()  # кожен місяць
    years_fmt = mdates.DateFormatter('%Y')

    # Форматування осі X
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    # Форматуємо мітки на осі Y для відображення тисяч
    import matplotlib.ticker as ticker
    formatter = ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    ax.yaxis.set_major_formatter(formatter)

    # Регулюємо масштаб осі Y для кращого відображення тренду
    # Визначаємо максимальне значення тренду з невеликим відступом вгору
    max_trend = max(
        daily_orders[0].rolling(window=window).mean().max(),
        daily_orders[1].rolling(window=window).mean().max(),
        daily_orders['total'].rolling(window=window).mean().max()
    )

    # Встановлюємо межі по Y від 0 до максимального значення з відступом 10%
    ax.set_ylim(0, max_trend * 1.1)

    # Знаходимо максимальні значення трендів
    max_total_idx = daily_orders['total'].rolling(window=window).mean().idxmax()
    max_total_val = daily_orders['total'].rolling(window=window).mean().max()

    # Підписуємо максимальне значення
    if not pd.isna(max_total_val):
        ax.annotate(f'Макс: {max_total_val:.0f}',
                    xy=(max_total_idx, max_total_val),
                    xytext=(max_total_idx, max_total_val + 0.05 * ax.get_ylim()[1]),
                    ha='center', va='bottom',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

    # Налаштування графіка
    plt.title('Тренди кількості замовлень по днях', fontsize=14)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Кількість замовлень', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Додаємо легенду
    plt.legend(fontsize=10, loc='upper right')

    # Додаємо ефекти для виділення тренду
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{column_dir}/daily_orders.png", dpi=300)
    plt.close()

    # Додаємо новий графік, що показує і щоденні дані з ковзним середнім, і місячну агрегацію
    # для відсотка успішних замовлень
    plt.figure(figsize=(14, 8))

    # Створюємо новий датафрейм з щоденними даними
    df_daily_rate = df.copy()
    # Додаємо фільтр, щоб виключити 2025 рік
    df_daily_rate = df_daily_rate[df_daily_rate[date_column].dt.year < 2025]
    df_daily_rate['date'] = df_daily_rate[date_column].dt.date
    df_daily_rate['year_month'] = df_daily_rate[date_column].dt.strftime('%Y-%m')

    # Групуємо дані по днях для щоденного відсотка успішності
    daily_counts = df_daily_rate.groupby(['date', 'is_successful']).size().unstack(fill_value=0)

    if 1 not in daily_counts.columns:
        daily_counts[1] = 0
    if 0 not in daily_counts.columns:
        daily_counts[0] = 0

    # Обчислюємо щоденний відсоток успішних замовлень
    daily_counts['total'] = daily_counts[0] + daily_counts[1]
    daily_counts['success_rate'] = (daily_counts[1] / daily_counts['total'] * 100).fillna(0)

    # Сортуємо за датою
    daily_counts = daily_counts.sort_index()

    # Обчислюємо ковзне середнє за 30 днів
    window = 30
    rolling_avg = daily_counts['success_rate'].rolling(window=window).mean()

    # Групуємо по місяцях для місячного відсотка успішності
    monthly_data = df_daily_rate.groupby(['year_month', 'is_successful']).size().unstack(fill_value=0)

    if 1 not in monthly_data.columns:
        monthly_data[1] = 0
    if 0 not in monthly_data.columns:
        monthly_data[0] = 0

    monthly_data['total'] = monthly_data[0] + monthly_data[1]
    monthly_data['success_rate'] = (monthly_data[1] / monthly_data['total'] * 100).fillna(0)

    # Створюємо графік
    fig, ax = plt.subplots(figsize=(14, 8))

    # Малюємо щоденні дані напівпрозорими точками
    ax.scatter(daily_counts.index, daily_counts['success_rate'],
               color='blue', alpha=0.1, s=10, label='Щоденні дані')

    # Малюємо ковзне середнє як лінію
    ax.plot(daily_counts.index, rolling_avg,
            color='red', linewidth=2.5, label=f'Ковзне середнє ({window} днів)')

    # Підготовка дат для місячних даних
    month_dates = []
    # Конвертуємо рядки 'YYYY-MM' в об'єкти дати (беремо середину місяця)
    for ym in monthly_data.index:
        year, month = map(int, ym.split('-'))
        month_dates.append(pd.Timestamp(year=year, month=month, day=15))

    # Малюємо місячну агрегацію як окремі точки з лінією
    ax.plot(month_dates, monthly_data['success_rate'],
            'o-', color='green', linewidth=2, markersize=8, label='Місячна агрегація')

    # Форматуємо дати на осі X
    # Створюємо локатор для позначок по роках
    import matplotlib.dates as mdates
    years = mdates.YearLocator()   # кожен рік
    months = mdates.MonthLocator()  # кожен місяць
    years_fmt = mdates.DateFormatter('%Y')

    # Форматування осі X
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    # Форматуємо мітки на осі Y для відображення відсотків
    formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.0f}%')
    ax.yaxis.set_major_formatter(formatter)

    # Встановлюємо діапазон осі Y від 0 до 100 відсотків
    ax.set_ylim(0, 100)

    # Додаємо середній рівень успішності
    mean_rate = daily_counts['success_rate'].mean()
    ax.axhline(y=mean_rate, color='gray', linestyle='--', alpha=0.7,
               label=f'Середній % успіху: {mean_rate:.1f}%')

    # Налаштування графіка
    plt.title('Порівняння методів агрегації відсотка успішних замовлень', fontsize=14)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Відсоток успішних замовлень', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Додаємо легенду
    plt.legend(fontsize=10, loc='lower right')

    # Додаємо ефекти для виділення тренду
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{column_dir}/success_rate_comparison.png", dpi=300)
    plt.close()

    # Додаємо новий графік, що показує тільки ковзне середнє для відсотка успішних замовлень
    plt.figure(figsize=(14, 8))

    # Створюємо новий датафрейм з щоденними даними
    df_daily_rate = df.copy()
    # Додаємо фільтр, щоб виключити 2025 рік
    df_daily_rate = df_daily_rate[df_daily_rate[date_column].dt.year < 2025]
    df_daily_rate['date'] = df_daily_rate[date_column].dt.date

    # Групуємо дані по днях для щоденного відсотка успішності
    daily_counts = df_daily_rate.groupby(['date', 'is_successful']).size().unstack(fill_value=0)

    if 1 not in daily_counts.columns:
        daily_counts[1] = 0
    if 0 not in daily_counts.columns:
        daily_counts[0] = 0

    # Обчислюємо щоденний відсоток успішних замовлень
    daily_counts['total'] = daily_counts[0] + daily_counts[1]
    daily_counts['success_rate'] = (daily_counts[1] / daily_counts['total'] * 100).fillna(0)

    # Сортуємо за датою
    daily_counts = daily_counts.sort_index()

    # Обчислюємо ковзне середнє за 30 днів
    window = 30
    rolling_avg = daily_counts['success_rate'].rolling(window=window).mean()

    # Створюємо графік
    fig, ax = plt.subplots(figsize=(14, 8))

    # Малюємо щоденні дані напівпрозорими точками
    ax.scatter(daily_counts.index, daily_counts['success_rate'],
               color='blue', alpha=0.1, s=10, label='Щоденні дані')

    # Малюємо ковзне середнє як лінію
    ax.plot(daily_counts.index, rolling_avg,
            color='red', linewidth=2.5, label=f'Ковзне середнє ({window} днів)')

    # Форматуємо дати на осі X
    import matplotlib.dates as mdates
    years = mdates.YearLocator()   # кожен рік
    months = mdates.MonthLocator()  # кожен місяць
    years_fmt = mdates.DateFormatter('%Y')

    # Форматування осі X
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    # Форматуємо мітки на осі Y для відображення відсотків
    formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.0f}%')
    ax.yaxis.set_major_formatter(formatter)

    # Встановлюємо діапазон осі Y від 0 до 100 відсотків
    ax.set_ylim(0, 100)

    # Додаємо середній рівень успішності
    mean_rate = daily_counts['success_rate'].mean()
    ax.axhline(y=mean_rate, color='gray', linestyle='--', alpha=0.7,
               label=f'Середній % успіху: {mean_rate:.1f}%')

    # Знаходимо максимальні і мінімальні значення ковзного середнього
    max_idx = rolling_avg.idxmax()
    min_idx = rolling_avg.idxmin()
    max_val = rolling_avg.max()
    min_val = rolling_avg.min()

    # Підписуємо максимальне і мінімальне значення
    if not pd.isna(max_val):
        ax.annotate(f'Макс: {max_val:.1f}%',
                    xy=(max_idx, max_val),
                    xytext=(0, 10),  # зсув тексту від точки
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

    if not pd.isna(min_val):
        ax.annotate(f'Мін: {min_val:.1f}%',
                    xy=(min_idx, min_val),
                    xytext=(0, -10),  # зсув тексту від точки
                    textcoords="offset points",
                    ha='center', va='top',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

    # Налаштування графіка
    plt.title('Тренд відсотка успішних замовлень (30-денне ковзне середнє)', fontsize=14)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Відсоток успішних замовлень', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Додаємо легенду
    plt.legend(fontsize=10, loc='lower right')

    # Додаємо ефекти для виділення тренду
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{column_dir}/success_rate_trend.png", dpi=300)
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
