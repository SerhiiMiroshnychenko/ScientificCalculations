#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Комплексний скрипт для аналізу даних: виявлення викидів та регресійний аналіз.

Скрипт дозволяє:
1. Зчитувати два стовпці даних з CSV файлу
2. Автоматично виявляти та видаляти грубі промахи (викиди)
3. Будувати лінійну регресійну модель на очищених даних
4. Оцінювати якість моделі через статистичні метрики
5. Візуалізувати результати через різноманітні графіки
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys

def print_header(text):
    """Функція виведення заголовка секції"""
    print("\n" + "="*60)
    print(text)
    print("="*60)

def read_csv_data(file_path, column_x, column_y, delimiter=','):
    """
    Зчитує два стовпці даних з CSV файлу

    Параметри:
    file_path (str): Шлях до CSV файлу
    column_x (str або int): Назва або індекс стовпця для X
    column_y (str або int): Назва або індекс стовпця для Y
    delimiter (str): Розділювач у CSV файлі

    Повертає:
    tuple: (x_data, y_data, df, x_name, y_name) - дані X та Y, датафрейм та назви стовпців
    """
    try:
        # Зчитування CSV файлу
        df = pd.read_csv(file_path, delimiter=delimiter)

        # Отримання даних з першого стовпця (X)
        if isinstance(column_x, int):
            if column_x < len(df.columns):
                x_data = df.iloc[:, column_x].values
                x_name = df.columns[column_x]
            else:
                raise ValueError(f"Стовпець з індексом {column_x} не існує в файлі. Доступні індекси: 0-{len(df.columns)-1}")
        else:
            if column_x in df.columns:
                x_data = df[column_x].values
                x_name = column_x
            else:
                raise ValueError(f"Стовпець '{column_x}' не знайдено в файлі. Доступні стовпці: {', '.join(df.columns)}")

        # Отримання даних з другого стовпця (Y)
        if isinstance(column_y, int):
            if column_y < len(df.columns):
                y_data = df.iloc[:, column_y].values
                y_name = df.columns[column_y]
            else:
                raise ValueError(f"Стовпець з індексом {column_y} не існує в файлі. Доступні індекси: 0-{len(df.columns)-1}")
        else:
            if column_y in df.columns:
                y_data = df[column_y].values
                y_name = column_y
            else:
                raise ValueError(f"Стовпець '{column_y}' не знайдено в файлі. Доступні стовпці: {', '.join(df.columns)}")

        print(f"Зчитано дані зі стовпців '{x_name}' (X) та '{y_name}' (Y), кількість значень: {len(x_data)}")

        # Перевірка, чи дані числові
        try:
            x_data = x_data.astype(float)
            y_data = y_data.astype(float)
        except ValueError:
            raise ValueError(f"Стовпці містять нечислові дані")

        return x_data, y_data, df, x_name, y_name

    except FileNotFoundError:
        raise FileNotFoundError(f"Файл '{file_path}' не знайдено")
    except Exception as e:
        raise Exception(f"Помилка при зчитуванні даних: {str(e)}")

def calculate_statistics(data):
    """
    Розрахунок статистичних характеристик для серії даних

    Параметри:
    data (numpy.ndarray): Масив даних

    Повертає:
    tuple: (n, x_mean, D, sigma) - кількість елементів, середнє, дисперсія, с.к.в.
    """
    n = len(data)
    x_mean = np.mean(data)  # середнє значення
    D = np.sum((data - x_mean)**2) / (n - 1)  # дисперсія
    sigma = np.sqrt(D)  # середньоквадратичне відхилення

    return n, x_mean, D, sigma

def calculate_gamma(data, x_mean, sigma):
    """
    Розрахунок коефіцієнтів γ1 та γ2 для виявлення викидів

    Параметри:
    data (numpy.ndarray): Масив даних
    x_mean (float): Середнє значення
    sigma (float): Середньоквадратичне відхилення

    Повертає:
    tuple: (gamma1, gamma2) - коефіцієнти для аналізу
    """
    gamma1 = (np.max(data) - x_mean) / sigma
    gamma2 = (x_mean - np.min(data)) / sigma

    return gamma1, gamma2

def get_critical_gamma(n, confidence=0.95):
    """
    Розрахунок критичного значення gamma_p для виявлення викидів
    з використанням розподілу Стьюдента

    Параметри:
    n (int): Кількість спостережень
    confidence (float): Рівень довіри

    Повертає:
    float: Критичне значення gamma_p
    """
    # Розрахунок критичного значення за формулою, що використовує
    # розподіл Стьюдента з (n-2) ступенями свободи
    t_critical = stats.t.ppf(1 - (1 - confidence) / (2 * n), n - 2)
    gamma_p = t_critical * (n - 1) / np.sqrt(n * (n - 2) + t_critical**2)

    return gamma_p

def check_outliers(data, label, confidence=0.95):
    """
    Перевірка наявності викидів у серії даних

    Параметри:
    data (numpy.ndarray): Масив даних
    label (str): Назва стовпця для виведення
    confidence (float): Рівень довіри

    Повертає:
    tuple: (has_outliers, outlier_indices, stats) - чи є викиди, їх індекси, статистики
    """
    n, x_mean, D, sigma = calculate_statistics(data)
    gamma1, gamma2 = calculate_gamma(data, x_mean, sigma)

    # Розрахунок критичного значення gamma_p
    gamma_p = get_critical_gamma(n, confidence)

    has_outliers = False
    outlier_indices = []

    # print(f"{label}: γ1 = {gamma1:.3f}, γ2 = {gamma2:.3f}, γp = {gamma_p:.3f}")

    if gamma1 > gamma_p:
        has_outliers = True
        max_index = np.argmax(data)
        outlier_indices.append(max_index)
    #     print(f"{label}: γ1 > γp: Викид у максимальному значенні {data[max_index]:.3f} (індекс {max_index})")
    # else:
    #     print(f"{label}: γ1 <= γp: Немає викиду у максимальному значенні")

    if gamma2 > gamma_p:
        has_outliers = True
        min_index = np.argmin(data)
        outlier_indices.append(min_index)
    #     print(f"{label}: γ2 > γp: Викид у мінімальному значенні {data[min_index]:.3f} (індекс {min_index})")
    # else:
    #     print(f"{label}: γ2 <= γp: Немає викиду у мінімальному значенні")

    return has_outliers, outlier_indices, (n, x_mean, D, sigma)

def plot_histograms(data_before, data_after, column_name):
    """
    Візуалізація гістограм до та після очищення

    Параметри:
    data_before (numpy.ndarray): Дані до очищення
    data_after (numpy.ndarray): Дані після очищення
    column_name (str): Назва стовпця для заголовків
    """
    # Налаштування загального стилю графіків
    plt.style.use('ggplot')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'axes.grid': True,
        'axes.grid.which': 'both',
        'grid.alpha': 0.3,
        'figure.figsize': (14, 8),
        'figure.dpi': 120
    })

    # Додаткові налаштування для гістограм
    bin_params = {}
    if len(data_before) > 1000:
        # Автоматичний розрахунок оптимального числа бінів
        # Використовуємо правило Freedman-Diaconis
        data_range = np.max(data_before) - np.min(data_before)
        bin_width = 2 * stats.iqr(data_before) / (len(data_before) ** (1/3))
        n_bins = int(data_range / bin_width) if bin_width > 0 else 50
        n_bins = min(100, max(20, n_bins))  # Обмежуємо кількість бінів
        bin_params['bins'] = n_bins

    # Створюємо фігуру
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Визначення статистик для обох наборів даних
    mean_before = np.mean(data_before)
    std_before = np.std(data_before)
    mean_after = np.mean(data_after)
    std_after = np.std(data_after)

    # Добавляємо легенду з інформацією про початкові та очищені дані
    before_label = f'Початкові дані: $\\mu={mean_before:.2f}$, $\\sigma={std_before:.2f}$'
    after_label = f'Очищені дані: $\\mu={mean_after:.2f}$, $\\sigma={std_after:.2f}$'

    # Налаштування прозорості для кращого розрізнення
    alpha_val = 0.6

    # Накладені гістограми
    sns.histplot(data_before, kde=True, color='blue', alpha=alpha_val, label=before_label,
                 ax=ax, edgecolor='darkblue', linewidth=1.2, **bin_params)
    sns.histplot(data_after, kde=True, color='green', alpha=alpha_val, label=after_label,
                 ax=ax, edgecolor='darkgreen', linewidth=1.2, **bin_params)

    # Додаємо вертикальні лінії для середніх значень
    ax.axvline(mean_before, color='blue', linestyle='--', linewidth=2, alpha=0.9)
    ax.axvline(mean_after, color='green', linestyle='--', linewidth=2, alpha=0.9)

    # Оформлення графіка
    ax.set_title(f'{column_name} (порівняння гістограм)', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel(column_name, fontsize=18, fontweight='bold')
    ax.set_ylabel('Частота', fontsize=18, fontweight='bold')

    # Покращення легенди
    legend = ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=14)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)

    # Покращення відображення сітки
    ax.grid(True, linestyle='--', alpha=0.7)

    # Покращення відображення меж графіка
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    # Встановлюємо межі по x для кращого відображення
    # Визначаємо межі на основі перцентилів даних
    data_all = np.concatenate([data_before, data_after])
    q_low, q_high = np.percentile(data_all, [1, 99])
    # Розширюємо межі на 10% в обидві сторони
    range_x = q_high - q_low
    ax.set_xlim([q_low - 0.1 * range_x, q_high + 0.1 * range_x])

    plt.tight_layout()
    plt.show()

def plot_scatter_before_after(x_before, y_before, x_after, y_after, x_name, y_name):
    """
    Візуалізація діаграм розсіювання до та після очищення

    Параметри:
    x_before, y_before: Дані X і Y до очищення
    x_after, y_after: Дані X і Y після очищення
    x_name, y_name: Назви стовпців для заголовків
    """
    # Накладені діаграми розсіювання
    plt.figure(figsize=(10, 6))
    plt.scatter(x_before, y_before, alpha=0.5, color='blue', label='Початкові дані')
    plt.scatter(x_after, y_after, alpha=0.5, color='green', label='Очищені дані')
    plt.title('Порівняння даних до та після видалення викидів')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Діаграма викидів (точки, які були видалені)
    if len(x_before) != len(x_after):
        # Створюємо маску викидів
        outlier_mask = np.ones(len(x_before), dtype=bool)
        for i, x_val in enumerate(x_before):
            if x_val in x_after:
                # Перевіряємо також відповідне значення Y
                idx = np.where(x_after == x_val)[0]
                if len(idx) > 0 and y_before[i] in y_after[idx]:
                    outlier_mask[i] = False

        # Отримуємо тільки викиди
        x_outliers = x_before[outlier_mask]
        y_outliers = y_before[outlier_mask]

        if len(x_outliers) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(x_before, y_before, alpha=0.3, color='blue', label='Всі дані')
            plt.scatter(x_outliers, y_outliers, alpha=0.7, color='red', label='Викиди')
            plt.title('Виявлені викиди')
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

def remove_outliers_from_data(x_data, y_data, x_name, y_name, confidence=0.95, max_iterations=3):
    """
    Виявлення та видалення викидів з двох стовпців даних

    Параметри:
    x_data, y_data: Масиви даних X і Y
    x_name, y_name: Назви стовпців
    confidence: Рівень довіри
    max_iterations: Максимальна кількість ітерацій очищення

    Повертає:
    tuple: (x_clean, y_clean, outlier_indices) - очищені дані та індекси викидів
    """
    print_header("Аналіз та виявлення викидів")

    # Копіювання оригінальних даних
    x_clean = x_data.copy()
    y_clean = y_data.copy()

    # Створення маски для відстеження індексів (початково всі True)
    valid_indices = np.ones(len(x_data), dtype=bool)

    # Інформація про дані до очищення
    n_x, mean_x, var_x, std_x = calculate_statistics(x_data)
    n_y, mean_y, var_y, std_y = calculate_statistics(y_data)

    print(f"Початкові статистики X: n={n_x}, середнє={mean_x:.4f}, дисперсія={var_x:.4f}, с.к.в.={std_x:.4f}")
    print(f"Початкові статистики Y: n={n_y}, середнє={mean_y:.4f}, дисперсія={var_y:.4f}, с.к.в.={std_y:.4f}")

    # Створення таблиці для відстеження викидів
    outliers_data = {"Ітерація": [], "Змінна": [], "Індекс": [], "Значення": []}

    iteration = 0
    total_outliers = 0

    while iteration < max_iterations:
        iteration += 1
        # print(f"\nІтерація {iteration}:")

        # Перевірка наявності викидів у X та Y
        x_has_outliers, x_outlier_indices, x_stats = check_outliers(x_clean, f"X ({x_name})", confidence)
        y_has_outliers, y_outlier_indices, y_stats = check_outliers(y_clean, f"Y ({y_name})", confidence)

        # Якщо немає викидів в обох стовпцях, завершуємо цикл
        if not x_has_outliers and not y_has_outliers:
            print("Викиди не виявлені в обох стовпцях, завершуємо очищення")
            break

        # Збір індексів викидів
        current_outliers = set()

        # Додаємо викиди X
        for idx in x_outlier_indices:
            # Знаходимо оригінальний індекс
            orig_idx = np.where(valid_indices)[0][idx]
            current_outliers.add(orig_idx)
            outliers_data["Ітерація"].append(iteration)
            outliers_data["Змінна"].append(x_name)
            outliers_data["Індекс"].append(orig_idx)
            outliers_data["Значення"].append(x_data[orig_idx])

        # Додаємо викиди Y
        for idx in y_outlier_indices:
            # Знаходимо оригінальний індекс
            orig_idx = np.where(valid_indices)[0][idx]
            current_outliers.add(orig_idx)
            outliers_data["Ітерація"].append(iteration)
            outliers_data["Змінна"].append(y_name)
            outliers_data["Індекс"].append(orig_idx)
            outliers_data["Значення"].append(y_data[orig_idx])

        # Оновлюємо маску дійсних індексів
        for idx in current_outliers:
            valid_indices[idx] = False

        # Оновлюємо очищені дані
        x_clean = x_data[valid_indices]
        y_clean = y_data[valid_indices]

        total_outliers += len(current_outliers)
        # print(f"Знайдено {len(current_outliers)} викидів в ітерації {iteration}")
        # print(f"Загальна кількість видалених викидів: {total_outliers}")

        if len(current_outliers) == 0:
            print("Викиди не знайдені, завершуємо очищення")
            break

    # Підсумкові статистики після очищення
    if total_outliers > 0:
        n_x, mean_x, var_x, std_x = calculate_statistics(x_clean)
        n_y, mean_y, var_y, std_y = calculate_statistics(y_clean)

        print("\nСтатистики після очищення:")
        print(f"X: n={n_x}, середнє={mean_x:.4f}, дисперсія={var_x:.4f}, с.к.в.={std_x:.4f}")
        print(f"Y: n={n_y}, середнє={mean_y:.4f}, дисперсія={var_y:.4f}, с.к.в.={std_y:.4f}")
        print(f"Видалено {total_outliers} викидів, залишилось {len(x_clean)} спостережень")

        # Створення DataFrame з даними про викиди
        outliers_df = pd.DataFrame(outliers_data)
        if not outliers_df.empty:
            print("\nВидалені викиди:")
            print(outliers_df)
    else:
        print("\nВикиди не виявлені, дані залишаються без змін")

    # Повертаємо очищені дані та індекси видалених спостережень
    outlier_indices = np.where(~valid_indices)[0]
    return x_clean, y_clean, outlier_indices

def calculate_regression_parameters(x, y):
    """
    Розраховує параметри лінійної регресії методом найменших квадратів.

    Параметри:
    x (numpy.ndarray): Масив незалежних змінних
    y (numpy.ndarray): Масив залежних змінних

    Повертає:
    tuple: (a0, a1) - коефіцієнти лінійної регресії y = a0 + a1*x
    """
    n = len(x)

    # Обчислення середніх значень
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Обчислення середнього значення квадратів x
    x2_mean = np.mean(x**2)

    # Обчислення середнього значення добутку x*y
    xy_mean = np.mean(x*y)

    # Розрахунок коефіцієнтів регресії
    a1 = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean**2)
    a0 = y_mean - a1 * x_mean

    return a0, a1

def calculate_regression_metrics(x, y, a0, a1):
    """
    Розраховує метрики якості лінійної регресійної моделі.

    Параметри:
    x (numpy.ndarray): Масив незалежних змінних
    y (numpy.ndarray): Масив залежних змінних
    a0 (float): Вільний член рівняння регресії
    a1 (float): Коефіцієнт нахилу

    Повертає:
    dict: Словник з метриками якості моделі
    """
    n = len(x)

    # Прогнозовані значення
    y_pred = a0 + a1 * x

    # Середні значення
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Залишки (похибки)
    residuals = y - y_pred

    # Сума квадратів залишків
    sse = np.sum(residuals**2)

    # Загальна сума квадратів
    sst = np.sum((y - y_mean)**2)

    # Сума квадратів регресії
    ssr = np.sum((y_pred - y_mean)**2)

    # Коефіцієнт детермінації R^2
    r_squared = ssr / sst

    # Скоригований R^2
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)

    # Стандартна похибка регресії
    se_regression = np.sqrt(sse / (n - 2))

    # Коефіцієнт кореляції
    r_xy = np.sum((x - x_mean) * (y - y_mean)) / n
    r_xy /= (np.sqrt(np.sum((x - x_mean)**2) / n) * np.sqrt(np.sum((y - y_mean)**2) / n))

    # Стандартні похибки коефіцієнтів регресії
    se_a1 = se_regression / np.sqrt(np.sum((x - x_mean)**2))
    se_a0 = se_regression * np.sqrt(1/n + x_mean**2 / np.sum((x - x_mean)**2))

    # t-статистики для коефіцієнтів
    t_a0 = a0 / se_a0
    t_a1 = a1 / se_a1

    # F-статистика для загальної значущості моделі
    f_statistic = (ssr / 1) / (sse / (n - 2))

    # Усереднений коефіцієнт еластичності
    elasticity = a1 * np.mean(x) / np.mean(y)

    # Критичні значення для статистичних тестів
    alpha = 0.05  # Рівень значущості 5%
    t_critical = stats.t.ppf(1 - alpha/2, n - 2)
    f_critical = stats.f.ppf(1 - alpha, 1, n - 2)

    # Довірчі інтервали для коефіцієнтів регресії
    a0_lower = a0 - t_critical * se_a0
    a0_upper = a0 + t_critical * se_a0
    a1_lower = a1 - t_critical * se_a1
    a1_upper = a1 + t_critical * se_a1

    # Збереження результатів у словник
    metrics = {
        "n": n,
        "a0": a0,
        "a1": a1,
        "r_xy": r_xy,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "se_regression": se_regression,
        "std_error": se_regression,  # Додано для сумісності з plot_regression_results
        "se_a0": se_a0,
        "se_a1": se_a1,
        "t_a0": t_a0,
        "t_a1": t_a1,
        "t_critical": t_critical,
        "f_statistic": f_statistic,
        "f_critical": f_critical,
        "elasticity": elasticity,
        "a0_ci": (a0_lower, a0_upper),
        "a1_ci": (a1_lower, a1_upper),
        "residuals": residuals,
        "y_pred": y_pred
    }

    return metrics

def print_regression_results(x, y, x_name, y_name, metrics):
    """
    Виводить результати регресійного аналізу.

    Параметри:
    x (numpy.ndarray): Масив незалежних змінних
    y (numpy.ndarray): Масив залежних змінних
    x_name (str): Назва незалежної змінної
    y_name (str): Назва залежної змінної
    metrics (dict): Словник з метриками якості моделі
    """
    print_header("Результати регресійного аналізу")

    # Видобування метрик
    a0 = metrics["a0"]
    a1 = metrics["a1"]
    r_squared = metrics["r_squared"]
    adj_r_squared = metrics["adj_r_squared"]
    r_xy = metrics["r_xy"]
    se_regression = metrics["se_regression"]
    se_a0 = metrics["se_a0"]
    se_a1 = metrics["se_a1"]
    t_a0 = metrics["t_a0"]
    t_a1 = metrics["t_a1"]
    t_critical = metrics["t_critical"]
    f_statistic = metrics["f_statistic"]
    f_critical = metrics["f_critical"]
    elasticity = metrics["elasticity"]
    a0_ci = metrics["a0_ci"]
    a1_ci = metrics["a1_ci"]

    # Виведення рівняння регресії
    print(f"\nРівняння регресії: {y_name} = {a0:.4f} + {a1:.4f} * {x_name}")

    # Виведення коефіцієнтів та їх статистик
    print("\nКоефіцієнти регресії та їх статистична значущість:")
    print(f"a0 = {a0:.4f}, стандартна похибка = {se_a0:.4f}, t-статистика = {t_a0:.4f}")
    print(f"a1 = {a1:.4f}, стандартна похибка = {se_a1:.4f}, t-статистика = {t_a1:.4f}")
    print(f"Критичне значення t-статистики (α=0.05, df={len(x)-2}) = {t_critical:.4f}")

    # Висновок про значущість коефіцієнтів
    print("\nСтатистична значущість коефіцієнтів:")
    print(f"a0: {abs(t_a0) > t_critical}")
    print(f"a1: {abs(t_a1) > t_critical}")

    # Виведення довірчих інтервалів
    print("\nДовірчі інтервали для коефіцієнтів (95%):")
    print(f"a0: [{a0_ci[0]:.4f}, {a0_ci[1]:.4f}]")
    print(f"a1: [{a1_ci[0]:.4f}, {a1_ci[1]:.4f}]")

    # Виведення метрик якості моделі
    print("\nМетрики якості моделі:")
    print(f"Коефіцієнт кореляції: r = {r_xy:.4f}")
    print(f"Коефіцієнт детермінації: R² = {r_squared:.4f}")
    print(f"Скоригований R²: {adj_r_squared:.4f}")
    print(f"Стандартна похибка регресії: {se_regression:.4f}")

    # Виведення F-статистики
    print("\nF-статистика для загальної значущості моделі:")
    print(f"F = {f_statistic:.4f}, критичне значення = {f_critical:.4f}")
    print(f"Модель статистично значуща: {f_statistic > f_critical}")

    # Виведення коефіцієнту еластичності
    print(f"\nУсереднений коефіцієнт еластичності: {elasticity:.4f}")
    interpretation = ""  # Інтерпретація еластичності
    if abs(elasticity) < 0.5:
        interpretation = "низька еластичність (нееластичний зв'язок)"
    elif abs(elasticity) < 1:
        interpretation = "середня еластичність"
    else:
        interpretation = "висока еластичність (еластичний зв'язок)"
    print(f"Інтерпретація: {interpretation}")

def compare_regression_models(x_before, y_before, x_after, y_after, metrics_before, metrics_after, x_name, y_name):
    """
    Візуалізує порівняння регресійних моделей до та після очищення викидів.

    Параметри:
    x_before (numpy.ndarray): Масив незалежних змінних до очищення
    y_before (numpy.ndarray): Масив залежних змінних до очищення
    x_after (numpy.ndarray): Масив незалежних змінних після очищення
    y_after (numpy.ndarray): Масив залежних змінних після очищення
    metrics_before (dict): Словник з метриками якості моделі до очищення
    metrics_after (dict): Словник з метриками якості моделі після очищення
    x_name (str): Назва незалежної змінної
    y_name (str): Назва залежної змінної
    """
    # Налаштування стилю
    plt.style.use('ggplot')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.figsize': (14, 8),
        'figure.dpi': 120
    })

    # Видобування метрик для обох моделей
    a0_before = metrics_before["a0"]
    a1_before = metrics_before["a1"]
    r_squared_before = metrics_before["r_squared"]
    y_pred_before = metrics_before["y_pred"]

    a0_after = metrics_after["a0"]
    a1_after = metrics_after["a1"]
    r_squared_after = metrics_after["r_squared"]
    y_pred_after = metrics_after["y_pred"]

    # Порівняння ліній регресії на одному графіку
    fig, ax = plt.subplots(figsize=(14, 8))

    # Відображення точок даних до очищення
    ax.scatter(x_before, y_before, alpha=0.5, color='blue', s=30, label='Дані до очищення')

    # Відображення точок даних після очищення
    ax.scatter(x_after, y_after, alpha=0.7, color='green', s=40, label='Дані після очищення')

    # Формування меж для ліній регресії (для кращого відображення)
    all_x = np.concatenate([x_before, x_after])
    x_min, x_max = np.min(all_x), np.max(all_x)
    x_range = np.linspace(x_min, x_max, 100)

    # Обчислення прогнозних значень для плавної лінії
    y_range_before = a0_before + a1_before * x_range
    y_range_after = a0_after + a1_after * x_range

    # Додавання ліній регресії
    ax.plot(x_range, y_range_before, color='darkblue', linestyle='-', linewidth=2,
            label=f'Модель до: y = {a0_before:.4f} + {a1_before:.4f}x, R² = {r_squared_before:.4f}')

    ax.plot(x_range, y_range_after, color='darkgreen', linestyle='-', linewidth=2,
            label=f'Модель після: y = {a0_after:.4f} + {a1_after:.4f}x, R² = {r_squared_after:.4f}')

    # Оформлення графіка
    ax.set_title('Порівняння регресійних моделей до та після очищення викидів', fontsize=20, fontweight='bold')
    ax.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax.set_ylabel(y_name, fontsize=16, fontweight='bold')

    # Покращення легенди
    legend = ax.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)

    # Покращення відображення сітки
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Додатковий графік: Порівняння залишків
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Отримання залишків
    residuals_before = metrics_before["residuals"]
    residuals_after = metrics_after["residuals"]

    # Статистики залишків
    mean_before = np.mean(residuals_before)
    std_before = np.std(residuals_before)
    mean_after = np.mean(residuals_after)
    std_after = np.std(residuals_after)

    # Налаштування гістограм
    bin_params_before = {}
    bin_params_after = {}

    if len(residuals_before) > 1000:
        # Оптимальна кількість бінів для великих даних
        data_range = np.max(residuals_before) - np.min(residuals_before)
        bin_width = 2 * stats.iqr(residuals_before) / (len(residuals_before) ** (1/3))
        n_bins = int(data_range / bin_width) if bin_width > 0 else 50
        n_bins = min(100, max(30, n_bins))
        bin_params_before['bins'] = n_bins

    if len(residuals_after) > 1000:
        data_range = np.max(residuals_after) - np.min(residuals_after)
        bin_width = 2 * stats.iqr(residuals_after) / (len(residuals_after) ** (1/3))
        n_bins = int(data_range / bin_width) if bin_width > 0 else 50
        n_bins = min(100, max(30, n_bins))
        bin_params_after['bins'] = n_bins

    # Покращення відображення гістограми
    for i, (ax, residuals, title, color, mean_val, std_val, params) in enumerate([
        (axes[0], residuals_before, 'Залишки до очищення', 'royalblue', mean_before, std_before, bin_params_before),
        (axes[1], residuals_after, 'Залишки після очищення', 'forestgreen', mean_after, std_after, bin_params_after)
    ]):
        # Гістограма з кращим візуальним розподілом
        sns.histplot(residuals, kde=True, color=color, alpha=0.6, edgecolor='black', linewidth=1.0, ax=ax, **params)

        # Додавання вертикальної лінії середнього значення
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Середнє = {mean_val:.4f}')

        # Додавання нормального розподілу
        x_norm = np.linspace(np.percentile(residuals, 0.1), np.percentile(residuals, 99.9), 1000)
        y_norm = stats.norm.pdf(x_norm, mean_val, std_val)

        # Масштабування нормального розподілу
        bin_heights = [p.get_height() for p in ax.patches] if ax.patches else []
        max_height = max(bin_heights) if bin_heights else len(residuals) / 20
        scale_factor = max_height / (np.max(y_norm) if np.max(y_norm) > 0 else 1)

        # Додавання кривої нормального розподілу
        ax.plot(x_norm, y_norm * scale_factor, 'r-', alpha=0.7, linewidth=2,
                label=f'Нормальний розподіл')

        # Додавання статистичної інформації
        stats_text = (f'n = {len(residuals):,}\n'
                      f'\u03BC = {mean_val:.4f}\n'
                      f'\u03C3 = {std_val:.4f}')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                fontsize=14)

        # Налаштування зовнішнього вигляду
        ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel('Залишки', fontsize=16, fontweight='bold')
        ax.set_ylabel('Частота', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12, loc='upper left')

        # Додавання горизонтальних ліній для стандартних відхилень
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')

        # Встановлення однакових меж по осі X для обох графіків
        # Визначення спільних меж на основі перцентилів
        all_residuals = np.concatenate([residuals_before, residuals_after])
        q_low, q_high = np.percentile(all_residuals, [0.5, 99.5])
        # Розширення меж на 10% для кращого відображення
        range_x = q_high - q_low
        xlim_min, xlim_max = q_low - 0.1 * range_x, q_high + 0.1 * range_x
        axes[i].set_xlim([xlim_min, xlim_max])

    # Спочатку застосовуємо tight_layout, щоб оптимізувати розміщення графіків
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Залишаємо місце для заголовка

    # Додаємо загальний заголовок після налаштування макету
    fig.suptitle('Порівняння розподілу залишків', fontsize=20, fontweight='bold', y=0.98)

    plt.show()

def plot_regression_results(x, y, x_name, y_name, metrics):
    """
    Створює візуалізації результатів регресійного аналізу.

    Параметри:
    x (numpy.ndarray): Масив незалежних змінних
    y (numpy.ndarray): Масив залежних змінних
    x_name (str): Назва незалежної змінної
    y_name (str): Назва залежної змінної
    metrics (dict): Словник з метриками якості моделі
    """
    # Налаштування стилю
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.figsize': (12, 7),
        'figure.dpi': 100,
    })

    # Видобування метрик
    a0 = metrics["a0"]
    a1 = metrics["a1"]
    r_squared = metrics["r_squared"]
    residuals = metrics["residuals"]
    y_pred = metrics["y_pred"]
    std_error = metrics["std_error"]

    # Графік 1: Діаграма розсіювання з лінією регресії
    plt.figure(figsize=(12, 7))

    # Додаємо полігон довірчого інтервалу (якщо бажаєте)
    if len(x) > 2:  # Перевіряємо, що є достатньо точок для обчислення довірчого інтервалу
        # Сортуємо X та прогнозні значення для правильної побудови області
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        # Обчислюємо довірчий інтервал (приблизно)
        conf_interval = 1.96 * std_error  # 95% довірчий інтервал
        lower_bound = y_pred_sorted - conf_interval
        upper_bound = y_pred_sorted + conf_interval

        # Додаємо довірчий інтервал на графік
        plt.fill_between(x_sorted, lower_bound, upper_bound,
                         color='lightblue', alpha=0.3,
                         label='95% довірчий інтервал')

    # Додаємо точки даних та лінію регресії
    plt.scatter(x, y, alpha=0.7, color='blue', edgecolor='navy', s=50, label='Дані')
    plt.plot(x, y_pred, color='red', linewidth=2, label=f'y = {a0:.4f} + {a1:.4f}x')

    # Додаємо інформацію про модель
    equation_text = f'y = {a0:.4f} + {a1:.4f}x\nR² = {r_squared:.4f}\nСтандартна похибка = {std_error:.4f}'
    plt.annotate(equation_text, xy=(0.02, 0.97), xycoords='axes fraction',
                 ha='left', va='top', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.title('Діаграма розсіювання з лінією регресії', fontweight='bold', pad=15)
    plt.xlabel(x_name, fontweight='bold')
    plt.ylabel(y_name, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.tight_layout()
    # plt.savefig('regression_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Графік 2: Залишки vs Прогнозовані значення
    plt.figure(figsize=(12, 7))
    plt.scatter(y_pred, residuals, alpha=0.7, color='green', edgecolor='darkgreen', s=50)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=1.5)

    # Додаємо горизонтальні лінії для стандартних відхилень
    std_residuals = np.std(residuals)
    plt.axhline(y=std_residuals, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    plt.axhline(y=-std_residuals, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    plt.axhline(y=2*std_residuals, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    plt.axhline(y=-2*std_residuals, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Додаємо зглажену лінію тренду для візуалізації паттернів
    if len(y_pred) > 10:  # Перевіряємо, що є достатньо точок
        try:
            from scipy.ndimage import gaussian_filter1d
            # Сортуємо за прогнозованими значеннями
            sort_idx = np.argsort(y_pred)
            y_pred_sorted = y_pred[sort_idx]
            residuals_sorted = residuals[sort_idx]
            # Застосовуємо згладжування
            smoothed = gaussian_filter1d(residuals_sorted, sigma=3)
            plt.plot(y_pred_sorted, smoothed, 'r--', alpha=0.5, linewidth=1.5)
        except ImportError:
            pass  # Якщо gaussian_filter1d недоступний, пропускаємо цей крок

    plt.title('Залишки vs Прогнозовані значення', fontweight='bold', pad=15)
    plt.xlabel('Прогнозовані значення', fontweight='bold')
    plt.ylabel('Залишки', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('residuals_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Графік 3: Гістограма залишків
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)

    # Налаштування гістограми
    bin_params = {}
    if len(residuals) > 1000:
        # Оптимальне число бінів для великих наборів даних
        data_range = np.max(residuals) - np.min(residuals)
        bin_width = 2 * stats.iqr(residuals) / (len(residuals) ** (1/3))
        n_bins = int(data_range / bin_width) if bin_width > 0 else 50
        n_bins = min(100, max(20, n_bins))  # Обмеження кількості бінів
        bin_params['bins'] = n_bins

    # Визначення статистик залишків
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)
    std_error_mean = stats.sem(residuals)

    # Добавляємо гістограму з кращим форматуванням
    hist_color = '#440154'
    hist = sns.histplot(residuals, kde=False, color=hist_color, alpha=0.7,
                        edgecolor='black', linewidth=1.0, ax=ax, stat='count', **bin_params)

    # Додаємо нормальний розподіл для порівняння
    q_low, q_high = np.percentile(residuals, [0.1, 99.9])
    x_norm = np.linspace(q_low, q_high, 1000)
    y_norm = stats.norm.pdf(x_norm, mean_residuals, std_residuals)

    # Масштабування нормального розподілу до відповідної висоти гістограми
    # Використовуємо простіший метод масштабування, без використання np.diff
    bin_heights = [p.get_height() for p in ax.patches] if len(ax.patches) > 0 else []
    max_height = max(bin_heights) if bin_heights else len(residuals) / 20
    scale_factor = max_height / np.max(y_norm) if np.max(y_norm) > 0 else 1

    # Додаємо лінію нормального розподілу
    ax.plot(x_norm, y_norm * scale_factor, color='red', linewidth=2.5, alpha=0.7, label='Нормальний розподіл')

    # Додаємо вертикальну лінію для середнього значення
    ax.axvline(mean_residuals, color='red', linestyle='--', linewidth=2.0,
               label=f'Середнє значення: {mean_residuals:.4f}')

    # Налаштування зовнішнього вигляду графіка
    ax.set_title('Гістограма залишків', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Залишки', fontsize=18, fontweight='bold')
    ax.set_ylabel('Частота', fontsize=18, fontweight='bold')

    # Додаємо текстову інформацію про статистику
    info_text = f'n = {len(residuals):,}\n'
    info_text += f'\u03BC = {mean_residuals:.4f}\n'
    info_text += f'\u03C3 = {std_residuals:.4f}'

    # Додаємо текстове поле з інформацією
    ax.text(0.97, 0.97, info_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=14)

    # Покращення легенди
    legend = ax.legend(loc='upper left', fontsize=14, frameon=True)
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('black')

    # Покращення сітки
    ax.grid(True, linestyle='--', alpha=0.6)

    # Покращення меж графіка
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    # Встановлюємо межі по x для кращого відображення
    # Використовуючи перцентилі замість просто min/max
    q_low, q_high = np.percentile(residuals, [0.5, 99.5])
    x_margin = (q_high - q_low) * 0.2  # 20% запас
    ax.set_xlim([q_low - x_margin, q_high + x_margin])

    plt.tight_layout()
    plt.show()

    # Графік 4: QQ-plot залишків з покращеним форматуванням
    plt.figure(figsize=(12, 7))

    # Використовуємо stats.probplot для створення QQ-plot
    (quantiles, ordered_residuals), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

    # Малюємо точки QQ-plot з покращеним форматуванням
    plt.scatter(quantiles, ordered_residuals, color='darkblue', alpha=0.7, s=50)

    # Додаємо лінію очікуваного нормального розподілу
    line_x = np.array([quantiles.min(), quantiles.max()])
    line_y = intercept + slope * line_x
    plt.plot(line_x, line_y, 'r-', linewidth=2)

    # Додаємо інформацію про R корелології на QQ-plot
    plt.annotate(f'R = {r:.4f}', xy=(0.02, 0.97), xycoords='axes fraction',
                 ha='left', va='top', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.title('QQ-plot залишків (перевірка нормальності)', fontweight='bold', pad=15)
    plt.xlabel('Теоретичні квантилі', fontweight='bold')
    plt.ylabel('Зразкові квантилі', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('qq_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Графік 5: Залишки vs Незалежна змінна з вдосконаленим форматуванням
    plt.figure(figsize=(12, 7))
    plt.scatter(x, residuals, alpha=0.7, color='blue', edgecolor='navy', s=50)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=1.5)

    # Додаємо горизонтальні лінії для стандартних відхилень
    plt.axhline(y=std_residuals, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label=f'+1 σ = {std_residuals:.4f}')
    plt.axhline(y=-std_residuals, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label=f'-1 σ = {-std_residuals:.4f}')

    # Додаємо локально зважену регресію (LOWESS) для виявлення тренду, якщо можливо
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        if len(x) > 10:  # Перевіряємо, що є достатньо точок
            # Сортуємо за x
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            residuals_sorted = residuals[sort_idx]
            # Застосовуємо LOWESS
            lowess_result = lowess(residuals_sorted, x_sorted, frac=0.3)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], 'g-',
                     alpha=0.7, linewidth=2, label='LOWESS тренд')
    except ImportError:
        pass  # Якщо lowess недоступний, пропускаємо цей крок

    plt.title(f'Залишки vs {x_name}', fontweight='bold', pad=15)
    plt.xlabel(x_name, fontweight='bold')
    plt.ylabel('Залишки', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('residuals_vs_x.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_linear_regression(x, y, x_name, y_name):
    """
    Проводить повний аналіз лінійної регресії.

    Параметри:
    x (numpy.ndarray): Масив незалежних змінних
    y (numpy.ndarray): Масив залежних змінних
    x_name (str): Назва незалежної змінної
    y_name (str): Назва залежної змінної

    Повертає:
    dict: Результати аналізу
    """
    print_header("Лінійний регресійний аналіз")

    # Розрахунок параметрів регресії
    a0, a1 = calculate_regression_parameters(x, y)
    print(f"Рівняння регресії: {y_name} = {a0:.4f} + {a1:.4f} * {x_name}")

    # Розрахунок метрик якості моделі
    metrics = calculate_regression_metrics(x, y, a0, a1)

    # Виведення повних результатів
    print_regression_results(x, y, x_name, y_name, metrics)

    # Візуалізація результатів
    plot_regression_results(x, y, x_name, y_name, metrics)

    return metrics

def analyze_data_with_outlier_removal(file_path, column_x, column_y, delimiter=',', confidence=0.95, max_iterations=3):
    """
    Повний аналіз даних з виявленням і видаленням викидів та подальшим регресійним аналізом

    Параметри:
    file_path (str): Шлях до CSV файлу
    column_x (str, int): Назва або індекс стовпця для X
    column_y (str, int): Назва або індекс стовпця для Y
    delimiter (str): Розділювач у CSV файлі
    confidence (float): Рівень довіри для виявлення викидів
    max_iterations (int): Максимальна кількість ітерацій очищення

    Повертає:
    tuple: (results_before, results_after) - результати аналізу до та після очищення
    """
    # Зчитування даних
    x_data, y_data, df, x_name, y_name = read_csv_data(file_path, column_x, column_y, delimiter)

    # Аналіз початкових даних
    print_header("АНАЛІЗ ПОЧАТКОВИХ ДАНИХ")
    results_before = analyze_linear_regression(x_data, y_data, x_name, y_name)

    # Виявлення та видалення викидів
    x_clean, y_clean, outlier_indices = remove_outliers_from_data(
        x_data, y_data, x_name, y_name, confidence, max_iterations
    )

    # Візуалізація гістограм до та після очищення
    plot_histograms(x_data, x_clean, x_name)
    plot_histograms(y_data, y_clean, y_name)

    # Візуалізація діаграм розсіювання до та після очищення
    plot_scatter_before_after(x_data, y_data, x_clean, y_clean, x_name, y_name)

    # Аналіз очищених даних
    if len(outlier_indices) > 0:
        print_header("АНАЛІЗ ДАНИХ ПІСЛЯ ВИДАЛЕННЯ ВИКИДІВ")
        results_after = analyze_linear_regression(x_clean, y_clean, x_name, y_name)
    else:
        print("Викиди не виявлені, результати аналізу не змінилися")
        results_after = results_before

    # Порівняння результатів до та після очищення
    if len(outlier_indices) > 0:
        print_header("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ ДО ТА ПІСЛЯ ОЧИЩЕННЯ")
        print(f"Кількість спостережень до очищення: {len(x_data)}")
        print(f"Кількість спостережень після очищення: {len(x_clean)}")
        print(f"Видалено викидів: {len(outlier_indices)}")
        print("\nРівняння регресії:")
        print(f"До:    {y_name} = {results_before['a0']:.4f} + {results_before['a1']:.4f} * {x_name}")
        print(f"Після: {y_name} = {results_after['a0']:.4f} + {results_after['a1']:.4f} * {x_name}")
        print("\nКоефіцієнт детермінації R²:")
        print(f"До:    {results_before['r_squared']:.4f}")
        print(f"Після: {results_after['r_squared']:.4f}")
        print(f"Зміна: {results_after['r_squared'] - results_before['r_squared']:.4f}")

        # Візуалізація порівняння моделей до та після очищення
        print_header("ВІЗУАЛІЗАЦІЯ ПОРІВНЯННЯ МОДЕЛЕЙ")
        # Викликаємо функцію порівняння моделей
        compare_regression_models(x_data, y_data, x_clean, y_clean, results_before, results_after, x_name, y_name)

    return results_before, results_after

def main():
    """
    Головна функція програми, яка керує процесом аналізу даних
    """
    try:
        # Зчитування параметрів від користувача
        print("=" * 60)

        # Отримання шляху до файлу
        file_path = "cleanest_data.csv"

        # Отримання інформації про стовпці
        column_x = "partner_total_orders"  # Стовпець з незалежною змінною (X)
        column_y = "partner_total_messages"  # Стовпець з залежною змінною (Y)


        # Отримання розділювача
        delimiter = ","  # Розділювач у CSV файлі

        # Отримання рівня довіри
        confidence = 0.95  # Рівень довіри

        # Отримання кількості ітерацій
        max_iterations = 1000000  # Максимальна кількість ітерацій для виявлення викидів

        # Виконання аналізу
        analyze_data_with_outlier_removal(
            file_path, column_x, column_y, delimiter, confidence, max_iterations
        )

    except KeyboardInterrupt:
        print("\nРоботу програми перервано користувачем.")
    except Exception as e:
        print(f"\nПомилка: {str(e)}")

if __name__ == "__main__":
    # Налаштування стилю графіків
    sns.set_theme(style="whitegrid")
    main()
