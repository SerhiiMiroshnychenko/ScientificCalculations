#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

"""
Скрипт для виявлення та ідентифікації грубих промахів у рядах спостереження.

Дозволяє:
- зчитувати два стовпці даних з CSV файлу
- вказувати конкретні стовпці для аналізу
- автоматично розраховувати критичні значення 
- виявляти та видаляти грубі промахи з обох стовпців одночасно
- обчислювати статистичні показники
- візуалізувати результати аналізу
"""

def print_header(text):
    """Функція виведення заголовка секції"""
    print("\n" + "="*50)
    print(text)
    print("="*50)

def read_csv_data(file_path, column1, column2, delimiter=','):
    """
    Зчитує два стовпці даних з CSV файлу

    Параметри:
    file_path (str): Шлях до CSV файлу
    column1 (str або int): Назва або індекс першого стовпця для аналізу
    column2 (str або int): Назва або індекс другого стовпця для аналізу
    delimiter (str): Розділювач у CSV файлі

    Повертає:
    tuple: (data1, data2, df) - дані першого стовпця, дані другого стовпця, датафрейм
    """
    try:
        # Зчитування CSV файлу
        df = pd.read_csv(file_path, delimiter=delimiter)

        # Отримання даних з першого стовпця
        if isinstance(column1, int):
            if column1 < len(df.columns):
                data1 = df.iloc[:, column1].values
                column1_name = df.columns[column1]
            else:
                raise ValueError(f"Стовпець з індексом {column1} не існує в файлі. Доступні індекси: 0-{len(df.columns)-1}")
        else:
            if column1 in df.columns:
                data1 = df[column1].values
                column1_name = column1
            else:
                raise ValueError(f"Стовпець '{column1}' не знайдено в файлі. Доступні стовпці: {', '.join(df.columns)}")

        # Отримання даних з другого стовпця
        if isinstance(column2, int):
            if column2 < len(df.columns):
                data2 = df.iloc[:, column2].values
                column2_name = df.columns[column2]
            else:
                raise ValueError(f"Стовпець з індексом {column2} не існує в файлі. Доступні індекси: 0-{len(df.columns)-1}")
        else:
            if column2 in df.columns:
                data2 = df[column2].values
                column2_name = column2
            else:
                raise ValueError(f"Стовпець '{column2}' не знайдено в файлі. Доступні стовпці: {', '.join(df.columns)}")

        print(f"Зчитано дані зі стовпців '{column1_name}' та '{column2_name}', кількість значень: {len(data1)}")

        # Перевірка, чи дані числові
        try:
            data1 = data1.astype(float)
            data2 = data2.astype(float)
        except ValueError:
            raise ValueError(f"Стовпці містять нечислові дані")

        return data1, data2, df, column1_name, column2_name

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

    print(f"{label}: γ1 = {gamma1:.3f}, γ2 = {gamma2:.3f}, γp = {gamma_p:.3f}")

    if gamma1 > gamma_p:
        has_outliers = True
        max_index = np.argmax(data)
        outlier_indices.append(max_index)
        print(f"{label}: γ1 > γp: Викид у максимальному значенні {data[max_index]:.3f} (індекс {max_index})")
    else:
        print(f"{label}: γ1 <= γp: Немає викиду у максимальному значенні")

    if gamma2 > gamma_p:
        has_outliers = True
        min_index = np.argmin(data)
        outlier_indices.append(min_index)
        print(f"{label}: γ2 > γp: Викид у мінімальному значенні {data[min_index]:.3f} (індекс {min_index})")
    else:
        print(f"{label}: γ2 <= γp: Немає викиду у мінімальному значенні")

    return has_outliers, outlier_indices, (n, x_mean, D, sigma)

def visualize_initial_data(data1, data2, column1_name, column2_name):
    """
    Візуалізація початкових даних

    Параметри:
    data1 (numpy.ndarray): Масив даних першого стовпця
    data2 (numpy.ndarray): Масив даних другого стовпця
    column1_name (str): Назва першого стовпця
    column2_name (str): Назва другого стовпця
    """
    plt.figure(figsize=(16, 12))

    # Графік розсіювання
    plt.subplot(2, 2, 1)
    plt.scatter(data1, data2, alpha=0.7, color='blue')
    plt.title(f'Діаграма розсіювання: {column1_name} vs {column2_name}')
    plt.xlabel(column1_name)
    plt.ylabel(column2_name)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Гістограма першого стовпця
    plt.subplot(2, 2, 2)
    sns.histplot(data1, kde=True, color='blue')
    plt.title(f'Розподіл даних: {column1_name}')
    plt.xlabel(column1_name)
    plt.ylabel('Частота')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Гістограма другого стовпця
    plt.subplot(2, 2, 3)
    sns.histplot(data2, kde=True, color='green')
    plt.title(f'Розподіл даних: {column2_name}')
    plt.xlabel(column2_name)
    plt.ylabel('Частота')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Коробковий графік для обох стовпців
    plt.subplot(2, 2, 4)
    boxplot_data = [data1, data2]
    plt.boxplot(boxplot_data, labels=[column1_name, column2_name])
    plt.title('Коробковий графік для обох стовпців')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('initial_data_visualization.png', dpi=300)
    plt.close()
    print("\nСтворено візуалізацію початкових даних: initial_data_visualization.png")

def visualize_outliers(data1, data2, outlier_indices, column1_name, column2_name):
    """
    Візуалізація даних з позначеними викидами

    Параметри:
    data1 (numpy.ndarray): Масив даних першого стовпця
    data2 (numpy.ndarray): Масив даних другого стовпця
    outlier_indices (list): Індекси викидів
    column1_name (str): Назва першого стовпця
    column2_name (str): Назва другого стовпця
    """
    plt.figure(figsize=(16, 8))

    # Створення масок для викидів та нормальних даних
    mask_outliers = np.zeros(len(data1), dtype=bool)
    for idx in outlier_indices:
        if idx < len(mask_outliers):
            mask_outliers[idx] = True
    mask_normal = ~mask_outliers

    # Графік розсіювання з позначеними викидами
    plt.subplot(1, 2, 1)
    plt.scatter(data1[mask_normal], data2[mask_normal], alpha=0.7, color='blue', label='Нормальні дані')
    plt.scatter(data1[mask_outliers], data2[mask_outliers], alpha=0.7, color='red', marker='x', s=100, label='Викиди')
    plt.title(f'Діаграма розсіювання з викидами: {column1_name} vs {column2_name}')
    plt.xlabel(column1_name)
    plt.ylabel(column2_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Коробковий графік з позначеними викидами для обох стовпців
    plt.subplot(1, 2, 2)
    outlier_props = dict(marker='o', markerfacecolor='red', markersize=10, markeredgecolor='black')
    boxplot_data = [data1, data2]
    plt.boxplot(boxplot_data, labels=[column1_name, column2_name], flierprops=outlier_props)
    plt.title('Коробковий графік з викидами для обох стовпців')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('outliers_visualization.png', dpi=300)
    plt.close()
    print("\nСтворено візуалізацію викидів: outliers_visualization.png")

def visualize_cleaned_data(clean_data1, clean_data2, orig_data1, orig_data2, column1_name, column2_name):
    """
    Візуалізація очищених даних порівняно з початковими

    Параметри:
    clean_data1 (numpy.ndarray): Очищений масив даних першого стовпця
    clean_data2 (numpy.ndarray): Очищений масив даних другого стовпця
    orig_data1 (numpy.ndarray): Початковий масив даних першого стовпця
    orig_data2 (numpy.ndarray): Початковий масив даних другого стовпця
    column1_name (str): Назва першого стовпця
    column2_name (str): Назва другого стовпця
    """
    plt.figure(figsize=(16, 12))

    # Графік розсіювання: початкові дані vs очищені
    plt.subplot(2, 2, 1)
    plt.scatter(orig_data1, orig_data2, alpha=0.4, color='red', label='Початкові дані')
    plt.scatter(clean_data1, clean_data2, alpha=0.7, color='blue', label='Очищені дані')
    plt.title(f'Порівняння даних: {column1_name} vs {column2_name}')
    plt.xlabel(column1_name)
    plt.ylabel(column2_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Гістограма першого стовпця: порівняння
    plt.subplot(2, 2, 2)
    sns.histplot(orig_data1, kde=True, color='red', alpha=0.5, label='Початкові дані')
    sns.histplot(clean_data1, kde=True, color='blue', alpha=0.5, label='Очищені дані')
    plt.title(f'Порівняння розподілів: {column1_name}')
    plt.xlabel(column1_name)
    plt.ylabel('Частота')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Гістограма другого стовпця: порівняння
    plt.subplot(2, 2, 3)
    sns.histplot(orig_data2, kde=True, color='red', alpha=0.5, label='Початкові дані')
    sns.histplot(clean_data2, kde=True, color='blue', alpha=0.5, label='Очищені дані')
    plt.title(f'Порівняння розподілів: {column2_name}')
    plt.xlabel(column2_name)
    plt.ylabel('Частота')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Коробковий графік: порівняння
    plt.subplot(2, 2, 4)
    boxplot_data = [orig_data1, clean_data1, orig_data2, clean_data2]
    plt.boxplot(boxplot_data, labels=[f'{column1_name}\nпочаткові', f'{column1_name}\nочищені',
                                      f'{column2_name}\nпочаткові', f'{column2_name}\nочищені'])
    plt.title('Порівняння коробкових графіків')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('cleaned_data_comparison.png', dpi=300)
    plt.close()
    print("\nСтворено візуалізацію порівняння даних: cleaned_data_comparison.png")

def visualize_confidence_intervals(data1, data2, conf_int1, conf_int2, column1_name, column2_name, confidence):
    """
    Візуалізація довірчих інтервалів

    Параметри:
    data1 (numpy.ndarray): Масив даних першого стовпця
    data2 (numpy.ndarray): Масив даних другого стовпця
    conf_int1 (tuple): Довірчий інтервал для першого стовпця (нижня, верхня межа)
    conf_int2 (tuple): Довірчий інтервал для другого стовпця (нижня, верхня межа)
    column1_name (str): Назва першого стовпця
    column2_name (str): Назва другого стовпця
    confidence (float): Рівень довіри
    """
    plt.figure(figsize=(16, 8))

    # Довірчий інтервал для першого стовпця
    plt.subplot(1, 2, 1)
    mean1 = np.mean(data1)
    sns.histplot(data1, kde=True, color='blue')
    plt.axvline(mean1, color='red', linestyle='--', label=f'Середнє: {mean1:.3f}')
    plt.axvline(conf_int1[0], color='green', linestyle='-', label=f'Нижня межа: {conf_int1[0]:.3f}')
    plt.axvline(conf_int1[1], color='green', linestyle='-', label=f'Верхня межа: {conf_int1[1]:.3f}')
    plt.fill_betweenx([0, plt.gca().get_ylim()[1]], conf_int1[0], conf_int1[1], alpha=0.2, color='green')
    plt.title(f'Довірчий інтервал для {column1_name} (P={confidence})')
    plt.xlabel(column1_name)
    plt.ylabel('Частота')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Довірчий інтервал для другого стовпця
    plt.subplot(1, 2, 2)
    mean2 = np.mean(data2)
    sns.histplot(data2, kde=True, color='blue')
    plt.axvline(mean2, color='red', linestyle='--', label=f'Середнє: {mean2:.3f}')
    plt.axvline(conf_int2[0], color='green', linestyle='-', label=f'Нижня межа: {conf_int2[0]:.3f}')
    plt.axvline(conf_int2[1], color='green', linestyle='-', label=f'Верхня межа: {conf_int2[1]:.3f}')
    plt.fill_betweenx([0, plt.gca().get_ylim()[1]], conf_int2[0], conf_int2[1], alpha=0.2, color='green')
    plt.title(f'Довірчий інтервал для {column2_name} (P={confidence})')
    plt.xlabel(column2_name)
    plt.ylabel('Частота')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('confidence_intervals.png', dpi=300)
    plt.close()
    print("\nСтворено візуалізацію довірчих інтервалів: confidence_intervals.png")

def analyze_dual_data(data1, data2, column1_name, column2_name, confidence=0.95, max_iterations=3):
    """
    Повний аналіз двох стовпців даних з виявленням викидів, очищенням та розрахунком статистик

    Параметри:
    data1 (numpy.ndarray): Масив даних першого стовпця
    data2 (numpy.ndarray): Масив даних другого стовпця
    column1_name (str): Назва першого стовпця
    column2_name (str): Назва другого стовпця
    confidence (float): Рівень довіри
    max_iterations (int): Максимальна кількість ітерацій для виявлення викидів

    Повертає:
    dict: Результати аналізу
    """
    print_header("ПОЧАТКОВІ ДАНІ")
    print(f"Кількість значень: {len(data1)}")
    print(f"Дані {column1_name}: {data1}")
    print(f"Дані {column2_name}: {data2}")

    # Візуалізація початкових даних
    visualize_initial_data(data1, data2, column1_name, column2_name)

    # Ітеративне виявлення та видалення викидів
    print_header("ПЕРЕВІРКА НАЯВНОСТІ ГРУБИХ ВИКИДІВ")

    current_data1 = data1.copy()
    current_data2 = data2.copy()
    iteration = 1
    all_removed_indices = []

    while iteration <= max_iterations:
        print(f"\nІтерація {iteration}:")

        has_outliers1, outlier_indices1, stats1 = check_outliers(current_data1, column1_name, confidence)
        has_outliers2, outlier_indices2, stats2 = check_outliers(current_data2, column2_name, confidence)

        # Об'єднання індексів викидів з обох стовпців
        all_outlier_indices = list(set(outlier_indices1 + outlier_indices2))

        if all_outlier_indices:
            # Візуалізація викидів перед видаленням
            if iteration == 1:
                visualize_outliers(current_data1, current_data2, all_outlier_indices, column1_name, column2_name)

            # Виведення викидів
            print(f"\nВикидів знайдено: {len(all_outlier_indices)}")
            for idx in all_outlier_indices:
                if idx < len(current_data1) and idx < len(current_data2):
                    print(f"Індекс {idx}: {column1_name}={current_data1[idx]:.3f}, {column2_name}={current_data2[idx]:.3f}")
                    all_removed_indices.append((len(current_data1) - len(all_outlier_indices) + idx, current_data1[idx], current_data2[idx]))

            # Видалення викидів з обох стовпців
            current_data1 = np.delete(current_data1, all_outlier_indices)
            current_data2 = np.delete(current_data2, all_outlier_indices)

            print(f"Залишилось значень: {len(current_data1)}")

            iteration += 1
        else:
            print(f"\nВикидів не виявлено. Аналіз завершено.")
            break

    if iteration > max_iterations:
        print(f"\nДосягнуто максимальну кількість ітерацій ({max_iterations}).")

    # Візуалізація порівняння початкових та очищених даних
    visualize_cleaned_data(current_data1, current_data2, data1, data2, column1_name, column2_name)

    # Розрахунок статистик для очищених даних
    print_header(f"СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ ОЧИЩЕНИХ ДАНИХ")

    n1, x_mean1, D1, sigma1 = calculate_statistics(current_data1)
    n2, x_mean2, D2, sigma2 = calculate_statistics(current_data2)

    print(f"{column1_name}:")
    print(f"  Кількість значень: {n1}")
    print(f"  Середнє значення: {x_mean1:.6f}")
    print(f"  Дисперсія: {D1:.6f}")
    print(f"  Середньоквадратичне відхилення: {sigma1:.6f}")

    print(f"\n{column2_name}:")
    print(f"  Кількість значень: {n2}")
    print(f"  Середнє значення: {x_mean2:.6f}")
    print(f"  Дисперсія: {D2:.6f}")
    print(f"  Середньоквадратичне відхилення: {sigma2:.6f}")

    # Розрахунок довірчих інтервалів
    print_header("ДОВІРЧІ ІНТЕРВАЛИ")

    # Функція для розрахунку довірчого інтервалу
    def get_confidence_interval(data, confidence):
        n, x_mean, D, sigma = calculate_statistics(data)
        se = sigma / np.sqrt(n)  # Стандартна похибка середнього
        t_p = stats.t.ppf(1 - (1 - confidence) / 2, n - 1)  # Коефіцієнт Стьюдента
        delta = t_p * se  # Довірча межа
        lower_bound = x_mean - delta
        upper_bound = x_mean + delta
        return lower_bound, upper_bound, delta, se, t_p

    # Довірчий інтервал для першого стовпця
    lower1, upper1, delta1, se1, t_p1 = get_confidence_interval(current_data1, confidence)

    print(f"{column1_name}:")
    print(f"  Рівень довіри: {confidence}")
    print(f"  Стандартна похибка середнього: {se1:.6f}")
    print(f"  Коефіцієнт Стьюдента: {t_p1:.4f}")
    print(f"  Довірча межа: {delta1:.6f}")
    print(f"  Довірчий інтервал: {lower1:.6f} - {upper1:.6f}")
    print(f"  Інтервальна оцінка: x ∈ [{lower1:.6f} : {upper1:.6f}], P={confidence}")

    # Довірчий інтервал для другого стовпця
    lower2, upper2, delta2, se2, t_p2 = get_confidence_interval(current_data2, confidence)

    print(f"\n{column2_name}:")
    print(f"  Рівень довіри: {confidence}")
    print(f"  Стандартна похибка середнього: {se2:.6f}")
    print(f"  Коефіцієнт Стьюдента: {t_p2:.4f}")
    print(f"  Довірча межа: {delta2:.6f}")
    print(f"  Довірчий інтервал: {lower2:.6f} - {upper2:.6f}")
    print(f"  Інтервальна оцінка: x ∈ [{lower2:.6f} : {upper2:.6f}], P={confidence}")

    # Візуалізація довірчих інтервалів
    visualize_confidence_intervals(current_data1, current_data2, (lower1, upper1), (lower2, upper2),
                                   column1_name, column2_name, confidence)

    # Результати аналізу
    results = {
        "column1": {
            "name": column1_name,
            "initial_data": data1,
            "cleaned_data": current_data1,
            "mean": x_mean1,
            "variance": D1,
            "std": sigma1,
            "confidence_interval": (lower1, upper1)
        },
        "column2": {
            "name": column2_name,
            "initial_data": data2,
            "cleaned_data": current_data2,
            "mean": x_mean2,
            "variance": D2,
            "std": sigma2,
            "confidence_interval": (lower2, upper2)
        },
        "confidence_level": confidence,
        "removed_indices": all_removed_indices
    }

    # Створення зведеного графіку з основними результатами
    create_summary_visualization(results)

    return results

def create_summary_visualization(results):
    """
    Створення зведеного графіку з основними результатами

    Параметри:
    results (dict): Словник з результатами аналізу
    """
