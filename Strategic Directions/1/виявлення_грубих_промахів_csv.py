#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
from scipy import stats

"""
Скрипт для виявлення та ідентифікації грубих промахів у рядах спостереження.

Дозволяє:
- зчитувати дані з CSV файлу
- вказувати конкретний стовпець для аналізу
- автоматично розраховувати критичні значення 
- виявляти та видаляти грубі промахи
- обчислювати статистичні показники
"""

def print_header(text):
    """Функція виведення заголовка секції"""
    print("\n" + "="*50)
    print(text)
    print("="*50)

def read_csv_data(file_path, column=None, delimiter=','):
    """
    Зчитує дані з CSV файлу

    Параметри:
    file_path (str): Шлях до CSV файлу
    column (str або int): Назва або індекс стовпця для аналізу
    delimiter (str): Розділювач у CSV файлі

    Повертає:
    numpy.ndarray: Масив даних для аналізу
    """
    try:
        # Зчитування CSV файлу
        df = pd.read_csv(file_path, delimiter=delimiter)

        # Якщо стовпець не вказано, використовуємо перший
        if column is None:
            column = 0

        # Отримання даних з вказаного стовпця
        if isinstance(column, int):
            if column < len(df.columns):
                data = df.iloc[:, column].values
                column_name = df.columns[column]
            else:
                raise ValueError(f"Стовпець з індексом {column} не існує в файлі. Доступні індекси: 0-{len(df.columns)-1}")
        else:
            if column in df.columns:
                data = df[column].values
                column_name = column
            else:
                raise ValueError(f"Стовпець '{column}' не знайдено в файлі. Доступні стовпці: {', '.join(df.columns)}")

        print(f"Зчитано дані зі стовпця '{column_name}', кількість значень: {len(data)}")

        # Перевірка, чи дані числові
        try:
            data = data.astype(float)
        except ValueError:
            raise ValueError(f"Стовпець '{column_name}' містить нечислові дані")

        return data

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

def check_outliers(data, confidence=0.95):
    """
    Перевірка наявності викидів у серії даних

    Параметри:
    data (numpy.ndarray): Масив даних
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

    print(f"γ1 = {gamma1:.3f}, γ2 = {gamma2:.3f}, γp = {gamma_p:.3f}")

    if gamma1 > gamma_p:
        has_outliers = True
        max_index = np.argmax(data)
        outlier_indices.append(max_index)
        print(f"γ1 > γp: Викид у максимальному значенні {data[max_index]:.3f} (індекс {max_index})")
    else:
        print(f"γ1 <= γp: Немає викиду у максимальному значенні")

    if gamma2 > gamma_p:
        has_outliers = True
        min_index = np.argmin(data)
        outlier_indices.append(min_index)
        print(f"γ2 > γp: Викид у мінімальному значенні {data[min_index]:.3f} (індекс {min_index})")
    else:
        print(f"γ2 <= γp: Немає викиду у мінімальному значенні")

    return has_outliers, outlier_indices, (n, x_mean, D, sigma)

def remove_outliers(data, outlier_indices):
    """
    Видалення викидів з серії даних

    Параметри:
    data (numpy.ndarray): Масив даних
    outlier_indices (list): Індекси викидів

    Повертає:
    numpy.ndarray: Очищений масив даних
    """
    return np.delete(data, outlier_indices)

def get_student_t(df, confidence=0.95):
    """
    Отримання коефіцієнта Стьюдента для заданих параметрів

    Параметри:
    df (int): Кількість ступенів свободи
    confidence (float): Рівень довіри

    Повертає:
    float: Коефіцієнт Стьюдента
    """
    return stats.t.ppf(1 - (1 - confidence) / 2, df)

def confidence_interval(data, confidence=0.95):
    """
    Розрахунок довірчого інтервалу для даних

    Параметри:
    data (numpy.ndarray): Масив даних
    confidence (float): Рівень довіри

    Повертає:
    tuple: (lower_bound, upper_bound, delta) - нижня межа, верхня межа, довірча межа
    """
    n, x_mean, D, sigma = calculate_statistics(data)

    # Стандартна похибка середнього
    se = sigma / np.sqrt(n)

    # Коефіцієнт Стьюдента
    t_p = get_student_t(n-1, confidence)

    # Довірча межа випадкової величини
    delta = t_p * se

    # Довірчий інтервал
    lower_bound = x_mean - delta
    upper_bound = x_mean + delta

    return lower_bound, upper_bound, delta, se, t_p

def analyze_data(data, confidence=0.95, max_iterations=3):
    """
    Повний аналіз даних з виявленням викидів, очищенням та розрахунком статистик

    Параметри:
    data (numpy.ndarray): Масив даних
    confidence (float): Рівень довіри
    max_iterations (int): Максимальна кількість ітерацій для виявлення викидів

    Повертає:
    dict: Результати аналізу
    """
    print_header("ПОЧАТКОВІ ДАНІ")
    print(f"Кількість значень: {len(data)}")
    print(f"Дані: {data}")

    # Ітеративне виявлення та видалення викидів
    print_header("ПЕРЕВІРКА НАЯВНОСТІ ГРУБИХ ВИКИДІВ")

    current_data = data.copy()
    iteration = 1

    while iteration <= max_iterations:
        print(f"\nІтерація {iteration}:")

        has_outliers, outlier_indices, stats = check_outliers(current_data, confidence)

        if has_outliers:
            outlier_values = current_data[outlier_indices]
            current_data = remove_outliers(current_data, outlier_indices)

            print(f"Видалені значення: {outlier_values}")
            print(f"Залишилось значень: {len(current_data)}")

            iteration += 1
        else:
            print(f"Викидів не виявлено. Аналіз завершено.")
            break

    if iteration > max_iterations:
        print(f"Досягнуто максимальну кількість ітерацій ({max_iterations}).")

    # Розрахунок статистик для очищених даних
    print_header("СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ ОЧИЩЕНИХ ДАНИХ")

    n, x_mean, D, sigma = calculate_statistics(current_data)

    print(f"Кількість значень: {n}")
    print(f"Середнє значення: {x_mean:.6f}")
    print(f"Дисперсія: {D:.6f}")
    print(f"Середньоквадратичне відхилення: {sigma:.6f}")

    # Розрахунок довірчого інтервалу
    print_header("ДОВІРЧИЙ ІНТЕРВАЛ")

    lower_bound, upper_bound, delta, se, t_p = confidence_interval(current_data, confidence)

    print(f"Рівень довіри: {confidence}")
    print(f"Стандартна похибка середнього: {se:.6f}")
    print(f"Коефіцієнт Стьюдента (t_p): {t_p:.4f}")
    print(f"Довірча межа: {delta:.6f}")
    print(f"Довірчий інтервал: {lower_bound:.6f} - {upper_bound:.6f}")
    print(f"Інтервальна оцінка результату: x ∈ [{lower_bound:.6f} : {upper_bound:.6f}], P={confidence}")

    # Результати аналізу
    results = {
        "initial_data": data,
        "cleaned_data": current_data,
        "mean": x_mean,
        "variance": D,
        "std": sigma,
        "confidence_interval": (lower_bound, upper_bound),
        "confidence_level": confidence
    }

    return results

def main():
    """Основна функція програми"""

    # Параметри задаються прямо в скрипті
    file_path = "cleanest_data.csv"  # Шлях до CSV файлу з даними
    column = "partner_order_age_days"  # Стовпець для аналізу (індекс або назва)
    delimiter = ","  # Розділювач у CSV файлі
    confidence = 0.95  # Рівень довіри
    max_iterations = 100  # Максимальна кількість ітерацій для виявлення викидів

    try:
        # Визначення стовпця для аналізу
        if isinstance(column, str) and column.isdigit():
            column = int(column)

        # Зчитування даних
        data = read_csv_data(file_path, column, delimiter)

        # Аналіз даних
        analyze_data(data, confidence, max_iterations)

    except Exception as e:
        print(f"Помилка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
