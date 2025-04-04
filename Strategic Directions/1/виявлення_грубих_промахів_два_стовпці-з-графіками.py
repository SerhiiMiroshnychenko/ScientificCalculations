#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
from scipy import stats
import matplotlib.pyplot as plt

"""
Скрипт для виявлення та ідентифікації грубих промахів у рядах спостереження.

Дозволяє:
- зчитувати два стовпці даних з CSV файлу
- вказувати конкретні стовпці для аналізу
- автоматично розраховувати критичні значення 
- виявляти та видаляти грубі промахи з обох стовпців одночасно
- обчислювати статистичні показники
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

        return data1, data2, df

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

    # Ітеративне виявлення та видалення викидів
    print_header("ПЕРЕВІРКА НАЯВНОСТІ ГРУБИХ ВИКИДІВ")

    current_data1 = data1.copy()
    current_data2 = data2.copy()

    # Збереження початкових статистик для використання у візуалізації
    initial_stats1 = calculate_statistics(data1)
    initial_stats2 = calculate_statistics(data2)

    iteration = 1

    while iteration <= max_iterations:
        print(f"\nІтерація {iteration}:")

        has_outliers1, outlier_indices1, stats1 = check_outliers(current_data1, column1_name, confidence)
        has_outliers2, outlier_indices2, stats2 = check_outliers(current_data2, column2_name, confidence)

        # Об'єднання індексів викидів з обох стовпців
        all_outlier_indices = list(set(outlier_indices1 + outlier_indices2))

        if all_outlier_indices:
            # Виведення викидів
            print(f"\nВикидів знайдено: {len(all_outlier_indices)}")
            for idx in all_outlier_indices:
                if idx < len(current_data1) and idx < len(current_data2):
                    print(f"Індекс {idx}: {column1_name}={current_data1[idx]:.3f}, {column2_name}={current_data2[idx]:.3f}")

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
        "removed_indices": list(set(range(len(data1))) - set(range(len(current_data1))))
    }

    # Створення інформативної візуалізації результатів аналізу
    plt.figure(figsize=(16, 12))

    # 1. Порівняння початкових та очищених даних (розмір серій)
    plt.subplot(2, 2, 1)
    plt.title('Порівняння розміру серій даних')
    labels = ['Початкові дані', 'Очищені дані']
    sizes_1 = [len(data1), len(current_data1)]
    sizes_2 = [len(data2), len(current_data2)]

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width/2, sizes_1, width, label=column1_name, color='blue')
    plt.bar(x + width/2, sizes_2, width, label=column2_name, color='red')

    plt.ylabel('Кількість спостережень')
    for i, v in enumerate(sizes_1):
        plt.text(i - width/2, v + 5, str(v), ha='center', va='bottom', color='blue')
    for i, v in enumerate(sizes_2):
        plt.text(i + width/2, v + 5, str(v), ha='center', va='bottom', color='red')

    plt.xticks(x, labels)
    plt.grid(True, axis='y')
    plt.legend()

    # 2. Статистичні показники (середнє, СКВ, довірчі інтервали)
    plt.subplot(2, 2, 2)
    plt.title('Статистичні показники')

    # Створюємо таблицю для відображення статистик
    column_labels = ['Початкове', 'Очищене', 'Зміна (%)']
    row_labels = [f'{column1_name}\nСереднє',
                  f'{column1_name}\nСКВ',
                  f'{column2_name}\nСереднє',
                  f'{column2_name}\nСКВ']

    # Обчислюємо процентну зміну статистик
    mean1_change = (x_mean1 - initial_stats1[1]) / initial_stats1[1] * 100 if initial_stats1[1] != 0 else 0
    std1_change = (sigma1 - initial_stats1[3]) / initial_stats1[3] * 100 if initial_stats1[3] != 0 else 0
    mean2_change = (x_mean2 - initial_stats2[1]) / initial_stats2[1] * 100 if initial_stats2[1] != 0 else 0
    std2_change = (sigma2 - initial_stats2[3]) / initial_stats2[3] * 100 if initial_stats2[3] != 0 else 0

    table_data = [
        [f'{initial_stats1[1]:.3f}', f'{x_mean1:.3f}', f'{mean1_change:.2f}%'],
        [f'{initial_stats1[3]:.3f}', f'{sigma1:.3f}', f'{std1_change:.2f}%'],
        [f'{initial_stats2[1]:.3f}', f'{x_mean2:.3f}', f'{mean2_change:.2f}%'],
        [f'{initial_stats2[3]:.3f}', f'{sigma2:.3f}', f'{std2_change:.2f}%']
    ]

    # Відображаємо таблицю
    table = plt.table(cellText=table_data,
                      rowLabels=row_labels,
                      colLabels=column_labels,
                      cellLoc='center',
                      rowLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.axis('off')

    # 3. Інформація про вилучені дані та довірчі інтервали
    plt.subplot(2, 2, 3)
    plt.title('Довірчі інтервали')

    # Створюємо графік довірчих інтервалів для обох стовпців
    y_pos = [1, 0]
    labels = [column1_name, column2_name]
    means = [x_mean1, x_mean2]
    errors = [delta1, delta2]

    # Нормалізуємо дані для однакового масштабу
    max_mean = max(abs(x_mean1), abs(x_mean2))
    norm_means = [m/max_mean for m in means]
    norm_errors = [e/max_mean for e in errors]

    plt.barh(y_pos, norm_means, xerr=norm_errors, align='center',
             alpha=0.7, capsize=5, color=['blue', 'red'])

    # Додаємо значення та текст
    for i, (m, e) in enumerate(zip(means, errors)):
        plt.text(norm_means[i] + norm_errors[i] + 0.05, y_pos[i],
                 f'{m:.3f} ± {e:.3f}', va='center')

    plt.yticks(y_pos, labels)
    plt.xlabel('Нормалізовані значення')
    plt.grid(True, axis='x')

    # 4. Інформація про ітерації та видалені дані
    plt.subplot(2, 2, 4)
    plt.title('Підсумок аналізу')

    # Кількість видалених спостережень
    removed_count = len(data1) - len(current_data1)

    # Створюємо таблицю для відображення підсумків
    summary_labels = ['Показник', 'Значення']
    summary_data = [
        ['Вхідних спостережень', f'{len(data1)}'],
        ['Видалених спостережень', f'{removed_count}'],
        ['Відсоток видалених', f'{removed_count/len(data1)*100:.2f}%'],
        ['Кількість ітерацій', f'{iteration}'],
        ['Рівень довіри', f'{confidence}'],
        ['γp критичне', f'{get_critical_gamma(len(current_data1), confidence):.3f}']
    ]

    # Відображаємо таблицю
    summary_table = plt.table(cellText=summary_data,
                              colLabels=summary_labels,
                              cellLoc='center',
                              loc='center')
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(10)
    summary_table.scale(1, 1.5)
    plt.axis('off')

    # Збереження графіка
    plt.tight_layout()
    output_path = r'D:\WINDSURF\SCRIPTs\Стратегичні-Напрямки\1\результати_аналізу_два_стовпці.png'
    plt.savefig(output_path)
    plt.close()

    print(f"\nГрафік результатів збережено у файл '{output_path}'\n")

    return results

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

def main():
    """Основна функція програми"""

    # Параметри задаються прямо в скрипті
    file_path = "cleanest_data.csv"  # Шлях до CSV файлу з даними
    column1 = "partner_order_age_days"  # Перший стовпець для аналізу
    column2 = "partner_success_rate"  # Другий стовпець для аналізу
    delimiter = ","  # Розділювач у CSV файлі
    confidence = 0.95  # Рівень довіри
    max_iterations = 100  # Максимальна кількість ітерацій для виявлення викидів

    try:
        # Зчитування даних
        data1, data2, df = read_csv_data(file_path, column1, column2, delimiter)

        # Аналіз даних
        results = analyze_dual_data(data1, data2, column1, column2, confidence, max_iterations)

        print("\nАналіз виконано успішно і збережено візуалізацію даних")

    except Exception as e:
        print(f"Помилка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
