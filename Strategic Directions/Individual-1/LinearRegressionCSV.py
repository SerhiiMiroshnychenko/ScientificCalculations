#!/usr/bin/env python
# coding: utf-8

"""
Лінійний парний регресійний аналіз даних з CSV-файлу.
Цей скрипт дозволяє виконувати лінійний парний регресійний аналіз даних, 
зчитаних з CSV-файлу, та оцінювати метричні показники моделі.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys

def print_header(message):
    """Виводить заголовок розділу"""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)

def read_csv_data(file_path, column_x, column_y, delimiter=','):
    """
    Зчитує дані з CSV файлу.
    
    Args:
        file_path (str): Шлях до CSV файлу
        column_x (str/int): Назва або індекс стовпця для незалежної змінної (x)
        column_y (str/int): Назва або індекс стовпця для залежної змінної (y)
        delimiter (str): Розділювач у CSV файлі (за замовчуванням кома)
        
    Returns:
        tuple: (x_data, y_data, dataframe) - набори даних x та y, а також вся таблиця
    """
    try:
        # Зчитування CSV файлу
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        # Отримання даних для x
        if isinstance(column_x, int):
            if column_x < len(df.columns):
                x_data = df.iloc[:, column_x]
                x_name = df.columns[column_x]
            else:
                raise ValueError(f"Індекс стовпця {column_x} перевищує кількість стовпців")
        else:
            if column_x in df.columns:
                x_data = df[column_x]
                x_name = column_x
            else:
                raise ValueError(f"Стовпець '{column_x}' не знайдено в файлі. Доступні стовпці: {', '.join(df.columns)}")
                
        # Отримання даних для y
        if isinstance(column_y, int):
            if column_y < len(df.columns):
                y_data = df.iloc[:, column_y]
                y_name = df.columns[column_y]
            else:
                raise ValueError(f"Індекс стовпця {column_y} перевищує кількість стовпців")
        else:
            if column_y in df.columns:
                y_data = df[column_y]
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
        
        return x_data.values, y_data.values, df, x_name, y_name
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл '{file_path}' не знайдено")
    except Exception as e:
        raise Exception(f"Помилка при зчитуванні даних: {str(e)}")

def calculate_regression_parameters(x, y):
    """
    Розраховує параметри лінійної регресії методом найменших квадратів.
    
    Args:
        x (numpy.ndarray): Масив незалежних змінних
        y (numpy.ndarray): Масив залежних змінних
    
    Returns:
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
    
    Args:
        x (numpy.ndarray): Масив незалежних змінних
        y (numpy.ndarray): Масив залежних змінних
        a0 (float): Вільний член рівняння регресії
        a1 (float): Коефіцієнт нахилу
    
    Returns:
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
        "se_a0": se_a0,
        "se_a1": se_a1,
        "t_a0": t_a0,
        "t_a1": t_a1,
        "t_critical": t_critical,
        "f_statistic": f_statistic,
        "f_critical": f_critical,
        "elasticity": elasticity,
        "a0_lower": a0_lower,
        "a0_upper": a0_upper,
        "a1_lower": a1_lower,
        "a1_upper": a1_upper,
        "residuals": residuals,
        "y_pred": y_pred
    }
    
    return metrics

def print_regression_results(x, y, x_name, y_name, metrics):
    """
    Виводить результати регресійного аналізу.
    
    Args:
        x (numpy.ndarray): Масив незалежних змінних
        y (numpy.ndarray): Масив залежних змінних
        x_name (str): Назва незалежної змінної
        y_name (str): Назва залежної змінної
        metrics (dict): Словник з метриками якості моделі
    """
    a0 = metrics["a0"]
    a1 = metrics["a1"]
    
    print_header("ОСНОВНІ РЕЗУЛЬТАТИ РЕГРЕСІЙНОГО АНАЛІЗУ")
    
    print(f"Кількість спостережень: {metrics['n']}")
    print(f"Рівняння регресії: {y_name} = {a0:.4f} + {a1:.4f} * {x_name}")
    print(f"Коефіцієнт кореляції (r): {metrics['r_xy']:.4f}")
    print(f"Коефіцієнт детермінації (R²): {metrics['r_squared']:.4f}")
    print(f"Скоригований R²: {metrics['adj_r_squared']:.4f}")
    print(f"Стандартна похибка регресії: {metrics['se_regression']:.4f}")
    
    print_header("АНАЛІЗ КОЕФІЦІЄНТІВ РЕГРЕСІЇ")
    
    print("Вільний член (a₀):")
    print(f"  Значення: {a0:.4f}")
    print(f"  Стандартна похибка: {metrics['se_a0']:.4f}")
    print(f"  t-статистика: {metrics['t_a0']:.4f}")
    print(f"  Довірчий інтервал (95%): [{metrics['a0_lower']:.4f}, {metrics['a0_upper']:.4f}]")
    print(f"  Статистична значущість: {'Значущий' if abs(metrics['t_a0']) > metrics['t_critical'] else 'Незначущий'}")
    
    print("\nКоефіцієнт нахилу (a₁):")
    print(f"  Значення: {a1:.4f}")
    print(f"  Стандартна похибка: {metrics['se_a1']:.4f}")
    print(f"  t-статистика: {metrics['t_a1']:.4f}")
    print(f"  Довірчий інтервал (95%): [{metrics['a1_lower']:.4f}, {metrics['a1_upper']:.4f}]")
    print(f"  Статистична значущість: {'Значущий' if abs(metrics['t_a1']) > metrics['t_critical'] else 'Незначущий'}")
    
    print_header("ЗАГАЛЬНА ЯКІСТЬ МОДЕЛІ")
    
    print(f"F-статистика: {metrics['f_statistic']:.4f}")
    print(f"Критичне значення F-розподілу (α = 0.05): {metrics['f_critical']:.4f}")
    print(f"Загальна значущість моделі: {'Значуща' if metrics['f_statistic'] > metrics['f_critical'] else 'Незначуща'}")
    
    print_header("ДОДАТКОВІ ПОКАЗНИКИ")
    
    print(f"Усереднений коефіцієнт еластичності: {metrics['elasticity']:.4f}")
    print(f"Інтерпретація: При зміні {x_name} на 1%, {y_name} змінюється в середньому на {metrics['elasticity']:.4f}%")

def plot_regression_results(x, y, x_name, y_name, metrics):
    """
    Створює візуалізації результатів регресійного аналізу.
    
    Args:
        x (numpy.ndarray): Масив незалежних змінних
        y (numpy.ndarray): Масив залежних змінних
        x_name (str): Назва незалежної змінної
        y_name (str): Назва залежної змінної
        metrics (dict): Словник з метриками якості моделі
    """
    a0 = metrics["a0"]
    a1 = metrics["a1"]
    y_pred = metrics["y_pred"]
    residuals = metrics["residuals"]
    n = metrics["n"]
    
    # Налаштування стилю графіків
    sns.set_theme(style="whitegrid")
    
    # 1. Графік даних та лінії регресії
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', marker='.', label='Спостереження')
    
    # Створення лінії регресії на всьому діапазоні x
    x_line = np.linspace(min(x), max(x), 100)
    y_line = a0 + a1 * x_line
    plt.plot(x_line, y_line, color='blue', label=f'Лінія регресії: y = {a0:.4f} + {a1:.4f}x')
    
    # Розрахунок довірчих інтервалів для лінії регресії
    t_critical = metrics["t_critical"]
    se_regression = metrics["se_regression"]
    x_mean = np.mean(x)
    sum_sq_x = np.sum((x - x_mean)**2)
    
    # Побудова 95% довірчих інтервалів для лінії регресії
    se_fit = se_regression * np.sqrt(1/n + (x_line - x_mean)**2 / sum_sq_x)
    
    # Додаємо довірчі інтервали для лінії регресії
    plt.fill_between(x_line, y_line - t_critical * se_fit, y_line + t_critical * se_fit,
                     color='blue', alpha=0.2, label='95% довірчий інтервал для лінії регресії')
    
    # Довірчі інтервали для прогнозу
    se_pred = se_regression * np.sqrt(1 + 1/n + (x_line - x_mean)**2 / sum_sq_x)
    plt.fill_between(x_line, y_line - t_critical * se_pred, y_line + t_critical * se_pred,
                     color='gray', alpha=0.2, label='95% довірчий інтервал для прогнозу')
    
    plt.xlabel(f'{x_name}', fontsize=12)
    plt.ylabel(f'{y_name}', fontsize=12)
    plt.title('Лінійна регресійна модель', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 2. Графік залишків
    plt.figure(figsize=(10, 6))
    plt.scatter(x, residuals, color='green', marker='.')
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    # Довірчі межі для залишків (±2σ)
    plt.axhline(y=2*se_regression, color='red', linestyle='--', alpha=0.7, label='+2σ')
    plt.axhline(y=-2*se_regression, color='red', linestyle='--', alpha=0.7, label='-2σ')
    
    plt.xlabel(f'{x_name}', fontsize=12)
    plt.ylabel('Залишки', fontsize=12)
    plt.title('Залишки регресійної моделі', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 3. Гістограма розподілу залишків
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='blue')
    
    # Додаємо теоретичну криву нормального розподілу
    x_norm = np.linspace(min(residuals), max(residuals), 100)
    y_norm = stats.norm.pdf(x_norm, 0, se_regression)
    plt.plot(x_norm, y_norm * len(residuals) * (max(residuals) - min(residuals)) / 10, 
             'r-', linewidth=2, label='Нормальний розподіл')
    
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.3)
    plt.xlabel('Залишки', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.title('Розподіл залишків', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 4. Q-Q графік для перевірки нормальності розподілу залишків
    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q графік залишків', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 5. Відстань Кука для виявлення впливових спостережень
    leverage = (x - x_mean)**2 / sum_sq_x
    h_ii = 1/n + leverage
    cooks_d = residuals**2 / (metrics["se_regression"]**2 * (n - 2)) * (h_ii / (1 - h_ii)**2)
    
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(cooks_d)), cooks_d, markerfmt='.', linefmt='b-')
    plt.axhline(y=4/(n-2), color='red', linestyle='--', label='Порогове значення')
    
    # Виявлення впливових спостережень
    influential = np.where(cooks_d > 4/(n-2))[0]
    if len(influential) > 0:
        plt.scatter(influential, cooks_d[influential], color='red', s=80, label='Впливові спостереження')
    
    plt.xlabel('Індекс спостереження', fontsize=12)
    plt.ylabel('Відстань Кука', fontsize=12)
    plt.title('Відстань Кука для виявлення впливових спостережень', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 6. Порівняльний графік реальних та прогнозованих значень
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, color='purple', marker='.')
    
    # Додаємо діагональну лінію ідеального прогнозу
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ідеальний прогноз')
    
    plt.xlabel(f'Фактичні значення {y_name}', fontsize=12)
    plt.ylabel(f'Прогнозовані значення {y_name}', fontsize=12)
    plt.title('Порівняння фактичних і прогнозованих значень', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_linear_regression(x, y, x_name, y_name):
    """
    Проводить повний аналіз лінійної регресії.
    
    Args:
        x (numpy.ndarray): Масив незалежних змінних
        y (numpy.ndarray): Масив залежних змінних
        x_name (str): Назва незалежної змінної
        y_name (str): Назва залежної змінної
    
    Returns:
        dict: Результати аналізу
    """
    # Розрахунок параметрів регресії
    a0, a1 = calculate_regression_parameters(x, y)
    
    # Розрахунок метрик якості моделі
    metrics = calculate_regression_metrics(x, y, a0, a1)
    
    # Виведення результатів аналізу
    print_regression_results(x, y, x_name, y_name, metrics)
    
    # Візуалізація результатів
    plot_regression_results(x, y, x_name, y_name, metrics)
    
    return {
        "parameters": {"a0": a0, "a1": a1},
        "metrics": metrics
    }

def main():
    """Основна функція програми"""
    
    # Параметри задаються прямо в скрипті
    file_path = "cleanest_data.csv"  # Шлях до CSV файлу з даними
    column_x = "partner_total_orders"  # Стовпець з незалежною змінною (X)
    column_y = "partner_total_messages"  # Стовпець з залежною змінною (Y)
    delimiter = ","  # Розділювач у CSV файлі
    
    try:
        # Зчитування даних
        x_data, y_data, df, x_name, y_name = read_csv_data(file_path, column_x, column_y, delimiter)
        
        # Аналіз лінійної регресії
        results = analyze_linear_regression(x_data, y_data, x_name, y_name)
        
        print("\nАналіз виконано успішно")
        
    except Exception as e:
        print(f"Помилка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Налаштування стилю графіків
    sns.set_theme(style="whitegrid")
    main()
