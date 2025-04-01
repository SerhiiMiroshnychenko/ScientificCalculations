#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Побудова лінійної регресійної моделі методом найменших квадратів
та оцінка її якості.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, r2_score
import os

# Налаштування графіків для відображення кирилиці
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

def generate_data(a0, a1, noise_range, n=51, x_min=0, x_max=10, seed=42):
    """
    Генерує дані згідно з лінійним рівнянням y = a0 + a1*x + випадковий_шум

    Параметри:
    a0 - вільний член
    a1 - коефіцієнт при x
    noise_range - максимальне значення випадкового шуму
    n - кількість точок
    x_min, x_max - діапазон значень x
    seed - зерно для генератора випадкових чисел

    Повертає:
    x - масив значень незалежної змінної
    y - масив значень залежної змінної
    y_true - масив точних значень без шуму
    """
    np.random.seed(seed)
    x = np.linspace(x_min, x_max, n)
    noise = np.random.uniform(-noise_range, noise_range, n)
    y_true = a0 + a1 * x
    y = y_true + noise
    return x, y, y_true

def linear_regression(x, y):
    """
    Обчислює параметри лінійної регресії методом найменших квадратів

    Параметри:
    x - масив значень незалежної змінної
    y - масив значень залежної змінної

    Повертає:
    b0 - оцінка вільного члена
    b1 - оцінка коефіцієнта при x
    r_value - коефіцієнт кореляції
    p_value - p-значення для перевірки статистичної значущості
    std_err - стандартна похибка оцінювання
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return intercept, slope, r_value, p_value, std_err

def calculate_determination_coefficient(y, y_pred):
    """
    Обчислює коефіцієнт детермінації R^2

    Параметри:
    y - фактичні значення
    y_pred - прогнозовані значення

    Повертає:
    r2 - коефіцієнт детермінації
    """
    return r2_score(y, y_pred)

def calculate_standard_error_regression(y, y_pred, n, p=2):
    """
    Обчислює стандартну похибку регресії (Standard Error of Regression)

    Параметри:
    y - фактичні значення
    y_pred - прогнозовані значення
    n - кількість спостережень
    p - кількість параметрів моделі (включаючи вільний член)

    Повертає:
    se - стандартна похибка регресії
    """
    sse = np.sum((y - y_pred) ** 2)
    return np.sqrt(sse / (n - p))

def calculate_confidence_intervals_coefficients(x, y, b0, b1, std_err, confidence_level=0.95):
    """
    Обчислює довірчі інтервали для коефіцієнтів регресії

    Параметри:
    x - масив значень незалежної змінної
    y - масив значень залежної змінної
    b0 - оцінка вільного члена
    b1 - оцінка коефіцієнта при x
    std_err - стандартна похибка оцінювання
    confidence_level - рівень довіри (за замовчуванням 0.95)

    Повертає:
    b0_conf_int - довірчий інтервал для b0
    b1_conf_int - довірчий інтервал для b1
    """
    n = len(x)
    dof = n - 2  # Ступені свободи
    t_critical = stats.t.ppf((1 + confidence_level) / 2, dof)

    # Обчислення середнього значення X і суми квадратів відхилень
    x_mean = np.mean(x)
    sse_x = np.sum((x - x_mean) ** 2)

    # Стандартні похибки для коефіцієнтів
    se_b1 = std_err / np.sqrt(sse_x)
    se_b0 = std_err * np.sqrt(1/n + x_mean**2/sse_x)

    # Довірчі інтервали
    b0_conf_int = (b0 - t_critical * se_b0, b0 + t_critical * se_b0)
    b1_conf_int = (b1 - t_critical * se_b1, b1 + t_critical * se_b1)

    return b0_conf_int, b1_conf_int

def calculate_confidence_intervals_prediction(x, x_new, y, b0, b1, std_err, confidence_level=0.95, prediction=False):
    """
    Обчислює довірчі інтервали для прогнозних значень

    Параметри:
    x - масив значень незалежної змінної навчальних даних
    x_new - нові значення незалежної змінної для прогнозу
    y - масив значень залежної змінної навчальних даних
    b0 - оцінка вільного члена
    b1 - оцінка коефіцієнта при x
    std_err - стандартна похибка оцінювання
    confidence_level - рівень довіри (за замовчуванням 0.95)
    prediction - True для інтервалу передбачення, False для довірчого інтервалу середнього

    Повертає:
    lower_bounds - нижні межі довірчих інтервалів
    upper_bounds - верхні межі довірчих інтервалів
    """
    n = len(x)
    dof = n - 2  # Ступені свободи
    t_critical = stats.t.ppf((1 + confidence_level) / 2, dof)

    # Обчислення середнього значення X і суми квадратів відхилень
    x_mean = np.mean(x)
    sse_x = np.sum((x - x_mean) ** 2)

    # Прогнозні значення
    y_pred = b0 + b1 * x_new

    # Масиви для зберігання меж інтервалів
    lower_bounds = np.zeros_like(x_new)
    upper_bounds = np.zeros_like(x_new)

    for i, xi in enumerate(x_new):
        # Стандартна похибка прогнозу
        if prediction:
            # Інтервал передбачення (для індивідуальних значень)
            se_pred = std_err * np.sqrt(1 + 1/n + (xi - x_mean)**2/sse_x)
        else:
            # Довірчий інтервал (для середніх значень)
            se_pred = std_err * np.sqrt(1/n + (xi - x_mean)**2/sse_x)

        # Межі довірчого інтервалу
        lower_bounds[i] = y_pred[i] - t_critical * se_pred
        upper_bounds[i] = y_pred[i] + t_critical * se_pred

    return lower_bounds, upper_bounds

def calculate_elasticity_coefficient(x, y, b1):
    """
    Обчислює усереднений коефіцієнт еластичності для лінійної моделі

    Параметри:
    x - масив значень незалежної змінної
    y - масив значень залежної змінної
    b1 - оцінка коефіцієнта при x

    Повертає:
    e - усереднений коефіцієнт еластичності
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return b1 * (x_mean / y_mean)

def perform_linear_regression_analysis():
    """
    Виконує повний аналіз лінійної регресії, включаючи:
    - Побудову моделі
    - Обчислення метрик якості
    - Візуалізацію результатів
    - Виведення статистичних показників
    """
    print("Аналіз лінійної регресії методом найменших квадратів\n")

    # Параметри моделі згідно з варіантом 4
    a0 = 3.4  # Вільний член
    a1 = 2.7  # Коефіцієнт при x
    noise_range = 1.10  # Діапазон випадкового шуму

    # Генерація даних
    n = 51  # Кількість точок
    x, y, y_true = generate_data(a0, a1, noise_range, n)

    # Побудова лінійної регресії
    b0, b1, r_value, p_value, std_err = linear_regression(x, y)

    # Обчислення прогнозних значень
    y_pred = b0 + b1 * x

    # Обчислення метрик якості моделі
    r2 = calculate_determination_coefficient(y, y_pred)
    ser = calculate_standard_error_regression(y, y_pred, n)
    b0_conf_int, b1_conf_int = calculate_confidence_intervals_coefficients(x, y, b0, b1, std_err)

    # Довірчі інтервали для прогнозу (для графіка)
    x_sorted = np.sort(x)
    conf_lower, conf_upper = calculate_confidence_intervals_prediction(x, x_sorted, y, b0, b1, std_err)
    pred_lower, pred_upper = calculate_confidence_intervals_prediction(x, x_sorted, y, b0, b1, std_err, prediction=True)

    # Коефіцієнт еластичності
    elasticity = calculate_elasticity_coefficient(x, y, b1)

    # Виведення результатів
    print(f"Справжнє рівняння: y = {a0} + {a1}*x + випадковий шум в діапазоні [-{noise_range}, {noise_range}]")
    print(f"Оцінене рівняння регресії: y = {b0:.4f} + {b1:.4f}*x")
    print("\nСтатистичні показники:")
    print(f"Коефіцієнт кореляції (r): {r_value:.4f}")
    print(f"Коефіцієнт детермінації (R²): {r2:.4f}")
    print(f"Стандартна похибка регресії: {ser:.4f}")
    print(f"p-значення: {p_value:.6f}")
    print(f"Стандартна похибка оцінювання: {std_err:.4f}")

    print("\nДовірчі інтервали для коефіцієнтів (95%):")
    print(f"Вільний член (b0): [{b0_conf_int[0]:.4f}, {b0_conf_int[1]:.4f}]")
    print(f"Коефіцієнт при x (b1): [{b1_conf_int[0]:.4f}, {b1_conf_int[1]:.4f}]")

    print(f"\nУсереднений коефіцієнт еластичності: {elasticity:.4f}")
    print("Це означає, що при зміні x на 1%, y в середньому змінюється на {:.2f}%".format(elasticity*100))

    # Створення таблиці з результатами
    results_df = pd.DataFrame({
        'x': x,
        'y фактичні': y,
        'y прогнозні': y_pred,
        'Залишки': y - y_pred
    })

    print("\nФрагмент таблиці з результатами:")
    print(tabulate(results_df.head(10), headers='keys', tablefmt='psql', floatfmt='.4f'))

    # Створення директорії для графіків, якщо вона не існує
    output_dir = 'regression_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Графік 1: Лінія регресії та точки
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, color='blue', label='Фактичні дані')
    plt.plot(x_sorted, b0 + b1 * x_sorted, color='red', label=f'Регресія: y = {b0:.3f} + {b1:.3f}*x')
    plt.plot(x, y_true, '--', color='green', label=f'Справжня функція: y = {a0} + {a1}*x')
    plt.fill_between(x_sorted, conf_lower, conf_upper, color='red', alpha=0.1, label='95% довірчий інтервал')
    plt.fill_between(x_sorted, pred_lower, pred_upper, color='blue', alpha=0.1, label='95% інтервал передбачення')
    plt.title('Лінійна регресійна модель')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_regression_line.png'), dpi=300)
    plt.close()

    # Графік 2: Залишки
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y - y_pred, color='orange')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Залишки vs x')
    plt.xlabel('x')
    plt.ylabel('Залишки')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_residuals.png'), dpi=300)
    plt.close()

    # Графік 3: Розподіл залишків
    plt.figure(figsize=(10, 8))
    sns.histplot(y - y_pred, kde=True, color='skyblue')
    plt.title('Розподіл залишків')
    plt.xlabel('Залишки')
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_residuals_distribution.png'), dpi=300)
    plt.close()

    # Графік 4: QQ-графік залишків
    plt.figure(figsize=(10, 8))
    stats.probplot(y - y_pred, dist="norm", plot=plt)
    plt.title('QQ-графік залишків')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_qq_plot.png'), dpi=300)
    plt.close()

    print(f"\nГрафіки збережено у директорію '{output_dir}':")
    print(f"1. Лінійна регресійна модель: {output_dir}/1_regression_line.png")
    print(f"2. Графік залишків: {output_dir}/2_residuals.png")
    print(f"3. Розподіл залишків: {output_dir}/3_residuals_distribution.png")
    print(f"4. QQ-графік залишків: {output_dir}/4_qq_plot.png")

    # Інтерпретація результатів
    print("\nІнтерпретація результатів:")

    # Інтерпретація коефіцієнта детермінації
    if r2 < 0.3:
        r2_interpretation = "дуже низька - модель погано описує дані"
    elif r2 < 0.5:
        r2_interpretation = "низька - модель слабко описує дані"
    elif r2 < 0.7:
        r2_interpretation = "середня - модель задовільно описує дані"
    elif r2 < 0.9:
        r2_interpretation = "висока - модель добре описує дані"
    else:
        r2_interpretation = "дуже висока - модель дуже добре описує дані"

    print(f"1. Якість моделі ({r2:.2f}): {r2_interpretation}")

    # Інтерпретація p-значення
    if p_value < 0.001:
        p_interpretation = "статистично дуже значуща (p < 0.001)"
    elif p_value < 0.01:
        p_interpretation = "статистично значуща (p < 0.01)"
    elif p_value < 0.05:
        p_interpretation = "статистично значуща (p < 0.05)"
    else:
        p_interpretation = "статистично не значуща (p > 0.05)"

    print(f"2. Модель є {p_interpretation}")

    # Інтерпретація коефіцієнтів
    print(f"3. Вільний член b0 = {b0:.4f} - очікуване значення y при x = 0")
    print(f"4. Коефіцієнт нахилу b1 = {b1:.4f} - при збільшенні x на 1 одиницю, y збільшується на {b1:.4f} одиниць")

    # Інтерпретація залишків
    mse = mean_squared_error(y, y_pred)
    print(f"5. Середньоквадратична похибка (MSE): {mse:.4f} - середній квадрат відхилень прогнозних значень від фактичних")

    # Інтерпретація довірчих інтервалів
    print("6. Довірчі інтервали для коефіцієнтів показують діапазон, в якому з 95% ймовірністю знаходяться справжні значення")

    # Перевірка на входження реальних параметрів в довірчі інтервали
    if b0_conf_int[0] <= a0 <= b0_conf_int[1]:
        print(f"   - Справжнє значення a0 = {a0} входить в довірчий інтервал [{b0_conf_int[0]:.4f}, {b0_conf_int[1]:.4f}]")
    else:
        print(f"   - Справжнє значення a0 = {a0} НЕ входить в довірчий інтервал [{b0_conf_int[0]:.4f}, {b0_conf_int[1]:.4f}]")

    if b1_conf_int[0] <= a1 <= b1_conf_int[1]:
        print(f"   - Справжнє значення a1 = {a1} входить в довірчий інтервал [{b1_conf_int[0]:.4f}, {b1_conf_int[1]:.4f}]")
    else:
        print(f"   - Справжнє значення a1 = {a1} НЕ входить в довірчий інтервал [{b1_conf_int[0]:.4f}, {b1_conf_int[1]:.4f}]")

    print("\nВисновок:")
    print("Ми побудували лінійну регресійну модель методом найменших квадратів.")
    print(f"Модель має коефіцієнт детермінації R² = {r2:.4f}, що означає, що вона пояснює {r2*100:.1f}% варіації залежної змінної.")

    if r2 > 0.7 and p_value < 0.05:
        print("Модель є адекватною і може використовуватися для прогнозування.")
    elif p_value < 0.05:
        print("Модель є статистично значущою, але має обмежену прогностичну здатність.")
    else:
        print("Модель не є статистично значущою і не рекомендується для прогнозування.")

if __name__ == "__main__":
    perform_linear_regression_analysis()
