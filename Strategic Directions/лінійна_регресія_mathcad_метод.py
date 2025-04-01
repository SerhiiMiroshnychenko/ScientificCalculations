#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Побудова лінійної регресійної моделі методом найменших квадратів
на основі підходу, використаного в MathCad.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from tabulate import tabulate
import os
import seaborn as sns

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

def calculate_regression_coefficients_mathcad_method(x, y):
    """
    Обчислює параметри лінійної регресії методом найменших квадратів
    за формулами як у MathCad

    Параметри:
    x - масив значень незалежної змінної
    y - масив значень залежної змінної

    Повертає:
    a0 - оцінка вільного члена
    a1 - оцінка коефіцієнта при x
    x_mean - середнє значення x
    y_mean - середнє значення y
    x2_mean - середнє значення x^2
    xy_mean - середнє значення x*y
    """
    n = len(x)

    # Середні значення
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x2_mean = np.mean(x**2)
    xy_mean = np.mean(x * y)

    # Коефіцієнти регресії за формулами MathCad
    a1 = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean**2)
    a0 = y_mean - a1 * x_mean

    return a0, a1, x_mean, y_mean, x2_mean, xy_mean

def calculate_correlation_coefficient(x, y):
    """
    Обчислює коефіцієнт кореляції за формулою MathCad

    Параметри:
    x - масив значень незалежної змінної
    y - масив значень залежної змінної

    Повертає:
    r - коефіцієнт кореляції
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Чисельник: сума (x_i - x_mean)*(y_i - y_mean)
    numerator = np.sum((x - x_mean) * (y - y_mean)) / n

    # Знаменник: корінь з суми (x_i - x_mean)^2 * суми (y_i - y_mean)^2
    denominator = np.sqrt(np.sum((x - x_mean)**2) / n * np.sum((y - y_mean)**2) / n)

    r = numerator / denominator
    return r

def calculate_t_statistic(r, n):
    """
    Обчислює t-статистику для перевірки значущості коефіцієнта кореляції

    Параметри:
    r - коефіцієнт кореляції
    n - кількість спостережень

    Повертає:
    t_calculated - розрахункове значення t-статистики
    t_critical - критичне значення t-статистики для alpha = 0.05
    """
    # Розрахункове значення t-статистики
    t_calculated = r * np.sqrt((n - 2) / (1 - r**2))

    # Критичне значення t-статистики для alpha = 0.05
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha/2, n - 2)

    return t_calculated, t_critical

def calculate_determination_coefficient(y, y_pred, y_mean):
    """
    Обчислює коефіцієнт детермінації R^2 за формулою MathCad

    Параметри:
    y - фактичні значення
    y_pred - прогнозовані значення
    y_mean - середнє значення y

    Повертає:
    R2 - коефіцієнт детермінації
    """
    # Сума квадратів відхилень прогнозів від середнього
    tss = np.sum((y_pred - y_mean)**2)

    # Загальна сума квадратів відхилень
    ess = np.sum((y - y_mean)**2)

    # Коефіцієнт детермінації
    R2 = tss / ess

    return R2

def calculate_standard_error_regression(y, y_pred, n):
    """
    Обчислює стандартну похибку регресії за формулою MathCad

    Параметри:
    y - фактичні значення
    y_pred - прогнозовані значення
    n - кількість спостережень

    Повертає:
    S - стандартна похибка регресії
    """
    # Сума квадратів відхилень фактичних значень від прогнозованих
    sse = np.sum((y - y_pred)**2)

    # Стандартна похибка регресії
    S = np.sqrt(sse / (n - 2))

    return S

def calculate_f_statistic(R2, n):
    """
    Обчислює F-статистику для перевірки значущості регресії

    Параметри:
    R2 - коефіцієнт детермінації
    n - кількість спостережень

    Повертає:
    F_calculated - розрахункове значення F-статистики
    F_critical - критичне значення F-статистики для alpha = 0.05
    """
    # Розрахункове значення F-статистики
    F_calculated = R2 * (n - 2) / (1 - R2)

    # Критичне значення F-статистики для alpha = 0.05
    alpha = 0.05
    F_critical = stats.f.ppf(1 - alpha, 1, n - 2)

    return F_calculated, F_critical

def calculate_confidence_intervals_mathcad_method(x, y, a0, a1, S, n, alpha=0.05):
    """
    Обчислює довірчі інтервали для коефіцієнтів регресії за формулами MathCad

    Параметри:
    x - масив значень незалежної змінної
    y - масив значень залежної змінної
    a0 - оцінка вільного члена
    a1 - оцінка коефіцієнта при x
    S - стандартна похибка регресії
    n - кількість спостережень
    alpha - рівень значущості

    Повертає:
    S_a0 - стандартна похибка оцінки a0
    S_a1 - стандартна похибка оцінки a1
    a0_lower - нижня межа довірчого інтервалу для a0
    a0_upper - верхня межа довірчого інтервалу для a0
    a1_lower - нижня межа довірчого інтервалу для a1
    a1_upper - верхня межа довірчого інтервалу для a1
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Критичне значення t-статистики
    t_critical = stats.t.ppf(1 - alpha/2, n - 2)

    # Стандартні похибки коефіцієнтів
    sum_x2 = np.sum(x**2)
    sum_y2 = np.sum((y - y_mean)**2)
    sum_x_minus_mean_squared = np.sum((x - x_mean)**2)

    # Формули з MathCad
    S_a0 = np.sqrt((sum_x2 * sum_y2) / (n * (n - 2) * sum_x_minus_mean_squared))
    S_a1 = np.sqrt(sum_y2 / ((n - 2) * sum_x_minus_mean_squared))

    # Довірчі інтервали
    a0_lower = a0 - S_a0 * t_critical
    a0_upper = a0 + S_a0 * t_critical
    a1_lower = a1 - S_a1 * t_critical
    a1_upper = a1 + S_a1 * t_critical

    return S_a0, S_a1, a0_lower, a0_upper, a1_lower, a1_upper

def calculate_elasticity_coefficient(a1, x_mean, y_mean):
    """
    Обчислює усереднений коефіцієнт еластичності за формулою MathCad

    Параметри:
    a1 - оцінка коефіцієнта при x
    x_mean - середнє значення x
    y_mean - середнє значення y

    Повертає:
    elasticity - усереднений коефіцієнт еластичності
    """
    elasticity = a1 * (x_mean / y_mean)
    return elasticity

def perform_linear_regression_mathcad_method():
    """
    Виконує повний аналіз лінійної регресії методом MathCad, включаючи:
    - Побудову моделі
    - Обчислення метрик якості
    - Візуалізацію результатів
    - Виведення статистичних показників
    """
    print("Аналіз лінійної регресії методом найменших квадратів (підхід MathCad)\n")

    # Параметри моделі згідно з варіантом 4
    a0 = 3.4  # Вільний член
    a1 = 2.7  # Коефіцієнт при x
    noise_range = 1.10  # Діапазон випадкового шуму

    # Генерація даних
    n = 51  # Кількість точок
    x, y, y_true = generate_data(a0, a1, noise_range, n)

    # Побудова лінійної регресії за методом MathCad
    b0, b1, x_mean, y_mean, x2_mean, xy_mean = calculate_regression_coefficients_mathcad_method(x, y)

    # Обчислення прогнозних значень
    y_pred = b0 + b1 * x

    # Коефіцієнт кореляції
    r = calculate_correlation_coefficient(x, y)

    # t-статистика для перевірки значущості коефіцієнта кореляції
    t_calc, t_crit = calculate_t_statistic(r, n)

    # Коефіцієнт детермінації
    R2 = calculate_determination_coefficient(y, y_pred, y_mean)

    # Стандартна похибка регресії
    S = calculate_standard_error_regression(y, y_pred, n)

    # F-статистика для перевірки значущості регресії
    F_calc, F_crit = calculate_f_statistic(R2, n)

    # Довірчі інтервали для коефіцієнтів
    S_a0, S_a1, a0_lower, a0_upper, a1_lower, a1_upper = calculate_confidence_intervals_mathcad_method(x, y, b0, b1, S, n)

    # Коефіцієнт еластичності
    elasticity = calculate_elasticity_coefficient(b1, x_mean, y_mean)

    # Створення директорії для графіків, якщо вона не існує
    output_dir = 'mathcad_method_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Виведення основних результатів
    print(f"Справжнє рівняння: y = {a0} + {a1}*x + випадковий шум в діапазоні [-{noise_range}, {noise_range}]")
    print(f"Оцінене рівняння регресії: y = {b0:.4f} + {b1:.4f}*x")

    print("\nСередні значення (як у MathCad):")
    print(f"Середнє значення y (y_ср): {y_mean:.4f}")
    print(f"Середнє значення x (x_ср): {x_mean:.4f}")
    print(f"Середнє значення x² (x2_ср): {x2_mean:.4f}")
    print(f"Середнє значення xy (xy_ср): {xy_mean:.4f}")

    print("\nРозрахунок коефіцієнтів (як у MathCad):")
    print(f"a₁ = (xy_ср - x_ср·y_ср) / (x2_ср - x_ср²) = {b1:.4f}")
    print(f"a₀ = y_ср - a₁·x_ср = {b0:.4f}")

    print("\nСтатистичні показники:")
    print(f"Коефіцієнт кореляції (r): {r:.4f}")
    print(f"t-статистика: t_розрах = {t_calc:.4f}, t_крит = {t_crit:.4f}")
    conclusion = "t_розрах > t_крит, зв'язок статистично значущий" if t_calc > t_crit else "t_розрах < t_крит, зв'язок статистично не значущий"
    print(f"  Висновок: {conclusion}")

    print(f"Коефіцієнт детермінації (R²): {R2:.4f}")
    print(f"  Якість моделі: {'модель вважається точною' if R2 > 0.8 else 'модель незадовільна' if R2 < 0.5 else 'модель задовільна'}")

    print(f"Стандартна похибка регресії (S): {S:.4f}")

    print(f"F-статистика: F_розрах = {F_calc:.4f}, F_крит = {F_crit:.4f}")
    print(f"  Висновок: {'F_розрах > F_крит, існує лінійна регресія між показниками x і y' if F_calc > F_crit else 'F_розрах < F_крит, лінійна регресія статистично не значуща'}")

    print("\nДовірчі інтервали для коефіцієнтів (95%):")
    print(f"Стандартна похибка a₀ (S_a0): {S_a0:.4f}")
    print(f"Стандартна похибка a₁ (S_a1): {S_a1:.4f}")
    print(f"Вільний член (a₀): [{a0_lower:.4f}, {a0_upper:.4f}]")
    print(f"Коефіцієнт при x (a₁): [{a1_lower:.4f}, {a1_upper:.4f}]")

    print(f"\nУсереднений коефіцієнт еластичності: {elasticity:.4f}")
    print(f"При зміні x на 1%, y в середньому змінюється на {elasticity*100:.2f}%")

    # Створення таблиці з результатами
    results_df = pd.DataFrame({
        'x': x,
        'y фактичні': y,
        'y прогнозні': y_pred,
        'Залишки': y - y_pred
    })

    print("\nФрагмент таблиці з результатами:")
    print(tabulate(results_df.head(10), headers='keys', tablefmt='psql', floatfmt='.4f'))

    # Візуалізація регресійної моделі
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, color='blue', label='Фактичні дані')
    plt.plot(np.sort(x), b0 + b1 * np.sort(x), color='red', label=f'Регресія: y = {b0:.4f} + {b1:.4f}*x')
    plt.plot(x, y_true, '--', color='green', label=f'Справжня функція: y = {a0} + {a1}*x')
    plt.title('Лінійна регресійна модель (метод MathCad)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_regression_model_mathcad.png'), dpi=300)
    plt.close()

    # Графік 2: Залишки
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y - y_pred, color='orange')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Залишки vs x (метод MathCad)')
    plt.xlabel('x')
    plt.ylabel('Залишки')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_residuals_mathcad.png'), dpi=300)
    plt.close()

    # Графік 3: Розподіл залишків
    plt.figure(figsize=(10, 8))
    sns.histplot(y - y_pred, kde=True, color='skyblue')
    plt.title('Розподіл залишків (метод MathCad)')
    plt.xlabel('Залишки')
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_residuals_distribution_mathcad.png'), dpi=300)
    plt.close()

    # Графік 4: QQ-графік залишків
    plt.figure(figsize=(10, 8))
    stats.probplot(y - y_pred, dist="norm", plot=plt)
    plt.title('QQ-графік залишків (метод MathCad)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_qq_plot_mathcad.png'), dpi=300)
    plt.close()

    print("\nГрафіки збережено у директорію:", output_dir)
    print(f"1. Лінійна регресійна модель: {output_dir}/1_regression_model_mathcad.png")
    print(f"2. Графік залишків: {output_dir}/2_residuals_mathcad.png")
    print(f"3. Розподіл залишків: {output_dir}/3_residuals_distribution_mathcad.png")
    print(f"4. QQ-графік залишків: {output_dir}/4_qq_plot_mathcad.png")

    print("\nПеревірка чи справжні значення входять в довірчі інтервали:")
    if a0_lower <= a0 <= a0_upper:
        print(f"   - Справжнє значення a0 = {a0} входить в довірчий інтервал [{a0_lower:.4f}, {a0_upper:.4f}]")
    else:
        print(f"   - Справжнє значення a0 = {a0} НЕ входить в довірчий інтервал [{a0_lower:.4f}, {a0_upper:.4f}]")

    if a1_lower <= a1 <= a1_upper:
        print(f"   - Справжнє значення a1 = {a1} входить в довірчий інтервал [{a1_lower:.4f}, {a1_upper:.4f}]")
    else:
        print(f"   - Справжнє значення a1 = {a1} НЕ входить в довірчий інтервал [{a1_lower:.4f}, {a1_upper:.4f}]")

    print("\nВисновки за методом MathCad:")
    print(f"1. Побудоване рівняння регресії: y = {b0:.4f} + {b1:.4f}*x")
    print(f"2. Коефіцієнт детермінації R² = {R2:.4f} {'> 0.8, модель вважається точною' if R2 > 0.8 else '< 0.5, модель незадовільна' if R2 < 0.5 else ', модель задовільна'}")
    print(f"3. Кореляційний зв'язок між x і y: r = {r:.4f} {'статистично значущий (p < 0.05)' if t_calc > t_crit else 'статистично не значущий'}")
    print(f"4. Регресійна модель {'значуща (p < 0.05)' if F_calc > F_crit else 'не значуща'} за F-критерієм Фішера")
    print(f"5. Коефіцієнт еластичності {elasticity:.4f} показує, що при зміні x на 1%, y змінюється на {elasticity*100:.2f}%")

    if R2 > 0.8 and F_calc > F_crit:
        print("6. Модель є адекватною і може використовуватися для прогнозування.")
    elif F_calc > F_crit:
        print("6. Модель є статистично значущою, але має обмежену прогностичну здатність.")
    else:
        print("6. Модель не є статистично значущою і не рекомендується для прогнозування.")

if __name__ == "__main__":
    perform_linear_regression_mathcad_method()
