#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Аналіз часових рядів з використанням показника Херста.
Обчислення показника Херста виконується за допомогою модуля hurst.
Графіки зберігаються у директорії './plots/' з відповідними назвами.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import os
from hurst import compute_Hc  # Використовуємо модуль hurst для обчислення показника Херста
from matplotlib.gridspec import GridSpec  # Додаємо імпорт GridSpec для створення складних графіків

# Налаштування для графіків, вимикаємо інтерактивний режим
mpl.use('Agg')  # Режим без відображення графіків, лише для збереження
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_time_series(t_values):
    """
    Створює часові ряди s1 і s2 за заданими формулами
    """
    # Константи
    A1 = 2.02
    t_hat1 = 1300
    
    # Формула для s1(t)
    term1 = 2 - ((t_values / t_hat1) - 1) ** 2
    # Перевірка на від'ємні значення під логарифмом
    valid_indices = term1 > 0
    s1 = np.zeros_like(t_values, dtype=float)
    s1[valid_indices] = np.log(A1 / term1[valid_indices])
    s1[~valid_indices] = np.nan  # Встановлюємо NaN для неможливих значень
    
    # Різні значення A2 для s2(t)
    A2_values = {
        '4.2': 10 ** 4.2,
        '5': 10 ** 5,
        '7': 10 ** 7
    }
    t_hat2 = 30
    
    # Словник для зберігання різних варіантів s2
    s2_dict = {}
    
    # Обчислення s2 для різних значень A2
    for key, A2 in A2_values.items():
        term2 = 2 + ((t_values / t_hat2) - 1) ** 2
        s2_dict[f'A2_10_{key}'] = np.log(A2 / term2)
    
    return s1, s2_dict

def analyze_derivatives(series, title, plot=True, save_path=None, suffix=''):
    """
    Аналізує похідні часового ряду та їх автокореляцію
    
    Параметри:
    series - часовий ряд для аналізу
    title - назва ряду для відображення
    plot - чи створювати графіки
    save_path - шлях для збереження графіків
    suffix - суфікс для назв файлів
    
    Повертає:
    df_stats - статистики похідних
    eq_check_stats - статистики перевірки диференціального рівняння
    eq_check - значення залишків рівняння
    """
    # Видалення NaN значень
    series_clean = series[~np.isnan(series)]
    
    # Обчислення похідних
    first_diff = np.diff(series_clean)
    second_diff = np.diff(first_diff)
    # Додаємо обчислення третьої похідної
    third_diff = np.diff(second_diff)
    
    # Статистика похідних
    derivatives_stats = {
        'Статистика': ['Середнє', 'Стандартне відхилення', 'Мінімум', 'Максимум'],
        'Перша похідна': [np.mean(first_diff), np.std(first_diff), np.min(first_diff), np.max(first_diff)],
        'Друга похідна': [np.mean(second_diff), np.std(second_diff), np.min(second_diff), np.max(second_diff)],
        'Третя похідна': [np.mean(third_diff), np.std(third_diff), np.min(third_diff), np.max(third_diff)]
    }
    
    df_stats = pd.DataFrame(derivatives_stats)
    
    # Обчислюємо автокореляцію
    max_lag = min(50, len(first_diff) // 4)
    acf_first = np.array([np.corrcoef(first_diff[:-i], first_diff[i:])[0, 1] 
                         for i in range(1, max_lag + 1)])
    acf_second = np.array([np.corrcoef(second_diff[:-i], second_diff[i:])[0, 1] 
                          for i in range(1, max_lag + 1)])
    acf_third = np.array([np.corrcoef(third_diff[:-i], third_diff[i:])[0, 1] 
                         for i in range(1, max_lag + 1)])
    
    # Перевірка виконання диференціального рівняння
    eq_check = third_diff - 3 * second_diff[:-1] * first_diff[:-2] + first_diff[:-2]**3
    eq_residual_mean = np.mean(np.abs(eq_check))
    eq_residual_std = np.std(np.abs(eq_check))
    
    if plot and save_path:
        # Графік часового ряду та його похідних
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # Оригінальний ряд
        axes[0].plot(series_clean, color='blue')
        axes[0].set_ylabel('Значення')
        axes[0].set_title(f'Часовий ряд {title}')
        axes[0].grid(True, alpha=0.3)
        
        # Перша похідна
        axes[1].plot(first_diff, color='green')
        axes[1].set_ylabel('Перша похідна')
        axes[1].grid(True, alpha=0.3)
        
        # Друга похідна
        axes[2].plot(second_diff, color='red')
        axes[2].set_ylabel('Друга похідна')
        axes[2].grid(True, alpha=0.3)
        
        # Третя похідна
        axes[3].plot(third_diff, color='purple')
        axes[3].set_ylabel('Третя похідна')
        axes[3].set_xlabel('Індекс')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Збереження графіка за новою логікою
        safe_title = title.replace('$', '').replace(' ', '_').replace('=', '_')
        fig_path = os.path.join(save_path, f"derivatives_{safe_title}{suffix}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()  # Закрити графік після збереження
        
        # Графік автокореляції
        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(1, max_lag + 1), acf_first, color='blue', 
                 label='Автокореляція першої похідної')
        plt.plot(np.arange(1, max_lag + 1), acf_second, color='green', 
                 label='Автокореляція другої похідної')
        plt.plot(np.arange(1, max_lag + 1), acf_third, color='red',
                 label='Автокореляція третьої похідної')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Лаг')
        plt.ylabel('Автокореляція')
        plt.title(f'Автокореляційна функція похідних для {title}')
        plt.legend()
        plt.tight_layout()
        
        # Збереження графіка автокореляції
        fig_path = os.path.join(save_path, f"autocorrelation_{safe_title}{suffix}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()  # Закрити графік після збереження
        
        # Графік перевірки диференціального рівняння
        plt.figure(figsize=(12, 6))
        plt.plot(eq_check, color='blue')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Індекс')
        plt.ylabel('Значення залишків')
        plt.title(f'Перевірка диференціального рівняння для {title}\n'
                  f'Середнє абс. значення залишків: {eq_residual_mean:.6f}, '
                  f'Стд. відхилення: {eq_residual_std:.6f}')
        plt.tight_layout()
        
        # Збереження графіка перевірки рівняння
        fig_path = os.path.join(save_path, f"equation_check_{safe_title}{suffix}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()  # Закрити графік після збереження
    
    # Додаємо інформацію про перевірку диференціального рівняння
    eq_check_stats = pd.DataFrame({
        'Статистика': ['Середнє абс. значення залишків', 'Стд. відхилення залишків'],
        'Значення': [eq_residual_mean, eq_residual_std]
    })
    
    return df_stats, eq_check_stats, eq_check

def analyze_time_series_hurst(series, title, methods=None, plot=True, save_path=None, suffix=''):
    """
    Проводить аналіз часового ряду різними методами обчислення показника Херста
    використовуючи модуль hurst та візуалізує результати як у файлі порівняння_методів_херста.py
    
    Параметри:
    series - часовий ряд для аналізу
    title - назва ряду для відображення
    methods - список методів для обчислення показника Херста
    plot - чи створювати графіки
    save_path - шлях для збереження графіків
    suffix - суфікс для назв файлів
    
    Повертає:
    results - словник з результатами обчислень для кожного методу
    """
    if methods is None:
        methods = ['price', 'change', 'random_walk']
    
    # Видаляємо NaN значення перед обчисленням
    series_clean = series[~np.isnan(series)]
    
    if len(series_clean) < 100:
        print(f"Недостатньо даних для обчислення показника Херста для {title}")
        return None
    
    # Створюємо словник для зберігання результатів
    results = {}
    
    # Кольори для різних методів
    colors = ["deepskyblue", "green", "red"]
    
    if plot:
        # Створюємо фігуру для графіків
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(2, 3, figure=fig)
        
        # Графік часового ряду
        ax_series = fig.add_subplot(gs[0, :3])
        ax_series.plot(np.arange(len(series_clean)), series_clean, color='purple')
        ax_series.set_title(f'Часовий ряд {title}')
        ax_series.set_xlabel('Індекс')
        ax_series.set_ylabel('Значення')
        ax_series.grid(True, alpha=0.3)
    
    # Обчислення показника Херста різними методами
    for i, method in enumerate(methods):
        # Використовуємо функцію compute_Hc з модуля hurst
        H, c, data = compute_Hc(series=series_clean, kind=method, simplified=False)
        results[method] = {'H': H, 'c': c, 'data': data}
        
        if plot:
            # Додаємо графік R/S аналізу
            ax = fig.add_subplot(gs[1, i])
            ax.plot(data[0], c*data[0]**H, color=colors[i], 
                    label=f'Регресійна лінія (H={H:.4f})')
            ax.scatter(data[0], data[1], color='purple', alpha=0.6, s=30, 
                       label='R/S значення')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Часовий інтервал')
            ax.set_ylabel('Відношення R/S')
            ax.set_title(f'R/S аналіз для {title} (метод: {method})')
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend()
    
    if plot:
        plt.tight_layout()
        
        # Збереження графіка
        if save_path:
            safe_title = title.replace('$', '').replace(' ', '_').replace('=', '_')
            fig_path = os.path.join(save_path, f"hurst_analysis_{safe_title}{suffix}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()  # Закрити графік після збереження
    
    # Виведення результатів
    print(f"\nРезультати для {title}:")
    for method in methods:
        H = results[method]['H']
        c = results[method]['c']
        
        print(f"Метод '{method}': H={H:.4f}, c={c:.4f}")
        
        # Інтерпретація результатів
        if H > 1:
            interpretation = "Надперсистентна поведінка (H > 1)"
        elif H > 0.5:
            interpretation = "Персистентна поведінка (0.5 < H < 1)"
        elif H == 0.5:
            interpretation = "Випадкове блукання (H = 0.5)"
        else:
            interpretation = "Антиперсистентна поведінка (H < 0.5)"
        
        print(f"  Інтерпретація: {interpretation}")
    
    return results

def main():
    """
    Головна функція, яка виконує аналіз часових рядів s1 і s2
    та зберігає результати у вигляді графіків
    """
    print("=" * 50)
    print("АНАЛІЗ ЧАСОВИХ РЯДІВ З ВИКОРИСТАННЯМ ПОКАЗНИКА ХЕРСТА")
    print("=" * 50)
    
    # Створення папки для збереження графіків за новою логікою
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Створено директорію для збереження графіків: {output_dir}")
    
    # Створення часового ряду
    print("Створення часових рядів...")
    t_values = np.arange(-100, 1501, 10)
    s1, s2_dict = create_time_series(t_values)
    
    # Аналіз часового ряду s1
    print("\n" + "=" * 50)
    print("АНАЛІЗ ЧАСОВОГО РЯДУ s1(t)")
    print("=" * 50)
    
    # Додаємо суфікс для s1
    suffix_s1 = "_s1"
    
    results_s1 = analyze_time_series_hurst(s1, "$s_1(t)$", plot=True, 
                                          save_path=output_dir, suffix=suffix_s1)
    
    # Аналіз похідних s1
    print("\nАналіз похідних s1(t)...")
    stats_s1, eq_stats_s1, eq_check_s1 = analyze_derivatives(s1, "$s_1(t)$", 
                                                           plot=True, save_path=output_dir, suffix=suffix_s1)
    
    print("\nСтатистики похідних для $s_1(t)$:")
    print(stats_s1.to_string(index=False))
    
    print("\nПеревірка диференціального рівняння для $s_1(t)$:")
    print(eq_stats_s1.to_string(index=False))
    
    # Аналіз часових рядів s2 для різних значень A2
    for i, (key, s2) in enumerate(s2_dict.items(), 1):
        print("\n" + "=" * 50)
        print(f"АНАЛІЗ ЧАСОВОГО РЯДУ s2(t) з {key}")
        print("=" * 50)
        
        # Використовуємо суфікс для унікальних імен файлів
        suffix = f"_s2_{i}"
        
        results_s2 = analyze_time_series_hurst(s2, f"$s_2(t)$ з {key}", 
                                              plot=True, save_path=output_dir, suffix=suffix)
        
        # Аналіз похідних s2
        print(f"\nАналіз похідних s2(t) з {key}...")
        stats_s2, eq_stats_s2, eq_check_s2 = analyze_derivatives(s2, f"$s_2(t)$ з {key}", 
                                                               plot=True, save_path=output_dir, suffix=suffix)
        
        print(f"\nСтатистики похідних для $s_2(t)$ з {key}:")
        print(stats_s2.to_string(index=False))
        
        print(f"\nПеревірка диференціального рівняння для $s_2(t)$ з {key}:")
        print(eq_stats_s2.to_string(index=False))
    
    print("\n" + "=" * 50)
    print("АНАЛІЗ ЗАВЕРШЕНО")
    print("=" * 50)
    print(f"Всі графіки збережено в директорії: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
