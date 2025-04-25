import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Параметри з файлу "Опис_системи.md"
A1 = 2.02
t_hat1 = 1300  # секунди

A2_values = [10**(4.2), 10**5, 10**7]
t_hat2 = 30  # секунди для всіх варіантів A2

# Генерація часових точок
t = np.arange(-100, 1501, 10)  # від -100 до 1500 з кроком 10

# Функції для розрахунку s1 і s2
def s1(t):
    return np.log(A1 / (2 - ((t / t_hat1) - 1)**2))

def s2(t, A2):
    return np.log(A2 / (2 + ((t / t_hat2) - 1)**2))

# Розрахунок часових рядів
s1_values = s1(t)
s2_values_list = [s2(t, A2) for A2 in A2_values]

# Обробка можливих нескінченностей та NaN-ів
s1_values = np.nan_to_num(s1_values, nan=0)
for i in range(len(s2_values_list)):
    s2_values_list[i] = np.nan_to_num(s2_values_list[i], nan=0)

# Функція для обчислення похідної
def compute_derivative(series):
    derivative = np.diff(series)
    # Додаємо значення в кінці для збереження розмірності
    derivative = np.append(derivative, derivative[-1])
    return derivative

# Функція для обчислення похідних до заданого порядку
def compute_derivatives_up_to_order(series, max_order=3):
    derivatives = [series]  # 0-й порядок - сам ряд
    
    # Обчислюємо похідні послідовно
    current_series = series
    for order in range(1, max_order + 1):
        current_series = compute_derivative(current_series)
        derivatives.append(current_series)
    
    return derivatives

# Обчислення похідних часових рядів до 3-го порядку
s1_derivatives = compute_derivatives_up_to_order(s1_values, 3)
s2_derivatives_list = [compute_derivatives_up_to_order(s2, 3) for s2 in s2_values_list]

# Функція для обчислення автокореляційної функції
def calculate_autocorr(series, max_lag=50):
    # Обчислення автокореляційної функції
    return acf(series, nlags=max_lag, fft=True)

# Максимальний лаг для автокореляції
max_lag = 50

# Обчислення автокореляцій для s1 та його похідних
s1_autocorrs = []
for order in range(1, 4):  # 1, 2, 3 порядок (пропускаємо 0-й порядок - сам ряд)
    autocorr = calculate_autocorr(s1_derivatives[order], max_lag)
    s1_autocorrs.append(autocorr)

# Обчислення автокореляцій для s2 та його похідних
s2_autocorrs_list = []
for i, A2 in enumerate(A2_values):
    s2_autocorrs = []
    for order in range(1, 4):  # 1, 2, 3 порядок
        autocorr = calculate_autocorr(s2_derivatives_list[i][order], max_lag)
        s2_autocorrs.append(autocorr)
    s2_autocorrs_list.append(s2_autocorrs)

# Параметри для графіків
colors = ['r', 'g', 'b']
labels = ['Перша похідна', 'Друга похідна', 'Третя похідна']

# Візуалізація автокореляційних функцій для s1 (усі три похідні на одному графіку)
fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
fig.suptitle('Автокореляційні функції похідних s1(t)', fontsize=16)

for i in range(3):
    ax = axes[i]
    lags = np.arange(len(s1_autocorrs[i]))
    ax.stem(lags, s1_autocorrs[i], colors[i] + '-', markerfmt=colors[i] + 'o', basefmt='k-')
    ax.axhline(y=0, linestyle='-', alpha=0.3, color='gray')
    
    # Довірчі інтервали
    conf_interval = 1.96 / np.sqrt(len(t))
    ax.axhline(y=conf_interval, linestyle='--', alpha=0.3, color='gray')
    ax.axhline(y=-conf_interval, linestyle='--', alpha=0.3, color='gray')
    
    ax.set_title(f'{i+1}-а похідна s1(t)')
    ax.set_ylabel('Автокореляція')
    ax.grid(True, alpha=0.3)
    
    # Додаємо x-мітку тільки для нижнього графіка
    if i == 2:
        ax.set_xlabel('Лаг')

plt.tight_layout()
plt.subplots_adjust(top=0.92)

# Візуалізація автокореляційних функцій для кожного з трьох s2 (усі три похідні на одному графіку)
for i, A2 in enumerate(A2_values):
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f'Автокореляційні функції похідних s2(t) з A2=10^{np.log10(A2):.1f}', fontsize=16)
    
    for j in range(3):
        ax = axes[j]
        lags = np.arange(len(s2_autocorrs_list[i][j]))
        ax.stem(lags, s2_autocorrs_list[i][j], colors[j] + '-', 
                markerfmt=colors[j] + 'o', basefmt='k-')
        ax.axhline(y=0, linestyle='-', alpha=0.3, color='gray')
        
        # Довірчі інтервали
        conf_interval = 1.96 / np.sqrt(len(t))
        ax.axhline(y=conf_interval, linestyle='--', alpha=0.3, color='gray')
        ax.axhline(y=-conf_interval, linestyle='--', alpha=0.3, color='gray')
        
        ax.set_title(f'{j+1}-а похідна s2(t) з A2=10^{np.log10(A2):.1f}')
        ax.set_ylabel('Автокореляція')
        ax.grid(True, alpha=0.3)
        
        # Додаємо x-мітку тільки для нижнього графіка
        if j == 2:
            ax.set_xlabel('Лаг')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

# Порівняльний графік з усіма рядами на одному великому графіку з підграфіками
fig, axes = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
fig.suptitle('Порівняння автокореляційних функцій похідних', fontsize=18)

# Кольори та стилі для різних рядів даних
colors_s2 = ['g', 'b', 'c']
line_styles = ['-', '--', '-.', ':']

for i in range(3):  # 0, 1, 2 індекси для 1-ї, 2-ї, 3-ї похідних
    ax = axes[i]
    
    # Додаємо s1
    ax.plot(np.arange(max_lag+1), s1_autocorrs[i][:max_lag+1], 'r' + line_styles[0], 
            label=f's1(t)', alpha=0.8, linewidth=1.5)
    
    # Додаємо всі s2
    for j, A2 in enumerate(A2_values):
        ax.plot(np.arange(max_lag+1), s2_autocorrs_list[j][i][:max_lag+1], 
                colors_s2[j] + line_styles[j+1], 
                label=f's2(t), A2=10^{np.log10(A2):.1f}', alpha=0.8, linewidth=1.5)
    
    # Довірчий інтервал
    conf_interval = 1.96 / np.sqrt(len(t))
    ax.axhline(y=conf_interval, linestyle='--', alpha=0.3, color='gray')
    ax.axhline(y=-conf_interval, linestyle='--', alpha=0.3, color='gray')
    ax.axhline(y=0, linestyle='-', alpha=0.3, color='gray')
    
    ax.set_title(f'Порівняння автокореляційних функцій {i+1}-х похідних')
    ax.set_ylabel('Автокореляція')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Додаємо x-мітку тільки для нижнього графіка
    if i == 2:
        ax.set_xlabel('Лаг')

plt.tight_layout()
plt.subplots_adjust(top=0.95)

plt.show()