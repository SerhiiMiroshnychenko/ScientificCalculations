import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc
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
s1_values_clean = np.nan_to_num(s1_values, nan=0)
s2_values_list_clean = []
for i in range(len(s2_values_list)):
    s2_values_list_clean.append(np.nan_to_num(s2_values_list[i], nan=0))

# =========== ВІЗУАЛІЗАЦІЯ ЧАСОВИХ РЯДІВ ===========
plt.figure(figsize=(12, 8))
plt.plot(t, s1_values, label='s1(t), A1=2.02, t_hat1=1300s')
for i, A2 in enumerate(A2_values):
    plt.plot(t, s2_values_list[i], label=f's2(t), A2=10^{np.log10(A2):.1f}, t_hat2=30s')

plt.title('Часові ряди s1(t) та s2(t)')
plt.xlabel('Час (с)')
plt.ylabel('Значення s(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# =========== ЧАСТИНА 1: АНАЛІЗ ПОКАЗНИКА ХЕРСТА ===========
print("Розрахунок показника Герста:")

# Функція для розрахунку і виведення показника Герста
def calculate_hurst(series, name):
    if len(series) > 100:  # Перевіряємо, чи достатньо даних
        H, c, data = compute_Hc(series, kind='price', simplified=False)
        print(f"{name}: H={H:.4f}, c={c:.4f}")
        return H, c, data
    else:
        print(f"{name}: Недостатньо даних для розрахунку (після видалення NaN)")
        return None, None, None

# Розрахунок для s1
H1, c1, data1 = calculate_hurst(s1_values, "s1(t)")

# Розрахунок для s2 з різними A2
H2_list = []
c2_list = []
data2_list = []

for i, A2 in enumerate(A2_values):
    H2, c2, data2 = calculate_hurst(s2_values_list[i], f"s2(t) з A2=10^{np.log10(A2):.1f}")
    H2_list.append(H2)
    c2_list.append(c2)
    data2_list.append(data2)

# Кольори та маркери для різних рядів у R/S аналізі
colors = ['r', 'g', 'b', 'c']
markers = ['o', 's', '^', 'd']

# Візуалізація результатів R/S аналізу на одному графіку з підграфіками
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Аналіз показника Херста для часових рядів', fontsize=18)

# Підграфік для s1
ax = axes[0, 0]
if H1 is not None:
    ax.loglog(data1[0], c1*data1[0]**H1, 'b-', label=f'Регресія (H={H1:.4f})')
    ax.loglog(data1[0], data1[1], 'ro', label='R/S значення')
    ax.set_title('R/S аналіз для s1(t)')
    ax.set_xlabel('Часовий інтервал')
    ax.set_ylabel('R/S відношення')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)

# Підграфіки для s2 з різними A2
for i, A2 in enumerate(A2_values):
    # Визначаємо позицію підграфіка
    if i == 0:
        ax = axes[0, 1]
    elif i == 1:
        ax = axes[1, 0]
    else:
        ax = axes[1, 1]
        
    if H2_list[i] is not None:
        ax.loglog(data2_list[i][0], c2_list[i]*data2_list[i][0]**H2_list[i], 
                 f'{colors[i+1]}-', label=f'Регресія (H={H2_list[i]:.4f})')
        ax.loglog(data2_list[i][0], data2_list[i][1], 
                 f'r{markers[i+1]}', label=f'R/S значення')
        ax.set_title(f'R/S аналіз для s2(t) з A2=10^{np.log10(A2_values[i]):.1f}')
        ax.set_xlabel('Часовий інтервал')
        ax.set_ylabel('R/S відношення')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.subplots_adjust(top=0.92)

# =========== ЧАСТИНА 2: АНАЛІЗ АВТОКОРЕЛЯЦІЙ ПОХІДНИХ ===========

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
s1_derivatives = compute_derivatives_up_to_order(s1_values_clean, 3)
s2_derivatives_list = [compute_derivatives_up_to_order(s2, 3) for s2 in s2_values_list_clean]

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
autocorr_colors = ['r', 'g', 'b']
labels = ['Перша похідна', 'Друга похідна', 'Третя похідна']

# Візуалізація автокореляційних функцій для s1 (усі три похідні на одному графіку)
fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
fig.suptitle('Автокореляційні функції похідних s1(t)', fontsize=16)

for i in range(3):
    ax = axes[i]
    lags = np.arange(len(s1_autocorrs[i]))
    ax.stem(lags, s1_autocorrs[i], autocorr_colors[i] + '-', markerfmt=autocorr_colors[i] + 'o', basefmt='k-')
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
        ax.stem(lags, s2_autocorrs_list[i][j], autocorr_colors[j] + '-', 
                markerfmt=autocorr_colors[j] + 'o', basefmt='k-')
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