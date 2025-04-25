import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc

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

# # Обробка можливих нескінченностей та NaN-ів (може виникнути через логарифм від'ємних чисел)
# s1_values[~np.isfinite(s1_values)] = np.nan
# for i in range(len(s2_values_list)):
#     s2_values_list[i][~np.isfinite(s2_values_list[i])] = np.nan

# Візуалізація часових рядів
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

# Розрахунок показника Герста для кожного ряду
print("Розрахунок показника Герста:")

# Функція для розрахунку і виведення показника Герста
def calculate_hurst(series, name):
    # Видалення NaN-ів перед розрахунком
    # clean_series = series[~np.isnan(series)]
    clean_series = series

    if len(clean_series) > 100:  # Перевіряємо, чи достатньо даних
        H, c, data = compute_Hc(clean_series, kind='price', simplified=False)
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

# Візуалізація результатів R/S аналізу для s1, якщо можливо
if H1 is not None:
    plt.figure(figsize=(10, 6))
    plt.loglog(data1[0], c1*data1[0]**H1, 'b-', label=f'Regression line (H={H1:.4f})')
    plt.loglog(data1[0], data1[1], 'ro', label='R/S ratio')
    plt.title('R/S Analysis для s1(t)')
    plt.xlabel('Time interval')
    plt.ylabel('R/S ratio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# Візуалізація результатів R/S аналізу для всіх трьох s2
colors = ['g', 'b', 'c']
markers = ['s', '^', 'd']

for i, A2 in enumerate(A2_values):
    if H2_list[i] is not None:
        plt.figure(figsize=(10, 6))
        plt.loglog(data2_list[i][0], c2_list[i]*data2_list[i][0]**H2_list[i], 
                  f'{colors[i]}-', label=f'Regression line (H={H2_list[i]:.4f})')
        plt.loglog(data2_list[i][0], data2_list[i][1], 
                  f'r{markers[i]}', label=f'R/S ratio')
        plt.title(f'R/S Analysis для s2(t) з A2=10^{np.log10(A2_values[i]):.1f}')
        plt.xlabel('Time interval')
        plt.ylabel('R/S ratio')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

# # Порівняльний графік всіх R/S аналізів
# plt.figure(figsize=(12, 8))
#
# # Додаємо s1
# if H1 is not None:
#     plt.loglog(data1[0], data1[1], 'ro', label='s1(t)', alpha=0.7)
#     plt.loglog(data1[0], c1*data1[0]**H1, 'r-', label=f's1(t) регресія (H={H1:.4f})', alpha=0.7)
#
# # Додаємо всі s2
# for i, A2 in enumerate(A2_values):
#     if H2_list[i] is not None:
#         plt.loglog(data2_list[i][0], data2_list[i][1], f'{colors[i]}{markers[i]}',
#                   label=f's2(t), A2=10^{np.log10(A2):.1f}', alpha=0.7)
#         plt.loglog(data2_list[i][0], c2_list[i]*data2_list[i][0]**H2_list[i],
#                   f'{colors[i]}-', label=f's2(t), A2=10^{np.log10(A2):.1f} регресія (H={H2_list[i]:.4f})', alpha=0.7)
#
# plt.title('Порівняння R/S аналізу для всіх часових рядів')
# plt.xlabel('Time interval')
# plt.ylabel('R/S ratio')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

plt.show()