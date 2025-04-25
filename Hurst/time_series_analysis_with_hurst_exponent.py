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

# Візуалізація часових рядів (окремий графік)
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

plt.show()