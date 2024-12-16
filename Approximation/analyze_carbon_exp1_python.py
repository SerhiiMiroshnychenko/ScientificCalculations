"""
Аналіз залежності вмісту вуглецю від розміру фракцій (Експеримент №1)
Цей скрипт виконує аналіз експериментальних даних для визначення залежності
вмісту вуглецю від розміру фракцій в агломераційній шихті.

Виконуються наступні типи апроксимації:
1. Поліноміальні (лінійна, квадратична, кубічна, 4-го ступеню)
2. Експоненціальна (y = a*exp(b*x))
3. Логарифмічна (y = a*log(x) + b)
4. Степенева (y = a*x^b)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')  # Ігноруємо попередження

# Функції для нелінійної апроксимації
def exp_func(x, a, b):
    """Експоненціальна функція: a*exp(b*x)"""
    return a * np.exp(b * x)

def log_func(x, a, b):
    """Логарифмічна функція: a*log(x) + b"""
    return a * np.log(x) + b

def power_func(x, a, b):
    """Степенева функція: a*x^b"""
    return a * np.power(x, b)

# Вхідні дані
x = np.array([11.0, 8.0, 6.5, 4.0, 2.0, 0.5])  # Розміри фракцій (мм)
C = np.array([1.79, 2.88, 3.11, 3.66, 4.35, 4.42])  # Вміст вуглецю (%)

# Створюємо точки для побудови кривих апроксимації
x_fit = np.linspace(min(x), max(x), 100)

# Поліноміальні апроксимації
p1 = np.polyfit(x, C, 1)  # Лінійна
p2 = np.polyfit(x, C, 2)  # Квадратична
p3 = np.polyfit(x, C, 3)  # Кубічна
p4 = np.polyfit(x, C, 4)  # 4-го ступеню

# Додаткові типи апроксимації
popt_exp, _ = curve_fit(exp_func, x, C, p0=[3.0, -0.1])
popt_log, _ = curve_fit(log_func, x, C, p0=[-0.5, 4])
popt_pow, _ = curve_fit(power_func, x, C, p0=[3.0, -0.1])

# Розрахунок R²
def r_squared(y_true, y_pred):
    """Розрахунок коефіцієнту детермінації R²"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Розрахунок передбачених значень
y_pred_1 = np.polyval(p1, x)
y_pred_2 = np.polyval(p2, x)
y_pred_3 = np.polyval(p3, x)
y_pred_4 = np.polyval(p4, x)
y_pred_exp = exp_func(x, *popt_exp)
y_pred_log = log_func(x, *popt_log)
y_pred_pow = power_func(x, *popt_pow)

# Розрахунок R² для кожної моделі
R2_1 = r_squared(C, y_pred_1)
R2_2 = r_squared(C, y_pred_2)
R2_3 = r_squared(C, y_pred_3)
R2_4 = r_squared(C, y_pred_4)
R2_exp = r_squared(C, y_pred_exp)
R2_log = r_squared(C, y_pred_log)
R2_pow = r_squared(C, y_pred_pow)

# Налаштування розміру графіка
plt.figure(figsize=(10, 8))

# Побудова експериментальних точок
plt.plot(x, C, 'bo', markersize=10, linewidth=1.5, label='Експериментальні дані')

# Побудова кривих апроксимації
plt.plot(x_fit, np.polyval(p1, x_fit), 'k--', linewidth=1.5, 
         label=f'Лінійна (R² = {R2_1:.4f})')
plt.plot(x_fit, np.polyval(p2, x_fit), 'm-.', linewidth=1.5, 
         label=f'Квадратична (R² = {R2_2:.4f})')
plt.plot(x_fit, np.polyval(p3, x_fit), 'c-', linewidth=1.5, 
         label=f'Кубічна (R² = {R2_3:.4f})')
plt.plot(x_fit, np.polyval(p4, x_fit), 'y-', linewidth=1.5, 
         label=f'4-го ступеню (R² = {R2_4:.4f})')
plt.plot(x_fit, exp_func(x_fit, *popt_exp), '--', color='purple', linewidth=1.5, 
         label=f'Експоненціальна (R² = {R2_exp:.4f})')
plt.plot(x_fit, log_func(x_fit, *popt_log), '-.', color='green', linewidth=1.5, 
         label=f'Логарифмічна (R² = {R2_log:.4f})')
plt.plot(x_fit, power_func(x_fit, *popt_pow), ':', color=[0.8, 0.4, 0], linewidth=1.5, 
         label=f'Степенева (R² = {R2_pow:.4f})')

# Налаштування графіка
plt.title('Залежність вмісту вуглецю від розміру фракції (Експеримент №1)', 
          fontsize=14)
plt.xlabel('Розмір фракції, мм', fontsize=12)
plt.ylabel('Вміст вуглецю, %', fontsize=12)
plt.grid(True)
plt.grid(True, which='minor', linestyle=':', alpha=0.5)
plt.minorticks_on()
plt.legend(fontsize=10, loc='best')

# Виведення рівнянь
print("\nРівняння залежності вмісту вуглецю (Експеримент №1):")
print("\n1. Поліноміальні апроксимації:")
print(f"Лінійне: C = {p1[0]:.4f}x + {p1[1]:.4f}")
print(f"Квадратичне: C = {p2[0]:.4f}x^2 + {p2[1]:.4f}x + {p2[2]:.4f}")
print(f"Кубічне: C = {p3[0]:.4f}x^3 + {p3[1]:.4f}x^2 + {p3[2]:.4f}x + {p3[3]:.4f}")
print(f"4-го ступеню: C = {p4[0]:.4f}x^4 + {p4[1]:.4f}x^3 + {p4[2]:.4f}x^2 + {p4[3]:.4f}x + {p4[4]:.4f}")

print("\n2. Додаткові типи апроксимації:")
print(f"Експоненціальна: C = {popt_exp[0]:.4f} * exp({popt_exp[1]:.4f}x)")
print(f"Логарифмічна: C = {popt_log[0]:.4f} * log(x) + {popt_log[1]:.4f}")
print(f"Степенева: C = {popt_pow[0]:.4f} * x^({popt_pow[1]:.4f})")

print("\nКоефіцієнти детермінації (R^2):")
print("1. Для поліноміальних апроксимацій:")
print(f"Лінійна: {R2_1:.4f}")
print(f"Квадратична: {R2_2:.4f}")
print(f"Кубічна: {R2_3:.4f}")
print(f"4-го ступеню: {R2_4:.4f}")

print("\n2. Для додаткових типів апроксимації:")
print(f"Експоненціальна: {R2_exp:.4f}")
print(f"Логарифмічна: {R2_log:.4f}")
print(f"Степенева: {R2_pow:.4f}")

# Збереження графіка
plt.savefig('carbon_analysis_exp1_python.png', dpi=300, bbox_inches='tight')
plt.show()
