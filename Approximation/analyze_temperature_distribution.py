"""
Аналіз розподілу температури по глибині шару агломераційної шихти
Цей скрипт виконує аналіз експериментальних даних для визначення залежності
температури від глибини шару

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
    return a * np.log(x + 1) + b  # +1 щоб уникнути log(0)

def power_func(x, a, b):
    """Степенева функція: a*x^b"""
    return a * np.power(x + 1, b)  # +1 щоб уникнути 0^b

# Вхідні дані
depth = np.array([400, 375, 350, 325, 300, 275, 250, 225, 200, 175, 150, 125, 100, 75, 50, 25, 0])  # Глибина шару (мм)
temp = np.array([1000, 950, 900, 920, 980, 1100, 1250, 1320, 1350, 1360, 1370, 1380, 1385, 1390, 1395, 1400, 1410])  # Температура (°C)

# Створюємо точки для побудови кривих апроксимації
depth_fit = np.linspace(min(depth), max(depth), 100)

# Поліноміальні апроксимації
p1 = np.polyfit(depth, temp, 1)  # Лінійна
p2 = np.polyfit(depth, temp, 2)  # Квадратична
p3 = np.polyfit(depth, temp, 3)  # Кубічна
p4 = np.polyfit(depth, temp, 4)  # 4-го ступеню

# Додаткові типи апроксимації
popt_exp, _ = curve_fit(exp_func, depth, temp, p0=[1400, -0.001])
popt_log, _ = curve_fit(log_func, depth, temp, p0=[1400, -100])
popt_pow, _ = curve_fit(power_func, depth, temp, p0=[1400, -0.1])

# Розрахунок R²
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Розрахунок передбачених значень
y_pred_1 = np.polyval(p1, depth)
y_pred_2 = np.polyval(p2, depth)
y_pred_3 = np.polyval(p3, depth)
y_pred_4 = np.polyval(p4, depth)
y_pred_exp = exp_func(depth, *popt_exp)
y_pred_log = log_func(depth, *popt_log)
y_pred_pow = power_func(depth, *popt_pow)

# Розрахунок R² для кожної моделі
r2_1 = r_squared(temp, y_pred_1)
r2_2 = r_squared(temp, y_pred_2)
r2_3 = r_squared(temp, y_pred_3)
r2_4 = r_squared(temp, y_pred_4)
r2_exp = r_squared(temp, y_pred_exp)
r2_log = r_squared(temp, y_pred_log)
r2_pow = r_squared(temp, y_pred_pow)

# Створення графіку
plt.figure(figsize=(12, 8))
plt.scatter(depth, temp, color='red', label='Експериментальні дані')

# Побудова кривих апроксимації
plt.plot(depth_fit, np.polyval(p1, depth_fit), 'b--', label=f'Лінійна (R² = {r2_1:.4f})')
plt.plot(depth_fit, np.polyval(p2, depth_fit), 'g--', label=f'Квадратична (R² = {r2_2:.4f})')
plt.plot(depth_fit, np.polyval(p3, depth_fit), 'c--', label=f'Кубічна (R² = {r2_3:.4f})')
plt.plot(depth_fit, np.polyval(p4, depth_fit), 'm--', label=f'4-го ступеню (R² = {r2_4:.4f})')
plt.plot(depth_fit, exp_func(depth_fit, *popt_exp), 'y--', label=f'Експоненціальна (R² = {r2_exp:.4f})')
plt.plot(depth_fit, log_func(depth_fit, *popt_log), 'k--', label=f'Логарифмічна (R² = {r2_log:.4f})')
plt.plot(depth_fit, power_func(depth_fit, *popt_pow), 'r--', label=f'Степенева (R² = {r2_pow:.4f})')

plt.xlabel('Глибина шару (мм)')
plt.ylabel('Температура (°C)')
plt.title('Розподіл температури по глибині шару агломераційної шихти')
plt.legend()
plt.grid(True)

# Зберігаємо графік
plt.savefig('temperature_distribution.png')
plt.close()

print("Аналіз завершено. Графік збережено як 'temperature_distribution.png'")

# Виведення коефіцієнтів апроксимації
print("\nКоефіцієнти апроксимації:")
print(f"Лінійна: y = {p1[0]:.2f}x + {p1[1]:.2f}")
print(f"Квадратична: y = {p2[0]:.2f}x² + {p2[1]:.2f}x + {p2[2]:.2f}")
print(f"Кубічна: y = {p3[0]:.2f}x³ + {p3[1]:.2f}x² + {p3[2]:.2f}x + {p3[3]:.2f}")
print(f"Експоненціальна: y = {popt_exp[0]:.2f}*exp({popt_exp[1]:.4f}x)")
print(f"Логарифмічна: y = {popt_log[0]:.2f}*log(x+1) + {popt_log[1]:.2f}")
print(f"Степенева: y = {popt_pow[0]:.2f}*(x+1)^{popt_pow[1]:.4f}")
