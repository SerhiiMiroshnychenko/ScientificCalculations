"""
Аналіз залежності температури від вмісту вуглецю по висоті шару
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Дані по температурі (висота шару)
height = np.array([400, 300, 200, 100])
temperature = np.array([1000, 980, 1350, 1385])  # температура для відповідних висот
carbon_content = np.array([3.58, 3.77, 3.84, 3.92])  # вміст вуглецю для відповідних висот

# Функції для нелінійних регресій
def exponential(x, a, b):
    return a * np.exp(b * x)

def logarithmic(x, a, b):
    return a * np.log(x) + b

def power_law(x, a, b):
    return a * np.power(x, b)

print("\nНаявні дані:")
print("-" * 50)
print("Дані по висоті шару:")
print(f"Висота (мм): {height}")
print(f"Температура (°C): {temperature}")
print(f"Вміст вуглецю (%): {carbon_content}")

print("\nДані для аналізу залежності:")
print("-" * 50)
print("| Висота (мм) | Температура (°C) | Вміст вуглецю (%) |")
print("|-------------|-----------------|------------------|")
for h, t, c in zip(height, temperature, carbon_content):
    print(f"| {h:11d} | {t:15d} | {c:16.2f} |")

# Розрахунок кореляції
correlation = np.corrcoef(temperature, carbon_content)[0,1]

print("\nСтатистичний аналіз:")
print("-" * 50)
print(f"Коефіцієнт кореляції: {correlation:.4f}")

# Регресійний аналіз
print("\nРегресійний аналіз:")
print("-" * 50)

# Підготовка даних для регресії
X = carbon_content.reshape(-1, 1)
y = temperature.reshape(-1, 1)

# 1. Лінійна регресія
linear_reg = LinearRegression()
linear_reg.fit(X, y)
linear_r2 = r2_score(y, linear_reg.predict(X))

print("\n1. Лінійна регресія:")
print(f"T = {linear_reg.coef_[0][0]:.2f}C + {linear_reg.intercept_[0]:.2f}")
print(f"R² = {linear_r2:.4f}")

# 2-4. Поліноміальні регресії (2, 3 та 4 ступені)
for degree in [2, 3, 4]:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    poly_r2 = r2_score(y, poly_reg.predict(X_poly))
    
    print(f"\n{degree}. Поліноміальна регресія {degree}-го ступеня:")
    equation = f"T = "
    for i in range(degree, -1, -1):
        if i > 0:
            equation += f"{poly_reg.coef_[0][i]:.2f}C^{i} + "
        else:
            equation += f"{poly_reg.intercept_[0]:.2f}"
    print(equation)
    print(f"R² = {poly_r2:.4f}")

# 5. Експоненціальна регресія
try:
    popt_exp, _ = curve_fit(exponential, carbon_content, temperature)
    y_exp = exponential(carbon_content, *popt_exp)
    exp_r2 = r2_score(temperature, y_exp)
    print("\n5. Експоненціальна регресія:")
    print(f"T = {popt_exp[0]:.2f} * exp({popt_exp[1]:.2f}C)")
    print(f"R² = {exp_r2:.4f}")
except:
    print("\n5. Експоненціальна регресія не змогла зійтися")

# 6. Логарифмічна регресія
try:
    popt_log, _ = curve_fit(logarithmic, carbon_content, temperature)
    y_log = logarithmic(carbon_content, *popt_log)
    log_r2 = r2_score(temperature, y_log)
    print("\n6. Логарифмічна регресія:")
    print(f"T = {popt_log[0]:.2f} * ln(C) + {popt_log[1]:.2f}")
    print(f"R² = {log_r2:.4f}")
except:
    print("\n6. Логарифмічна регресія не змогла зійтися")

# 7. Степенева регресія
try:
    popt_pow, _ = curve_fit(power_law, carbon_content, temperature)
    y_pow = power_law(carbon_content, *popt_pow)
    pow_r2 = r2_score(temperature, y_pow)
    print("\n7. Степенева регресія:")
    print(f"T = {popt_pow[0]:.2f} * C^{popt_pow[1]:.2f}")
    print(f"R² = {pow_r2:.4f}")
except:
    print("\n7. Степенева регресія не змогла зійтися")

# Візуалізація
plt.figure(figsize=(12, 8))
plt.scatter(carbon_content, temperature, color='blue', label='Експериментальні дані')

# Точки для побудови кривих
X_line = np.linspace(min(carbon_content), max(carbon_content), 100)

# Лінійна регресія
y_line = linear_reg.predict(X_line.reshape(-1, 1))
plt.plot(X_line, y_line, color='red', label='Лінійна')

# Поліноміальні регресії
colors = ['green', 'purple', 'orange']
for degree, color in zip([2, 3, 4], colors):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X_line.reshape(-1, 1))
    poly_reg = LinearRegression().fit(poly_features.fit_transform(X), y)
    y_poly = poly_reg.predict(X_poly)
    plt.plot(X_line, y_poly, color=color, label=f'Поліном {degree}-го ступеня')

# Нелінійні регресії
try:
    plt.plot(X_line, exponential(X_line, *popt_exp), 'y--', label='Експоненціальна')
except: pass
try:
    plt.plot(X_line, logarithmic(X_line, *popt_log), 'm--', label='Логарифмічна')
except: pass
try:
    plt.plot(X_line, power_law(X_line, *popt_pow), 'c--', label='Степенева')
except: pass

plt.xlabel('Вміст вуглецю (%)')
plt.ylabel('Температура (°C)')
plt.title('Залежність температури від вмісту вуглецю')
plt.legend()
plt.grid(True)
plt.savefig('temperature_carbon_regression.png', dpi=300, bbox_inches='tight')
plt.close()
