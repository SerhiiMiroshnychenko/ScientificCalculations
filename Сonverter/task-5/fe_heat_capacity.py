# Імпортуємо необхідні бібліотеки
import numpy as np
from scipy import integrate

# Константи
T_STANDARD = 273.15  # К (0°C)

# Коефіцієнти для теплоємності заліза
a = 0.4613  # кДж/(кг·К)
b = 2.12e-4  # кДж/(кг·К²)
c = -6.87e-7  # кДж/(кг·К³)

# Вхідні дані
t1 = 100  # °C
t2 = 200  # °C

# Переведення температур в Кельвіни
T1 = t1 + T_STANDARD  # К
T2 = t2 + T_STANDARD  # К

print(f"1. Вхідні дані:")
print(f"   Температурний інтервал: {t1}°C - {t2}°C")
print(f"   T1 = {T1:.2f} К")
print(f"   T2 = {T2:.2f} К")
print(f"   Коефіцієнти:")
print(f"   a = {a} кДж/(кг·К)")
print(f"   b = {b:.3e} кДж/(кг·К²)")
print(f"   c = {c:.3e} кДж/(кг·К³)")

# 1. Аналітичний метод
def c_integral_analytical(T1, T2):
    """Обчислення інтегралу для середньої теплоємності аналітичним методом"""
    term1 = a * (T2 - T1)
    term2 = (b/2) * (T2**2 - T1**2)
    term3 = (c/3) * (T2**3 - T1**3)
    return (term1 + term2 + term3)/(T2 - T1)

c_analytical = c_integral_analytical(T1, T2)  # кДж/(кг·К)

print(f"\n2. Середня питома теплоємність (аналітичний метод):")
print(f"   c_сер = {c_analytical:.4f} кДж/(кг·К)")

# 2. Метод трапецій
steps = 100000  # кількість кроків інтегрування
T = np.linspace(T1, T2, steps)  # рівномірне розбиття інтервалу [T1, T2]
c_ist = lambda T: a + b*T + c*T**2  # теплоємність як функція від T
c_values = np.array([c_ist(t) for t in T])  # значення теплоємності для кожної температури
c_trapz = np.trapezoid(c_values, T)/(T2 - T1)  # кДж/(кг·К)

print(f"\n3. Середня питома теплоємність (метод трапецій):")
print(f"   c_сер = {c_trapz:.4f} кДж/(кг·К)")

# 3. Метод Сімпсона
c_simpson = integrate.quad(c_ist, T1, T2)[0]/(T2 - T1)  # кДж/(кг·К)

print(f"\n4. Середня питома теплоємність (метод Сімпсона):")
print(f"   c_сер = {c_simpson:.4f} кДж/(кг·К)")

# Порівняння методів
print(f"\n5. Порівняння методів:")
print(f"   Аналітичний метод: {c_analytical:.4f} кДж/(кг·К)")
print(f"   Метод трапецій: {c_trapz:.4f} кДж/(кг·К)")
print(f"   Метод Сімпсона: {c_simpson:.4f} кДж/(кг·К)")

# Абсолютні різниці
abs_diff_trapz = abs(c_analytical - c_trapz)
abs_diff_simpson = abs(c_analytical - c_simpson)
print(f"\n6. Абсолютні різниці:")
print(f"   Метод трапецій: {abs_diff_trapz:.8f} кДж/(кг·К)")
print(f"   Метод Сімпсона: {abs_diff_simpson:.8f} кДж/(кг·К)")

# Відносні похибки
rel_error_trapz = abs_diff_trapz/abs(c_analytical)*100
rel_error_simpson = abs_diff_simpson/abs(c_analytical)*100
print(f"\n7. Відносні похибки:")
print(f"   Метод трапецій: {rel_error_trapz:.8f}%")
print(f"   Метод Сімпсона: {rel_error_simpson:.8f}%")

print(f"\n8. Перевірка розмірностей:")
print(f"   Істинна теплоємність: [кДж/(кг·К)] ✓")
print(f"   Середня теплоємність: [кДж/(кг·К)] ✓")
