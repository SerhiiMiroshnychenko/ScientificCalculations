# Імпортуємо необхідні бібліотеки
import numpy as np
from scipy import integrate  # для методу Сімпсона

# Константи
T_STANDARD = 273.15  # К (0°C)
P_STANDARD = 101325  # Па (1 атм)
R_CAL = 1.986       # кал/(моль·К)
R = 8.31446261815   # Дж/(моль·К)
CAL_TO_JOULE = 4.1868  # Дж/кал
M_O2 = 0.032        # кг/моль (молярна маса O₂)

# Коефіцієнти для O₂ з таблиці 1.1 (в кал/(моль·К))
a = 6.66           # константа
b = 0.88e-3        # коефіцієнт при T
c = -0.12e5        # коефіцієнт при T^(-2)

# Параметри задачі
t1 = 0             # °C
t2 = 1000          # °C
t_avg = (t1 + t2)/2  # °C (середня температура)

T1 = t1 + T_STANDARD  # К
T2 = t2 + T_STANDARD  # К
T_avg = t_avg + T_STANDARD  # К

print(f"1. Вхідні дані:")
print(f"   Температурний інтервал: {t1}°C - {t2}°C")
print(f"   T1 = {T1:.2f} К")
print(f"   T2 = {T2:.2f} К")
print(f"   T_сер = {T_avg:.2f} К")
print(f"   Тиск: {P_STANDARD/1000:.1f} кПа")
print(f"   Коефіцієнти для O₂:")
print(f"   a = {a} кал/(моль·К)")
print(f"   b = {b:.3e} кал/(моль·К²)")
print(f"   c = {c:.3e} кал·К²/моль")

# Розрахунок середньої мольної теплоємності при сталому об'ємі

# 1. Аналітичний метод
def cv_integral_analytical(T1, T2):
    """Обчислення інтегралу для середньої теплоємності аналітичним методом"""
    term1 = (a - R_CAL) * (T2 - T1)
    term2 = (b/2) * (T2**2 - T1**2)
    term3 = -c * (1/T1 - 1/T2)
    return (term1 + term2 + term3)/(T2 - T1)

cv_cal_analytical = cv_integral_analytical(T1, T2)  # кал/(моль·К)
cv_analytical = cv_cal_analytical * CAL_TO_JOULE    # Дж/(моль·К)

print(f"\n2. Середня мольна теплоємність при сталому об'ємі (аналітичний метод):")
print(f"   cv_сер = {cv_cal_analytical:.3f} кал/(моль·К) = {cv_analytical:.3f} Дж/(моль·К)")

# 2. Числовий метод (метод трапецій)
steps = 100000  # кількість кроків інтегрування
T = np.linspace(T1, T2, steps)  # рівномірне розбиття інтервалу [T1, T2]
cv = lambda T: (a - R_CAL) + b*T + c/(T*T)  # теплоємність як функція від T
cv_values = np.array([cv(t) for t in T])  # значення теплоємності для кожної температури
cv_cal_trapz = np.trapezoid(cv_values, T)/(T2 - T1)  # кал/(моль·К)
cv_trapz = cv_cal_trapz * CAL_TO_JOULE  # Дж/(моль·К)

print(f"\n3. Середня мольна теплоємність при сталому об'ємі (метод трапецій):")
print(f"   cv_сер = {cv_cal_trapz:.3f} кал/(моль·К) = {cv_trapz:.3f} Дж/(моль·К)")

# 3. Числовий метод (метод Сімпсона)
cv_cal_simpson = integrate.quad(cv, T1, T2)[0]/(T2 - T1)  # кал/(моль·К)
cv_simpson = cv_cal_simpson * CAL_TO_JOULE  # Дж/(моль·К)

print(f"\n4. Середня мольна теплоємність при сталому об'ємі (метод Сімпсона):")
print(f"   cv_сер = {cv_cal_simpson:.3f} кал/(моль·К) = {cv_simpson:.3f} Дж/(моль·К)")

# Порівняння методів
print(f"\n5. Порівняння методів:")
print(f"   Аналітичний метод: {cv_analytical:.3f} Дж/(моль·К)")
print(f"   Метод трапецій: {cv_trapz:.3f} Дж/(моль·К)")
print(f"   Метод Сімпсона: {cv_simpson:.3f} Дж/(моль·К)")

# Абсолютні різниці
abs_diff_trapz = abs(cv_analytical - cv_trapz)
abs_diff_simpson = abs(cv_analytical - cv_simpson)
print(f"\n6. Абсолютні різниці:")
print(f"   Метод трапецій: {abs_diff_trapz:.6f} Дж/(моль·К)")
print(f"   Метод Сімпсона: {abs_diff_simpson:.6f} Дж/(моль·К)")

# Відносні похибки
rel_error_trapz = abs_diff_trapz/abs(cv_analytical)*100
rel_error_simpson = abs_diff_simpson/abs(cv_analytical)*100
print(f"\n7. Відносні похибки:")
print(f"   Метод трапецій: {rel_error_trapz:.6f}%")
print(f"   Метод Сімпсона: {rel_error_simpson:.6f}%")

# Розрахунок середньої питомої теплоємності (використовуємо аналітичний результат)
cv_mass = cv_analytical/M_O2  # Дж/(кг·К)

print(f"\n8. Середня питома теплоємність:")
print(f"   cv_пит = {cv_mass:.2f} Дж/(кг·К)")

# Розрахунок густини при середній температурі
rho = (P_STANDARD * M_O2)/(R * T_avg)  # кг/м³

print(f"\n9. Густина O₂ при середній температурі {t_avg:.1f}°C:")
print(f"   ρ = {rho:.4f} кг/м³")

# Розрахунок середньої об'ємної теплоємності
cv_vol = cv_mass * rho  # Дж/(м³·К)

print(f"\n10. Середня об'ємна теплоємність:")
print(f"   cv_об = {cv_vol:.2f} Дж/(м³·К)")

print(f"\n11. Перевірка розмірностей:")
print(f"   Мольна теплоємність: [кал/(моль·К)] · [Дж/кал] = [Дж/(моль·К)] ✓")
print(f"   Питома теплоємність: [Дж/(моль·К)] / [кг/моль] = [Дж/(кг·К)] ✓")
print(f"   Об'ємна теплоємність: [Дж/(кг·К)] · [кг/м³] = [Дж/(м³·К)] ✓")
