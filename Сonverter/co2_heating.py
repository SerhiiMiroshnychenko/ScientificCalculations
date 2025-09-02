import numpy as np

R = 8.31446261815  # Дж/(моль·К)

CAL_TO_JOULE = 4.1868  # Дж/кал
P_STANDARD = 101325  # Па (1 атм)
T_STANDARD = 273.15  # К (0°C)

a = 10.55  # кал/(моль·К)
b = 2.16/1e3  # кал/(моль·К²)
c = -2.04/1e-5  # кал·К²/(моль)

T_MIN = 298  # К
T_MAX = 2500  # К

volume = 10  # м³
t1 = 25  # °C
t2 = 1000  # °C

T1 = t1 + T_STANDARD  # T1 = 25 + 273.15 = 298.15 К
T2 = t2 + T_STANDARD  # T2 = 1000 + 273.15 = 1273.15 К
print(f"1. Температури:")
print(f"   T1 = {t1}°C = {T1:.2f} К")
print(f"   T2 = {t2}°C = {T2:.2f} К")

for T in [T1, T2]:
    if not (T_MIN <= T <= T_MAX):
        print(f"\nУВАГА: Температура {T:.2f} К виходить за межі діапазону {T_MIN}-{T_MAX} К")
        print("Результати можуть бути неточними!")

n = (P_STANDARD * volume) / (R * T1)
print(f"\n2. Кількість речовини:")
print(f"   n = {n:.2f} моль")

R_cal = R / CAL_TO_JOULE  # R_cal = 8.31446261815 / 4.1868 ≈ 1.987 кал/(моль·К)
a_v = a - R_cal  # a_v = 10.55 - 1.987 = 8.563 кал/(моль·К)
b_v = b  # b_v = 2.16/10³ кал/(моль·К²)
c_v = c  # c_v = -2.04/10⁻⁵ кал·К²/(моль)

a_v *= CAL_TO_JOULE  # a_v = 8.563 * 4.1868 = 35.85 Дж/(моль·К)
b_v *= CAL_TO_JOULE  # b_v = 2.16/10³ * 4.1868 = 9.044/10³ Дж/(моль·К²)
c_v *= CAL_TO_JOULE  # c_v = -2.04/10⁻⁵ * 4.1868 = -8.541/10⁻⁵ Дж·К²/(моль)

Q_analytical = n * (
        a_v * (T2 - T1) +                  # Інтеграл від константи a_v
        b_v * (T2**2 - T1**2) / 2 +        # Інтеграл від лінійного члена b_v·T
        c_v * (-1/T2 + 1/T1))               # Інтеграл від оберненого квадрата c_v·T⁻²

print(f"   Аналітичний метод: {Q_analytical/1000:.2f} кДж")

steps = 1000  # кількість кроків інтегрування (більше кроків = вища точність)
T = np.linspace(T1, T2, steps)  # рівномірне розбиття інтервалу [T1, T2]
cv = lambda T: a_v + b_v*T + c_v/(T*T)  # теплоємність як функція від T
Q_numerical = n * np.trapezoid([cv(t) for t in T], T)
print(f"   Числовий метод: {Q_numerical/1000:.2f} кДж")

print(f"\n3. Результати розрахунку теплоти:")
print(f"\tАналітичний метод: {Q_analytical/1000:.2f} кДж")
print(f"\tЧисловий метод: {Q_numerical/1000:.2f} кДж")
print(f"\tРізниця між методами: {abs(Q_analytical - Q_numerical)/1000:.4f} кДж")
relative_error = abs(Q_analytical - Q_numerical)/abs(Q_analytical)*100
print(f"\tВідносна похибка: {relative_error:.6f}%")
