"""
Розрахунок теплоємності газової суміші та теплоти нагрівання.
Суміш: CO (87.4%), CO₂ (10.1%), O₂ (1.6%), N₂ (0.9%)
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Константи
R = 8.31446261815  # Дж/(моль·К)
CAL_TO_JOULE = 4.1868  # Дж/кал
P_STANDARD = 101325  # Па (1 атм)
T_STANDARD = 273.15  # К (0°C)

# Коефіцієнти для газів (в кал/(моль·К))
# CO
a_co = 6.79
b_co = 0.98/1e3  # Ділимо на 1e3 згідно з таблицею
c_co = -0.11*1e5  # Множимо на 1e5 згідно з таблицею

# CO₂
a_co2 = 10.55
b_co2 = 2.16/1e3
c_co2 = -2.04*1e5

# O₂
a_o2 = 7.16
b_o2 = 1.0/1e3
c_o2 = -0.4*1e5

# N₂
a_n2 = 6.66
b_n2 = 1.02/1e3
c_n2 = 0  # для N₂ немає коефіцієнта c

# Об'ємні частки компонентів
r_co = 0.874   # CO
r_co2 = 0.101  # CO₂
r_o2 = 0.016   # O₂
r_n2 = 0.009   # N₂

# Параметри процесу
volume = 100  # м³
t1 = 25       # °C
t2 = 1000     # °C
T1 = t1 + T_STANDARD  # K
T2 = t2 + T_STANDARD  # K

print("1. Температури:")
print(f"   T1 = {t1}°C = {T1:.2f} К")
print(f"   T2 = {t2}°C = {T2:.2f} К")

# Перевірка діапазону температур
T_MIN = 298  # К
T_MAX = 2500  # К

for T in [T1, T2]:
    if not (T_MIN <= T <= T_MAX):
        print(f"\nУВАГА: Температура {T:.2f} К виходить за межі діапазону {T_MIN}-{T_MAX} К")
        print("Результати можуть бути неточними!")


print("\n2. Склад газової суміші:")
print(f"   CO:  {r_co*100:.1f}%")
print(f"   CO₂: {r_co2*100:.1f}%")
print(f"   O₂:  {r_o2*100:.1f}%")
print(f"   N₂:  {r_n2*100:.1f}%")

# Розрахунок кількості речовини
n = (P_STANDARD * volume) / (R * T1)
print(f"\n3. Кількість речовини:")
print(f"   n = {n:.2f} моль")

# Розрахунок кількості речовини для кожного компонента
n_co = n * r_co
n_co2 = n * r_co2
n_o2 = n * r_o2
n_n2 = n * r_n2

print("\n   Кількість речовини кожного компонента:")
print(f"   CO:  {n_co:.2f} моль")
print(f"   CO₂: {n_co2:.2f} моль")
print(f"   O₂:  {n_o2:.2f} моль")
print(f"   N₂:  {n_n2:.2f} моль")

# Переведення R в кал/(моль·К)
R_cal = R / CAL_TO_JOULE

# Розрахунок коефіцієнтів для cv
# CO
a_v_co = (a_co - R_cal) * CAL_TO_JOULE
b_v_co = b_co * CAL_TO_JOULE
c_v_co = c_co * CAL_TO_JOULE

# CO₂
a_v_co2 = (a_co2 - R_cal) * CAL_TO_JOULE
b_v_co2 = b_co2 * CAL_TO_JOULE
c_v_co2 = c_co2 * CAL_TO_JOULE

# O₂
a_v_o2 = (a_o2 - R_cal) * CAL_TO_JOULE
b_v_o2 = b_o2 * CAL_TO_JOULE
c_v_o2 = c_o2 * CAL_TO_JOULE

# N₂
a_v_n2 = (a_n2 - R_cal) * CAL_TO_JOULE
b_v_n2 = b_n2 * CAL_TO_JOULE
c_v_n2 = 0

# Функції теплоємності для кожного компонента
cv_co = lambda T: a_v_co + b_v_co*T + c_v_co/(T*T)
cv_co2 = lambda T: a_v_co2 + b_v_co2*T + c_v_co2/(T*T)
cv_o2 = lambda T: a_v_o2 + b_v_o2*T + c_v_o2/(T*T)
cv_n2 = lambda T: a_v_n2 + b_v_n2*T + c_v_n2/(T*T)

# Аналітичний розрахунок для кожного компонента
Q_co_analytical = n_co * (
        a_v_co * (T2 - T1) +
        b_v_co * (T2**2 - T1**2) / 2 +
        c_v_co * (-1/T2 + 1/T1)
)

Q_co2_analytical = n_co2 * (
        a_v_co2 * (T2 - T1) +
        b_v_co2 * (T2**2 - T1**2) / 2 +
        c_v_co2 * (-1/T2 + 1/T1)
)

Q_o2_analytical = n_o2 * (
        a_v_o2 * (T2 - T1) +
        b_v_o2 * (T2**2 - T1**2) / 2 +
        c_v_o2 * (-1/T2 + 1/T1)
)

Q_n2_analytical = n_n2 * (
        a_v_n2 * (T2 - T1) +
        b_v_n2 * (T2**2 - T1**2) / 2 +
        c_v_n2 * (-1/T2 + 1/T1)
)

# Загальна теплота (аналітично)
Q_analytical = Q_co_analytical + Q_co2_analytical + Q_o2_analytical + Q_n2_analytical

# Числові розрахунки
steps = 1000
T = np.linspace(T1, T2, steps)

# Метод трапецій для кожного компонента
Q_co = n_co * np.trapezoid([cv_co(t) for t in T], T)
Q_co2 = n_co2 * np.trapezoid([cv_co2(t) for t in T], T)
Q_o2 = n_o2 * np.trapezoid([cv_o2(t) for t in T], T)
Q_n2 = n_n2 * np.trapezoid([cv_n2(t) for t in T], T)

# Загальна теплота (метод трапецій)
Q_trapz = Q_co + Q_co2 + Q_o2 + Q_n2

# Метод Сімпсона для кожного компонента
Q_co_simpson = n_co * integrate.simpson(y=[cv_co(t) for t in T], x=T)
Q_co2_simpson = n_co2 * integrate.simpson(y=[cv_co2(t) for t in T], x=T)
Q_o2_simpson = n_o2 * integrate.simpson(y=[cv_o2(t) for t in T], x=T)
Q_n2_simpson = n_n2 * integrate.simpson(y=[cv_n2(t) for t in T], x=T)

# Загальна теплота (метод Сімпсона)
Q_simpson = Q_co_simpson + Q_co2_simpson + Q_o2_simpson + Q_n2_simpson

print(f"\n5. Результати розрахунку теплоти:")
print(f"\tАналітичний метод: {Q_analytical/1000:.2f} кДж")
print(f"\tМетод трапецій: {Q_trapz/1000:.2f} кДж")
print(f"\tМетод Сімпсона: {Q_simpson/1000:.2f} кДж")

# Різниця між методами
diff_trapz = abs(Q_analytical - Q_trapz)/1000
diff_simpson = abs(Q_analytical - Q_simpson)/1000
print(f"\tРізниця між аналітичним та трапецій: {diff_trapz:.4f} кДж")
print(f"\tРізниця між аналітичним та Сімпсона: {diff_simpson:.4f} кДж")

# Відносні похибки
error_trapz = diff_trapz/(Q_analytical/1000)*100
error_simpson = diff_simpson/(Q_analytical/1000)*100
print(f"\tВідносна похибка (трапеції): {error_trapz:.6f}%")
print(f"\tВідносна похибка (Сімпсон): {error_simpson:.6f}%")

print("\n6. Внесок кожного компонента в теплоту нагрівання:")
print(f"   CO:  {Q_co/1000:.2f} кДж ({Q_co/Q_trapz*100:.1f}%)")
print(f"   CO₂: {Q_co2/1000:.2f} кДж ({Q_co2/Q_trapz*100:.1f}%)")
print(f"   O₂:  {Q_o2/1000:.2f} кДж ({Q_o2/Q_trapz*100:.1f}%)")
print(f"   N₂:  {Q_n2/1000:.2f} кДж ({Q_n2/Q_trapz*100:.1f}%)")

print("\n7. Перевірка розмірностей:")
print(f"   Теплоємність: [Дж/(моль·К)]")
print(f"   Кількість речовини: [моль]")
print(f"   Температура: [К]")
print(f"   Теплота: [Дж/(моль·К)] · [моль] · [К] = [Дж] ✓")

# Функція для розрахунку cv суміші в залежності від температури
def cv_mix(T, composition={"co": r_co, "co2": r_co2, "o2": r_o2, "n2": r_n2}):
    return (composition["co"] * cv_co(T) +
            composition["co2"] * cv_co2(T) +
            composition["o2"] * cv_o2(T) +
            composition["n2"] * cv_n2(T))

# Побудова графіка теплоємності для суміші та кожного компонента
plot_temperatures = np.linspace(T1, T2, 100)
plt.figure(figsize=(12, 8))

# Графік теплоємності для кожного компонента
plt.plot(plot_temperatures - 273.15, [cv_co(T) for T in plot_temperatures], 'r-', label='CO')
plt.plot(plot_temperatures - 273.15, [cv_co2(T) for T in plot_temperatures], 'b-', label='CO₂')
plt.plot(plot_temperatures - 273.15, [cv_o2(T) for T in plot_temperatures], 'g-', label='O₂')
plt.plot(plot_temperatures - 273.15, [cv_n2(T) for T in plot_temperatures], 'k-', label='N₂')

# Додаємо графік теплоємності суміші
plt.plot(plot_temperatures - 273.15, [cv_mix(T) for T in plot_temperatures], 'm--', linewidth=2, label='Суміш')
plt.xlabel('Температура, °C')
plt.ylabel('Теплоємність при сталому об\'ємі, Дж/(моль·К)')
plt.title('Теплоємність суміші та її компонентів при сталому об\'ємі')
plt.grid(True)
plt.legend(loc='best')

# Додаємо додаткові пояснення до графіка
plt.figtext(0.5, 0.01, 'Пунктирна лінія показує теплоємність суміші, яка є середньозваженою від теплоємностей компонентів',
            ha='center', fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("mixture_and_components_cv.png", dpi=300)
plt.show()

# Додатковий графік - зміна теплоємності суміші з температурою
plt.figure(figsize=(12, 6))
plt.plot(plot_temperatures - 273.15, [cv_mix(T) for T in plot_temperatures], 'r-', linewidth=2)
plt.xlabel('Температура, °C')
plt.ylabel('Теплоємність суміші, Дж/(моль·К)')
plt.title('Зміна теплоємності суміші з температурою')
plt.grid(True)
plt.tight_layout()
plt.savefig("mixture_cv.png", dpi=300)
plt.show()
