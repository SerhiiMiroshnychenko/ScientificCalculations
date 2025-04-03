#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Розрахунок зміни середньої об'ємної теплоємності газової суміші при зміні складу.
Склад газу на кожному кроці задається окремими списками для кожного компоненту.
Візуалізація залежності теплоємності від часу (кожен крок - 10 хвилин).
"""

import matplotlib.pyplot as plt
import numpy as np

# Константи
R = 8.31446261815  # Дж/(моль·К)
CAL_TO_JOULE = 4.1868  # Дж/кал
P_STANDARD = 101325  # Па (1 атм)
T_STANDARD = 273.15  # К (0°C)

# Коефіцієнти для газів (в кал/(моль·К))
# CO
a_co = 6.79
b_co = 0.98e-3  # Коефіцієнт b·10³ з таблиці
c_co = -0.11*1e5  # Множимо на 1e5 згідно з таблицею

# CO₂
a_co2 = 10.55
b_co2 = 2.16e-3
c_co2 = -2.04*1e5

# O₂
a_o2 = 7.16
b_o2 = 1.0e-3
c_o2 = -0.4*1e5

# N₂
a_n2 = 6.66
b_n2 = 1.02e-3
c_n2 = 0  # для N₂ немає коефіцієнта c

# Функція для розрахунку середньої об'ємної теплоємності суміші
def calculate_heat_capacity(r_co, r_co2, r_o2, r_n2, t_celsius):
    """
    Розраховує середню об'ємну теплоємність газової суміші з вказаним складом
    при заданій температурі.

    Parameters:
    r_co (float): Об'ємна частка CO
    r_co2 (float): Об'ємна частка CO₂
    r_o2 (float): Об'ємна частка O₂
    r_n2 (float): Об'ємна частка N₂
    t_celsius (float): Температура, °C

    Returns:
    float: Середня об'ємна теплоємність суміші, кДж/(м³·К)
    """
    # Перевірка суми часток
    total_fraction = r_co + r_co2 + r_o2 + r_n2
    if abs(total_fraction - 1.0) > 1e-10:
        raise ValueError(f"Сума об'ємних часток має дорівнювати 1.0, отримано: {total_fraction}")

    # Перетворення температури в Кельвіни
    T = t_celsius + T_STANDARD

    # Розрахунок мольних теплоємностей при сталому тиску (cp) в кал/(моль·К)
    cp_co_cal = a_co + b_co*T + c_co/(T*T)
    cp_co2_cal = a_co2 + b_co2*T + c_co2/(T*T)
    cp_o2_cal = a_o2 + b_o2*T + c_o2/(T*T)
    cp_n2_cal = a_n2 + b_n2*T + c_n2/(T*T)

    # Переведення в Дж/(моль·К)
    cp_co = cp_co_cal * CAL_TO_JOULE
    cp_co2 = cp_co2_cal * CAL_TO_JOULE
    cp_o2 = cp_o2_cal * CAL_TO_JOULE
    cp_n2 = cp_n2_cal * CAL_TO_JOULE

    # Об'єм 1 моля газу при даній температурі і тиску
    V_m = R * T / P_STANDARD  # м³/моль

    # Мольна теплоємність при сталому тиску
    cp_mix = r_co * cp_co + r_co2 * cp_co2 + r_o2 * cp_o2 + r_n2 * cp_n2  # Дж/(моль·К)

    # Об'ємна теплоємність при сталому тиску
    cp_mix_vol = cp_mix / V_m / 1000  # кДж/(м³·К)

    return cp_mix_vol

# Cклад газової суміші на кожному кроці (у відсотках)
co_percentages = [87.4, 86.1, 80.7, 75.6, 70.5, 67.3, 55.5, 40.1, 33.9, 20.3, 12.5, 8.8]
print(f"{len(co_percentages) = }")
co2_percentages = [10.1, 11.6, 17.2, 22.6, 28, 31.5, 43.5, 59.1, 65.4, 79.2, 87.1, 91.0]
print(f"{len(co2_percentages) = }")
o2_percentages = [1.6, 1.45, 1.3, 1.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
print(f"{len(o2_percentages) = }")
n2_percentages = [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1]
print(f"{len(n2_percentages) = }")

# Параметри для побудови графіка
time_per_step = 10  # Час на один крок, хв
t_celsius = 100  # Температура в градусах Цельсія

# Перевірка на однакову довжину всіх списків
if not (len(co_percentages) == len(co2_percentages) == len(o2_percentages) == len(n2_percentages)):
    raise ValueError("Всі списки з частками компонентів повинні мати однакову довжину")

steps = len(co_percentages)  # Кількість точок (кроків + початкова точка)
times = [i * time_per_step for i in range(steps)]  # Час у хвилинах

# Ініціалізація масиву для зберігання теплоємностей
heat_capacities = []

print("Зміна середньої об'ємної теплоємності газової суміші при зміні складу\n")
print(f"Температура: {t_celsius}°C ({t_celsius + T_STANDARD:.2f} К)\n")

print("| № кроку | Час, хв | CO, % | CO₂, % | O₂, % | N₂, % | Теплоємність, кДж/(м³·К) |")
print("|---------|---------|-------|--------|-------|-------|--------------------------|")

# Обчислення теплоємності для кожного кроку
for step in range(steps):
    # Отримуємо об'ємні частки у відсотках для поточного кроку
    co_percent = co_percentages[step]
    co2_percent = co2_percentages[step]
    o2_percent = o2_percentages[step]
    n2_percent = n2_percentages[step]

    # Перетворення відсотків в частки
    r_co = co_percent / 100
    r_co2 = co2_percent / 100
    r_o2 = o2_percent / 100
    r_n2 = n2_percent / 100

    # Розрахунок середньої об'ємної теплоємності
    try:
        cp_mix_vol = calculate_heat_capacity(r_co, r_co2, r_o2, r_n2, t_celsius)
        heat_capacities.append(cp_mix_vol)

        # Виведення результатів у вигляді таблиці
        print(f"| {step} | {step * time_per_step} | {co_percent:.1f} | {co2_percent:.1f} | {o2_percent:.1f} | {n2_percent:.1f} | {cp_mix_vol:.4f} |")

    except ValueError as e:
        print(f"| {step} | {step * time_per_step} | {co_percent:.1f} | {co2_percent:.1f} | {o2_percent:.1f} | {n2_percent:.1f} | ПОМИЛКА: {e} |")

# Побудова графіку, якщо є дані для побудови
if heat_capacities:
    plt.figure(figsize=(10, 6))
    plt.plot(times, heat_capacities, marker='o', linestyle='-', color='b')
    plt.grid(True)
    plt.xlabel('Час, хв', fontsize=12)
    plt.ylabel('Середня об\'ємна теплоємність, кДж/(м³·К)', fontsize=12)
    plt.title('Зміна середньої об\'ємної теплоємності газової суміші з часом', fontsize=14)

    # Додавання пояснень до графіку
    plt.annotate(f'Початковий склад:\nCO: {co_percentages[0]}%\nCO₂: {co2_percentages[0]}%\nO₂: {o2_percentages[0]}%\nN₂: {n2_percentages[0]}%',
                 xy=(times[0], heat_capacities[0]), xytext=(times[0]+5, heat_capacities[0]+0.05),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=1),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.annotate(f'Кінцевий склад:\nCO: {co_percentages[-1]}%\nCO₂: {co2_percentages[-1]}%\nO₂: {o2_percentages[-1]}%\nN₂: {n2_percentages[-1]}%',
                 xy=(times[-1], heat_capacities[-1]), xytext=(times[-1], heat_capacities[-1]-0.15),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=1),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # Відображення графіку
    plt.show()


    if len(heat_capacities) > 1:
        print("\nВисновки:")
        print(f"При зміні складу газової суміші від початкового до кінцевого")
        print(f"середня об'ємна теплоємність змінюється з {heat_capacities[0]:.4f} до {heat_capacities[-1]:.4f} кДж/(м³·К).")
        print(f"Це зумовлено в першу чергу збільшенням вмісту CO₂, який має вищу теплоємність порівняно з CO.")
else:
    print("\nНе вдалося побудувати графік через помилки в розрахунках.")
