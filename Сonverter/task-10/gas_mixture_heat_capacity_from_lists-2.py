#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Розрахунок зміни середньої об'ємної теплоємності газової суміші при зміні складу.
Склад газу на кожному кроці задається окремими списками для кожного компоненту.
"""

import matplotlib.pyplot as plt

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

# Функція для розрахунку середньої теплоємності суміші
def calculate_heat_capacity(r_co, r_co2, r_o2, r_n2, t_celsius):
    """
    Розраховує середню мольну та об'ємну теплоємність газової суміші з вказаним складом
    при заданій температурі.

    Parameters:
    r_co (float): Об'ємна частка CO
    r_co2 (float): Об'ємна частка CO₂
    r_o2 (float): Об'ємна частка O₂
    r_n2 (float): Об'ємна частка N₂
    t_celsius (float): Температура, °C

    Returns:
    tuple: (cp_mix, cp_mix_vol) де:
           cp_mix - середня мольна теплоємність, кДж/(моль·К)
           cp_mix_vol - середня об'ємна теплоємність, кДж/(м³·К)
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
    cp_mix_kj = cp_mix / 1000  # кДж/(моль·К)

    # Об'ємна теплоємність при сталому тиску
    cp_mix_vol = cp_mix / V_m / 1000  # кДж/(м³·К)

    return cp_mix_kj, cp_mix_vol

# Cклад газової суміші на кожному кроці (у відсотках)
co_percentages = [87.4, 87.1, 82.7, 77.6, 73.5, 67.3, 55.5, 40.1, 25.9, 20.3, 12.5, 8.8]
co2_percentages = [10.1, 10.6, 15.2, 20.6, 25, 31.5, 43.5, 59.1, 73.4, 79.2, 87.1, 91.0]
o2_percentages = [1.6, 1.45, 1.3, 1.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
n2_percentages = [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1]

# Параметри для побудови графіка
time_per_step = 10  # Час на один крок, хв

# Список температур на кожному кроці (в градусах Цельсія)
temperatures = [100, 110, 123, 136, 141, 154, 168, 152, 143, 131, 120, 110]
# temperatures = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

# Перевірка на однакову довжину всіх списків
if not (len(co_percentages) == len(co2_percentages) == len(o2_percentages) == len(n2_percentages)):
    raise ValueError("Всі списки з частками компонентів повинні мати однакову довжину")

# Перевірка на відповідність кількості кроків та температур
if len(temperatures) < len(co_percentages):
    raise ValueError(f"Список температур повинен мати не менше елементів ніж кількість кроків: {len(co_percentages)}")

steps = len(co_percentages)  # Кількість точок (кроків + початкова точка)
times = [i * time_per_step for i in range(steps)]  # Час у хвилинах

# Обмежуємо список температур кількістю кроків
temperatures = temperatures[:steps]

# Ініціалізація масивів для зберігання теплоємностей
heat_capacities_molar = []  # Мольні теплоємності
heat_capacities_volumetric = []  # Об'ємні теплоємності

print("Зміна середньої теплоємності газової суміші при зміні складу та температури\n")

# Виведення заголовка таблиці з використанням табличної форми
table_width = 145
header = f"| {'№':^4} | {'Час, хв':^10} | {'T, °C':^8} | {'CO, %':^10} | {'CO₂, %':^10} | {'O₂, %':^10} | {'N₂, %':^10} | {'Мольна теплоємність, кДж/(моль·К)':^30} | {'Об\'ємна теплоємність, кДж/(м³·К)':^30} |"

print("=" * table_width)
print(header)
print("=" * table_width)

# Обчислення теплоємності для кожного кроку
for step in range(steps):
    # Отримуємо об'ємні частки у відсотках для поточного кроку
    co_percent = co_percentages[step]
    co2_percent = co2_percentages[step]
    o2_percent = o2_percentages[step]
    n2_percent = n2_percentages[step]

    # Отримуємо температуру для поточного кроку
    t_celsius = temperatures[step]

    # Перетворення відсотків в частки
    r_co = co_percent / 100
    r_co2 = co2_percent / 100
    r_o2 = o2_percent / 100
    r_n2 = n2_percent / 100

    # Розрахунок середньої мольної та об'ємної теплоємності
    try:
        cp_mix_molar, cp_mix_vol = calculate_heat_capacity(r_co, r_co2, r_o2, r_n2, t_celsius)
        heat_capacities_molar.append(cp_mix_molar)
        heat_capacities_volumetric.append(cp_mix_vol)

        # Виведення результатів у вигляді таблиці
        row = f"| {step:^4} | {step * time_per_step:^10} | {t_celsius:^8} | {co_percent:^10.1f} | {co2_percent:^10.1f} | {o2_percent:^10.1f} | {n2_percent:^10.1f} | {cp_mix_molar:^30.4f} | {cp_mix_vol:^30.4f} |"
        print(row)

    except ValueError as e:
        error_row = f"| {step:^4} | {step * time_per_step:^10} | {t_celsius:^8} | {co_percent:^10.1f} | {co2_percent:^10.1f} | {o2_percent:^10.1f} | {n2_percent:^10.1f} | {'ПОМИЛКА: ' + str(e):^30} | {'ПОМИЛКА: ' + str(e):^30} |"
        print(error_row)

print("=" * table_width)

# Графік 1: Залежність об'ємної теплоємності від часу
plt.figure(figsize=(10, 6))
plt.plot(times, heat_capacities_volumetric, marker='o', linestyle='-', color='b')
plt.grid(True)
plt.xlabel('Час, хв', fontsize=12)
plt.ylabel('Середня об\'ємна теплоємність, кДж/(м³·К)', fontsize=12)
plt.title('Зміна середньої об\'ємної теплоємності газової суміші з часом', fontsize=14)

# Додавання температур до першого графіку
for i, temp in enumerate(temperatures):
    plt.annotate(f'{temp}°C', xy=(times[i], heat_capacities_volumetric[i]),
                 xytext=(times[i], heat_capacities_volumetric[i] + 0.05),  # Більший відступ
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.7),
                 ha='center',
                 arrowprops=dict(arrowstyle='->', lw=0.5, connectionstyle="arc3,rad=0"))

# Додавання пояснень до першого графіку
plt.annotate(f'Початковий склад:\nCO: {co_percentages[0]}%\nCO₂: {co2_percentages[0]}%\nO₂: {o2_percentages[0]}%\nN₂: {n2_percentages[0]}%\nT: {temperatures[0]}°C',
             xy=(times[0], heat_capacities_volumetric[0]), xytext=(times[0]+15, heat_capacities_volumetric[0]+0.05),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=1),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate(f'Кінцевий склад:\nCO: {co_percentages[-1]}%\nCO₂: {co2_percentages[-1]}%\nO₂: {o2_percentages[-1]}%\nN₂: {n2_percentages[-1]}%\nT: {temperatures[-1]}°C',
             xy=(times[-1], heat_capacities_volumetric[-1]), xytext=(times[-1]-15, heat_capacities_volumetric[-1]-0.15),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=1),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Графік 2: Залежність мольної теплоємності від часу
plt.figure(figsize=(10, 6))
plt.plot(times, heat_capacities_molar, marker='o', linestyle='-', color='g')
plt.grid(True)
plt.xlabel('Час, хв', fontsize=12)
plt.ylabel('Середня мольна теплоємність, кДж/(моль·К)', fontsize=12)
plt.title('Зміна середньої мольної теплоємності газової суміші з часом', fontsize=14)

# Додавання температур до графіку мольної теплоємності
for i, temp in enumerate(temperatures):
    plt.annotate(f'{temp}°C', xy=(times[i], heat_capacities_molar[i]),
                 xytext=(times[i], heat_capacities_molar[i] + 0.0005),  # Мінімальний відступ
                 bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="black", alpha=0.7),
                 ha='center',
                 arrowprops=dict(arrowstyle="-", lw=0.5, connectionstyle="arc3,rad=0"))

# Графік 3: Порівняння теплоємностей (нормалізованих) від часу
plt.figure(figsize=(12, 7))

# Нормалізація для кращого порівняння на одному графіку
norm_factor_molar = heat_capacities_molar[0]
norm_factor_vol = heat_capacities_volumetric[0]

normalized_molar = [cp / norm_factor_molar for cp in heat_capacities_molar]
normalized_vol = [cp / norm_factor_vol for cp in heat_capacities_volumetric]

plt.plot(times, normalized_molar, marker='o', linestyle='-', color='g', label='Мольна (нормалізована)')
plt.plot(times, normalized_vol, marker='s', linestyle='--', color='b', label='Об\'ємна (нормалізована)')
plt.grid(True)
plt.xlabel('Час, хв', fontsize=12)
plt.ylabel('Відносна зміна теплоємності', fontsize=12)
plt.title('Порівняння зміни мольної та об\'ємної теплоємності газової суміші з часом', fontsize=14)
plt.legend()

# Додавання температур до графіку порівняння
for i, temp in enumerate(temperatures):
    plt.annotate(f'{temp}°C', xy=(times[i], normalized_molar[i]),
                 xytext=(times[i], normalized_molar[i] + 0.1),  # Ще більший відступ для кращої видимості
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.7),
                 ha='center',
                 arrowprops=dict(arrowstyle='->', lw=0.5, connectionstyle="arc3,rad=0"))

# Графік 4: Залежність теплоємності від температури
plt.figure(figsize=(10, 6))
plt.plot(temperatures, heat_capacities_volumetric, marker='o', linestyle='-', color='r', label='Об\'ємна')
plt.plot(temperatures, heat_capacities_molar, marker='s', linestyle='-', color='darkgreen', label='Мольна')
plt.grid(True)
plt.xlabel('Температура, °C', fontsize=12)
plt.ylabel('Теплоємність', fontsize=12)
plt.title('Зміна теплоємності газової суміші з температурою', fontsize=14)
plt.legend()

# Додавання номерів кроків до графіку залежності від температури - розміщуємо їх вище лінії
for i in range(steps):
    # Визначаємо, куди розміщувати мітку, щоб не накладалася на графік
    if i % 2 == 0:  # Парні кроки - вгорі
        offset_y = 0.08  # Зміщення вгору
    else:  # Непарні - внизу
        offset_y = -0.08  # Зміщення вниз

    plt.annotate(f'Крок {i}', xy=(temperatures[i], heat_capacities_volumetric[i]),
                 xytext=(temperatures[i], heat_capacities_volumetric[i] + offset_y),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.7),
                 ha='center',
                 arrowprops=dict(arrowstyle='->', lw=1.0, connectionstyle="arc3,rad=0"))

# Відображення графіків
plt.show()

if len(heat_capacities_molar) > 1 and len(heat_capacities_volumetric) > 1:
    print("\nВисновки:")
    print(f"При зміні складу газової суміші та температури від початкового до кінцевого стану:")

    print(f"\n1. Мольна теплоємність:")
    print(f"   - Початкова: {heat_capacities_molar[0]:.4f} кДж/(моль·К)")
    print(f"   - Кінцева: {heat_capacities_molar[-1]:.4f} кДж/(моль·К)")
    print(f"   - Зміна: {(heat_capacities_molar[-1] - heat_capacities_molar[0]):.4f} кДж/(моль·К) або {(heat_capacities_molar[-1] / heat_capacities_molar[0] - 1)*100:.1f}%")

    print(f"\n2. Об'ємна теплоємність:")
    print(f"   - Початкова: {heat_capacities_volumetric[0]:.4f} кДж/(м³·К)")
    print(f"   - Кінцева: {heat_capacities_volumetric[-1]:.4f} кДж/(м³·К)")
    print(f"   - Зміна: {(heat_capacities_volumetric[-1] - heat_capacities_volumetric[0]):.4f} кДж/(м³·К) або {(heat_capacities_volumetric[-1] / heat_capacities_volumetric[0] - 1)*100:.1f}%")

    print(f"\nЦі зміни зумовлені двома факторами:")
    print(f"1. Зміною складу суміші: збільшенням вмісту CO₂ з {co2_percentages[0]}% до {co2_percentages[-1]}%, який має вищу мольну теплоємність порівняно з CO.")
    print(f"2. Зміною температури: з {temperatures[0]}°C до {temperatures[-1]}°C, що також впливає на теплоємність компонентів.")

    print(f"\nОсобливості впливу температури:")
    print(f"- Мольна теплоємність змінюється з температурою відповідно до формули c_p = a + b·T + c/T².")
    print(f"- Об'ємна теплоємність додатково залежить від зміни мольного об'єму з температурою (обернено пропорційна T).")
    if heat_capacities_molar[-1] > heat_capacities_molar[0] and heat_capacities_volumetric[-1] < heat_capacities_volumetric[0]:
        print(f"\nЦікавий ефект: мольна та об'ємна теплоємності змінюються у протилежних напрямках,")
        print(f"що демонструє складність термодинамічних залежностей при високих температурах.")
