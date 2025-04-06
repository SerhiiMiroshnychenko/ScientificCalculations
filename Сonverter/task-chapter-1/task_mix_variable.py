"""
Розрахунок теплоємності газової суміші та теплоти нагрівання з урахуванням зміни складу суміші.
Склад суміші змінюється на кожному кроці нагрівання.
"""

import numpy as np
import matplotlib.pyplot as plt

# Константи
R = 8.31446261815  # Дж/(моль·К)
CAL_TO_JOULE = 4.1868  # Дж/кал
P_STANDARD = 101325  # Па (1 атм)
T_STANDARD = 273.15  # К (0°C)

# Об'єм суміші
volume = 100  # м³

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

# Склад газової суміші на кожному кроці (у відсотках)
co_percentages = [87.4, 87.1, 82.7, 77.6, 73.5, 67.3, 55.5, 40.1, 25.9, 20.3, 12.5, 8.8]
co2_percentages = [10.1, 10.6, 15.2, 20.6, 25, 31.5, 43.5, 59.1, 73.4, 79.2, 87.1, 91.0]
o2_percentages = [1.6, 1.45, 1.3, 1.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
n2_percentages = [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1]

# co_percentages = [87.4]*12
# co2_percentages = [10.1]*12
# o2_percentages = [1.6]*12
# n2_percentages = [0.9]*12

# Параметри для побудови графіка
time_per_step = 10  # Час на один крок, хв

# Список температур на кожному кроці (в градусах Цельсія)
temperatures = [25, 114, 202, 291, 380, 468, 557, 645, 734, 823, 911, 1000]

# Перевірка на однакову довжину всіх списків
steps = len(temperatures)
if not (len(co_percentages) == steps and len(co2_percentages) == steps and
        len(o2_percentages) == steps and len(n2_percentages) == steps):
    raise ValueError("Усі списки повинні мати однакову довжину!")

# Перетворення температур у Кельвіни
temperatures_K = [t + T_STANDARD for t in temperatures]

# Переведення відсотків у долі одиниці
co_fractions = [p / 100 for p in co_percentages]
co2_fractions = [p / 100 for p in co2_percentages]
o2_fractions = [p / 100 for p in o2_percentages]
n2_fractions = [p / 100 for p in n2_percentages]

# Переведення R в кал/(моль·К)
R_cal = R / CAL_TO_JOULE

# Функція для розрахунку cv для компонентів
def get_cv_coefficients():
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

    return (a_v_co, b_v_co, c_v_co,
            a_v_co2, b_v_co2, c_v_co2,
            a_v_o2, b_v_o2, c_v_o2,
            a_v_n2, b_v_n2, c_v_n2)

# Отримання коефіцієнтів
a_v_co, b_v_co, c_v_co, a_v_co2, b_v_co2, c_v_co2, a_v_o2, b_v_o2, c_v_o2, a_v_n2, b_v_n2, c_v_n2 = get_cv_coefficients()

# Функції теплоємності для кожного компонента
cv_co = lambda T: a_v_co + b_v_co*T + c_v_co/(T*T)
cv_co2 = lambda T: a_v_co2 + b_v_co2*T + c_v_co2/(T*T)
cv_o2 = lambda T: a_v_o2 + b_v_o2*T + c_v_o2/(T*T)
cv_n2 = lambda T: a_v_n2 + b_v_n2*T + c_v_n2/(T*T)

# Вивід вхідних даних
print("=" * 60)
print("ВХІДНІ ДАНІ:")
print("=" * 60)
print(f"Об'єм газової суміші: {volume} м³")
print("\nСклад суміші на кожному кроці (відсотки):")
print("-" * 60)
print("Крок | Температура |    CO    |    CO₂   |    O₂    |    N₂    |")
print("-" * 60)
for i in range(steps):
    print(f"{i+1:4d} | {temperatures[i]:11d} | {co_percentages[i]:8.1f} | {co2_percentages[i]:8.1f} | {o2_percentages[i]:8.1f} | {n2_percentages[i]:8.1f} |")
print("-" * 60)

# Початок розрахунків
q_values = []  # Зберігатимемо теплоту для кожного кроку
step_details = []  # Зберігатимемо деталі по кожному кроку
total_heat = 0  # Загальна теплота нагрівання

print("\n" + "=" * 60)
print("РОЗРАХУНКИ:")
print("=" * 60)

# Розрахунок теплоти для кожного кроку
for i in range(steps - 1):
    T1 = temperatures_K[i]
    T2 = temperatures_K[i + 1]

    # Розрахунок кількості речовини на початку кроку
    n = (P_STANDARD * volume) / (R * T1)

    # Склад на поточному кроці
    r_co = co_fractions[i]
    r_co2 = co2_fractions[i]
    r_o2 = o2_fractions[i]
    r_n2 = n2_fractions[i]

    # Кількість речовини для кожного компонента
    n_co = n * r_co
    n_co2 = n * r_co2
    n_o2 = n * r_o2
    n_n2 = n * r_n2

    # Аналітичний розрахунок для кожного компонента
    Q_co = n_co * (
            a_v_co * (T2 - T1) +
            b_v_co * (T2**2 - T1**2) / 2 +
            c_v_co * (-1/T2 + 1/T1)
    )

    Q_co2 = n_co2 * (
            a_v_co2 * (T2 - T1) +
            b_v_co2 * (T2**2 - T1**2) / 2 +
            c_v_co2 * (-1/T2 + 1/T1)
    )

    Q_o2 = n_o2 * (
            a_v_o2 * (T2 - T1) +
            b_v_o2 * (T2**2 - T1**2) / 2 +
            c_v_o2 * (-1/T2 + 1/T1)
    )

    Q_n2 = n_n2 * (
            a_v_n2 * (T2 - T1) +
            b_v_n2 * (T2**2 - T1**2) / 2 +
            c_v_n2 * (-1/T2 + 1/T1)
    )

    # Загальна теплота для поточного кроку
    Q_step = Q_co + Q_co2 + Q_o2 + Q_n2

    # Додавання до списку значень
    q_values.append(Q_step)
    total_heat += Q_step

    # Зберігаємо деталі для кожного кроку
    step_details.append({
        "step": i + 1,
        "T1": T1 - T_STANDARD,
        "T2": T2 - T_STANDARD,
        "delta_T": T2 - T1,
        "n": n,
        "Q_co": Q_co,
        "Q_co2": Q_co2,
        "Q_o2": Q_o2,
        "Q_n2": Q_n2,
        "Q_step": Q_step,
        "co_percent": co_percentages[i],
        "co2_percent": co2_percentages[i],
        "o2_percent": o2_percentages[i],
        "n2_percent": n2_percentages[i]
    })

# Виведення результатів по кожному кроку
print("\nРезультати розрахунку по кожному кроку:")
print("-" * 90)
print("Крок | T1 (°C) | T2 (°C) | ΔT (K) |  n (моль) |  CO (%)  |  CO₂ (%) |  Q (кДж)  | % від загального")
print("-" * 90)

for detail in step_details:
    print(f"{detail['step']:4d} | {detail['T1']:7.1f} | {detail['T2']:7.1f} | {detail['delta_T']:6.1f} | {detail['n']:9.2f} | {detail['co_percent']:8.1f} | {detail['co2_percent']:8.1f} | {detail['Q_step']/1000:9.2f} | {detail['Q_step']/total_heat*100:8.2f}")

print("-" * 90)
print(f"Загальна теплота: {total_heat/1000:.2f} кДж")

print("\nВнесок компонентів у загальну теплоту:")
total_co = sum(detail["Q_co"] for detail in step_details)
total_co2 = sum(detail["Q_co2"] for detail in step_details)
total_o2 = sum(detail["Q_o2"] for detail in step_details)
total_n2 = sum(detail["Q_n2"] for detail in step_details)

print(f"CO:  {total_co/1000:.2f} кДж ({total_co/total_heat*100:.1f}%)")
print(f"CO₂: {total_co2/1000:.2f} кДж ({total_co2/total_heat*100:.1f}%)")
print(f"O₂:  {total_o2/1000:.2f} кДж ({total_o2/total_heat*100:.1f}%)")
print(f"N₂:  {total_n2/1000:.2f} кДж ({total_n2/total_heat*100:.1f}%)")

# Створення списку часу на основі часу на один крок
times = [i * time_per_step for i in range(steps - 1)]

# Побудова графіків окремо

# 1. Графік зміни складу суміші
plt.figure(figsize=(10, 6))
plt.plot(times, [detail["co_percent"] for detail in step_details], 'y-', label='CO')
plt.plot(times, [detail["co2_percent"] for detail in step_details], 'b-', label='CO₂')
plt.plot(times, [detail["o2_percent"] for detail in step_details], 'g-', label='O₂')
plt.plot(times, [detail["n2_percent"] for detail in step_details], 'k-', label='N₂')
plt.xlabel('Час, хв')
plt.ylabel('Склад, %')
plt.title('Зміна складу суміші під час процесу')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Графік зміни температури
plt.figure(figsize=(10, 6))
t_values = [detail["T1"] for detail in step_details] + [step_details[-1]["T2"]]
time_values = [i * time_per_step for i in range(len(t_values))]
plt.plot(time_values, t_values, 'r-o')
plt.xlabel('Час, хв')
plt.ylabel('Температура, °C')
plt.title('Зміна температури під час процесу')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Графік теплоти на кожному кроці
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(step_details) + 1), [detail["Q_step"]/1000 for detail in step_details])
plt.xlabel('Крок')
plt.ylabel('Теплота, кДж')
plt.title('Теплота, поглинута на кожному кроці')
plt.xticks(range(1, len(step_details) + 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Графік внеску компонентів на кожному кроці
plt.figure(figsize=(10, 6))
q_steps = np.array([detail["Q_step"] for detail in step_details])
q_co = np.array([detail["Q_co"] for detail in step_details])
q_co2 = np.array([detail["Q_co2"] for detail in step_details])
q_o2 = np.array([detail["Q_o2"] for detail in step_details])
q_n2 = np.array([detail["Q_n2"] for detail in step_details])

plt.stackplot(range(1, len(step_details) + 1),
              [q_co/1000, q_co2/1000, q_o2/1000, q_n2/1000],
              labels=['CO', 'CO₂', 'O₂', 'N₂'],
              colors=['gold', 'b', 'g', 'k'])
plt.xlabel('Крок')
plt.ylabel('Теплота, кДж')
plt.title('Внесок кожного компонента в теплоту на кожному кроці')
plt.xticks(range(1, len(step_details) + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
