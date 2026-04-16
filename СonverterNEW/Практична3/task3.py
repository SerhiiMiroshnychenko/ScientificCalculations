"""
Практична робота № 3
Прихідна частина теплового балансу
Варіант № 20
"""

# Вхідні дані
variant_no = 20
m_hm = 700
Si_hm = 1.4
t_hm = 1440
PCR = 0.15

dSi = 10.504
dMn = 5.179
dP = 0.202
dC = 31.879
m_Fe_to_slag = 16.631

q_Si = 32157
q_Mn = 6589
q_P = 24052
q_Fe = 4799
q_C_CO = 9158
q_C_CO2 = 32803
eta_PCR = 0.85

# Розрахунок
## Фізичне тепло чавуну
cp_hm = 0.8 + 0.04 * Si_hm
print(f"{cp_hm = :.3f}")
h_hm = cp_hm * t_hm + 20
print(f"{h_hm = :.3f}")
Q_phys = (m_hm * h_hm) / 1000
print(f"{Q_phys = :.3f}")

## Хімічне тепло елементів
Q_Si = dSi * q_Si / 1000
print(f"{Q_Si = :.3f}")
Q_Mn = dMn * q_Mn / 1000
print(f"{Q_Mn = :.3f}")
Q_P = dP * q_P / 1000
print(f"{Q_P = :.3f}")
Q_Fe = m_Fe_to_slag * q_Fe / 1000
print(f"{Q_Fe = :.3f}")

## Тепло вуглецю
q_C_avg = q_C_CO + PCR * eta_PCR * (q_C_CO2 - q_C_CO)
print(f"{q_C_avg = :.3f}")
Q_C = dC * q_C_avg / 1000
print(f"{Q_C = :.3f}")

## Загальні показники
Q_chem = Q_Si + Q_Mn + Q_P + Q_Fe + Q_C
print(f"{Q_chem = :.3f}")
Q_in = Q_phys + Q_chem
print(f"{Q_in = :.3f}")

## Контрольні частки
p_phys = (Q_phys / Q_in) * 100
print(f"{p_phys = :.3f}")
p_C = (Q_C / Q_chem) * 100
print(f"{p_C = :.3f}")
p_Si = (Q_Si / Q_chem) * 100
print(f"{p_Si = :.3f}")

print(f"ЧАСТКА ФІЗИЧНОГО ТЕПЛА: {p_phys:.1f} %")
print(f"ВНЕСОК C В ХІМІЧНЕ ТЕПЛО: {p_C:.1f} %")
print(f"ВНЕСОК Si В ХІМІЧНЕ ТЕПЛО: {p_Si:.1f} %")
