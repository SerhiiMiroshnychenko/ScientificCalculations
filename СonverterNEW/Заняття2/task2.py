"""
Практична робота №2
Варіант 20
"""

# ДАНО:
variant_no = 20

# 1) Загальне для моделі
m_charge = 1000
m_steel = 920.6
eta_O2 = 0.95
phi_O2 = 0.995

q_v_Si = 0.798
q_v_Mn = 0.204
q_v_P = 0.905
q_v_Fe = 0.201

q_v_C_CO = 0.933
q_v_C_CO2 = 1.866

# 2) З практичної № 1
dSi = 10.504
dMn = 5.179
dP = 0.202
dC = 31.879
m_Fe_to_slag = 16.631

# 3) Дані варіанта для поточної практичної
PCR = 0.15
i_sp = 4

# РОЗРАХУНОК

# 1) Об'єм кисню на окиснення елементів
V_Si = dSi * q_v_Si
V_Mn = dMn * q_v_Mn
V_P = dP * q_v_P
V_Fe = m_Fe_to_slag * q_v_Fe

q_v_C_avg = q_v_C_CO * (1 - PCR) + q_v_C_CO2 * PCR
V_C = dC * q_v_C_avg

# 2) Теоретичний об'єм кисню
V_O2_theor = V_Si + V_Mn + V_P + V_Fe + V_C

# 3) Фактичний (реальний) об'єм кисню
V_O2_actual = V_O2_theor / (eta_O2 * phi_O2)

# 4) Параметри продувки
Q_O2 = i_sp * (m_charge / 1000)
tau = V_O2_actual / Q_O2

# 5) Питомі витрати кисню
v_sh =V_O2_actual / (m_charge / 1000)
v_st = V_O2_actual / (m_steel / 1000)

# =========================================================
# ВИВІД
# =========================================================

print("=" * 72)
print(f"Практична робота №2 (Python). Варіант {variant_no}")
print("=" * 72)
print("1) Складові теоретичної потреби в кисні (Нм3):")
print(f"   V(Si) = {V_Si:.3f}")
print(f"   V(Mn) = {V_Mn:.3f}")
print(f"   V(P)  = {V_P:.3f}")
print(f"   V(Fe) = {V_Fe:.3f}")
print(f"   q_v(C,avg) = {q_v_C_avg:.5f} Нм3/кг")
print(f"   V(C)  = {V_C:.3f}")
print("-" * 72)
print("2) Теоретичний об'єм кисню:")
print(f"   V_theor   = {V_O2_theor:.3f} Нм3")
print("3) Фактичний (реальний) об'єм кисню:")
print(f"   V_actual  = {V_O2_actual:.3f} Нм3")
print("-" * 72)
print("4) Режим продувки:")
print(f"   Q_O2      = {Q_O2:.2f} Нм3/хв")
print(f"   Час продувки = {tau:.2f} хв")
print("-" * 72)
print("5) Питомі витрати кисню:")
print(f"   Питома витрата кисню на тону шихти = {v_sh:.3f} Нм3/т шихти")
print(f"   Питома витрата кисню на тону сталі = {v_st:.3f} Нм3/т сталі")
print("=" * 72)
