# --- Вхідні дані ---
variant_no = 20
m_charge = 1000.0

# Склад чавуну (%)
Si_hm = 1.40
Mn_hm = 0.70
P_hm = 0.04
C_hm = 4.60

# Параметри шихти та процесу
w_scr = 0.300
B = 3.4

# Склад брухту (%)
Si_scr = 0.25
Mn_scr = 0.40
P_scr = 0.02
C_scr = 0.20

# Залишковий вміст у сталі (%)
Si_st = 0.005
Mn_st = 0.10
P_st = 0.015
C_st = 0.10

# Параметри
alpha_CO = 0.90
x_CaO_lime = 0.90
x_SiO2_lime = 0.02
k_main = 0.80
w_FeO = 0.15
k_dust = 0.015

# Молярні маси
M_Si = 28.0855
M_SiO2 = 60.0843
M_Mn = 54.9380
M_MnO = 70.9374
M_P = 30.9738
M_P2O5 = 141.9445
M_C = 12.011
M_CO = 28.010
M_CO2 = 44.009
M_Fe = 55.845
M_FeO = 71.844

# --- Розрахунок ---
m_hm = m_charge * (1.0 - w_scr)
m_scr = m_charge * w_scr

Si_in = m_hm * (Si_hm / 100) + m_scr * (Si_scr / 100)
Mn_in = m_hm * (Mn_hm / 100) + m_scr * (Mn_scr / 100)
P_in = m_hm * (P_hm / 100) + m_scr * (P_scr / 100)
C_in = m_hm * (C_hm / 100) + m_scr * (C_scr / 100)

eps_steel = 0.1
max_iter = 50
iter_no = 0
m_steel0 = m_charge
ok = True

# Ініціалізація змінних перед циклом
m_steel = 0
Yield = 0
m_slag = 0
m_lime = 0
dSi, dMn, dP, dC = 0, 0, 0, 0
m_SiO2_ox, m_MnO_ox, m_P2O5_ox = 0, 0, 0
m_Fe_to_slag, m_dust = 0, 0

while True:
    Si_st_mass = m_steel0 * (Si_st / 100)
    Mn_st_mass = m_steel0 * (Mn_st / 100)
    P_st_mass = m_steel0 * (P_st / 100)
    C_st_mass = m_steel0 * (C_st / 100)

    dSi = max(0, Si_in - Si_st_mass)
    dMn = max(0, Mn_in - Mn_st_mass)
    dP = max(0, P_in - P_st_mass)
    dC = max(0, C_in - C_st_mass)

    m_SiO2_ox = dSi * (M_SiO2 / M_Si)
    m_MnO_ox = dMn * (M_MnO / M_Mn)
    m_P2O5_ox = dP * (M_P2O5 / (2.0 * M_P))

    denom = x_CaO_lime - B * x_SiO2_lime

    if denom <= 0:
        print('Помилка: неможливо забезпечити основність.')
        ok = False
        break

    m_lime = (B * m_SiO2_ox) / denom
    m_CaO_total = m_lime * x_CaO_lime
    m_SiO2_total = m_SiO2_ox + m_lime * x_SiO2_lime

    m_slag = (m_CaO_total + m_SiO2_total + m_MnO_ox + m_P2O5_ox) / k_main
    m_FeO_slag = w_FeO * m_slag
    m_Fe_to_slag = m_FeO_slag * (M_Fe / M_FeO)

    m_dust = k_dust * m_charge

    m_steel_new = m_charge - (dSi + dMn + dP + dC) - m_Fe_to_slag - m_dust

    if abs(m_steel_new - m_steel0) < eps_steel or iter_no >= max_iter:
        m_steel = m_steel_new
        Yield = (m_steel / m_charge) * 100.0
        break

    m_steel0 = m_steel_new
    iter_no += 1

# --- Вивід результатів ---
if ok:
    print('=' * 43)
    print(f'МАТЕРІАЛЬНИЙ БАЛАНС (Варіант {variant_no})')
    print('=' * 43)

    print(f'1. ШИХТА (База: {m_charge:.0f} кг):')
    print(f'   Чавун: {m_hm:8.2f} кг')
    print(f'   Брухт: {m_scr:8.2f} кг')

    print('-' * 43)
    print('2. ОКИСНЕНІ ДОМІШКИ (кг):')
    print(f'   Si: {dSi:6.3f}  |  Mn: {dMn:6.3f}  |  P: {dP:6.3f}')

    print('-' * 43)
    print('3. УТВОРЕНІ ОКСИДИ (кг):')
    print(f'   SiO2: {m_SiO2_ox:6.3f} | MnO: {m_MnO_ox:6.3f} | P2O5: {m_P2O5_ox:6.3f}')

    print('-' * 43)
    print('4. ШЛАК ТА ВАПНО:')
    print(f'   Вапно (для B={B:.1f}): {m_lime:8.2f} кг')
    print(f'   Маса шлаку:        {m_slag:8.2f} кг')

    print('-' * 43)
    print('5. ВТРАТИ ЗАЛІЗА:')
    print(f'   У шлак (чисте Fe): {m_Fe_to_slag:8.2f} кг')
    print(f'   З пилом:           {m_dust:8.2f} кг')

    print('-' * 43)
    print('6. РЕЗУЛЬТАТ:')
    print(f'   Маса сталі:        {m_steel:8.2f} кг')
    print(f'   Вихід придатного:  {Yield:8.2f} %')
    print('=' * 43)

