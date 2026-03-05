# Варіант 20

m_charge = 1000  # кг

Si_hm = 0.45
Mn_hm = 0.32
P_hm = 0.1
C_hm = 4.03

w_scr = 0.205
B = 3.2

Si_scr = 0.25
Mn_scr = 0.4
P_scr = 0.02
C_scr = 0.2

Si_st = 0.005
Mn_st = 0.1
P_st = 0.015
C_st = 0.1

alpha_CO = 0.9
x_CaO_lime = 0.9
x_SiO2_lime = 0.02
k_main = 0.8
w_FeO = 0.15
k_dust = 0.015

M_Si = 28.0855
M_SiO2 = 60.0843
M_Mn = 54.938
M_MnO = 70.9374
M_P = 30.9738
M_P2O5 = 141.9445
M_C = 12.011
M_CO = 28.010
M_CO2 = 44.009
M_Fe = 55.845
M_FeO = 71.844

m_hm = m_charge * ( 1 - w_scr)
m_scr = m_charge * w_scr

Si_in = m_hm * (Si_hm / 100) + m_scr * (Si_scr / 100)
Mn_in = m_hm * (Mn_hm / 100) + m_scr * (Mn_scr / 100)
P_in = m_hm * (P_hm / 100) + m_scr * (P_scr / 100)
C_in = m_hm * (C_hm / 100) + m_scr * (C_scr / 100)

m_dust = k_dust * m_charge

eps_steel = 0.1
iter_no = 0
m_steel0 = m_charge

while True:
    iter_no += 1
    
    Si_st_mass = m_steel0 * (Si_st / 100)
    Mn_st_mass = m_steel0 * (Mn_st / 100)
    P_st_mass = m_steel0 * (P_st / 100)
    C_st_mass = m_steel0 * (C_st / 100)
    
    dSi = Si_in - Si_st_mass
    dMn = Mn_in - Mn_st_mass
    dP = P_in - P_st_mass
    dC = C_in - C_st_mass
    
    m_SiO2_ox = dSi * (M_SiO2 / M_Si)
    m_MnO_ox = dMn * (M_MnO / M_Mn)
    m_P2O5_ox = dP * (M_P2O5 / (2 * M_P))
    
    m_CO = alpha_CO * dC * (M_CO / M_C)
    m_CO2 = (1 - alpha_CO) * dC * (M_CO2 / M_C)
    
    denom = x_CaO_lime - B * x_SiO2_lime
    
    if denom <= 0:
        print('Неможливо забезпечити основність')
        break
    
    m_lime = (B * m_SiO2_ox) / denom
    m_CaO_total = m_lime * x_CaO_lime
    m_SiO2_total = m_SiO2_ox + m_lime * x_SiO2_lime
    
    m_slag = (m_CaO_total + m_SiO2_total + m_MnO_ox + m_P2O5_ox) / k_main
    m_FeO_slag = w_FeO * m_slag
    m_Fe_to_slag = m_FeO_slag * (M_Fe / M_FeO)
    
    m_steel_new = m_charge - (dSi + dMn + dP + dC) - m_Fe_to_slag - m_dust
    
    if abs(m_steel_new - m_steel0) < eps_steel:
        print(f'Збіжність досягнуто на {iter_no} ітерації')
        m_steel = m_steel_new
        print(f'{m_steel = :.2f}')
        Yield = (m_steel / m_charge) * 100
        print(f'{Yield = :.2f}')
        break
    
    m_steel0 = m_steel_new
    
    
    