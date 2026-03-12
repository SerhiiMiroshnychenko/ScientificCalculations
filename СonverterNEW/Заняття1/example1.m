% Практична 1 - варіант 20
clc; clear;

variant_no = 20;

m_charge = 1000;

% Склад чавуну
Si_hm = 1.4;
Mn_hm = 0.7;
P_hm = 0.04;
C_hm = 4.6;

% Параметри шихти і процесу
w_scr = 0.3;
B = 3.4;

% Склад брухту
Si_scr = 0.25;
Mn_scr = 0.4;
P_scr = 0.02;
C_scr = 0.2;

% Залишковий вміст у сталі
Si_st = 0.005;
Mn_st = 0.1;
P_st = 0.015;
C_st = 0.1;

% Коефіціенти моделі
alpha_CO = 0.9;
x_CaO_lime = 0.9;
x_SiO2_lime = 0.02;
k_main = 0.8;
w_FeO = 0.15;
k_dust = 0.015;

% Молярні маси
% Молярні маси елементів
M_Fe = 55.845; % Молярна маса заліза
M_C = 12.011;  % Молярна маса вуглецю
M_Si = 28.0855; % Молярна маса кремнію
M_Mn = 54.938; % Молярна маса марганцю
M_P = 30.9738;  % Молярна маса фосфору
M_SiO2 = 60.0843;
M_MnO = 70.9374;
M_P2O5 = 141.9445;
M_CO = 28.01;
M_CO2 = 44.009;
M_FeO = 71.844;

% РОЗРАХУНОК
m_hm = m_charge * (1 - w_scr);
m_scr = m_charge * w_scr; % Mass of scrap

Si_in = m_hm * (Si_hm/100) + m_scr * (Si_scr/100);
Mn_in = m_hm * (Mn_hm/100) + m_scr * (Mn_scr/100);
P_in = m_hm * (P_hm/100) + m_scr * (P_scr/100);
C_in = m_hm * (C_hm/100) + m_scr * (C_scr/100);

% Параметри ітерацій
eps_steel = 0.1;
iter_no = 0;

m_steel0 = m_charge;
m_dust = k_dust * m_charge;

% Блок ітерацій
while true
    iter_no = iter_no + 1;

    Si_st_mass = m_steel0 * (Si_st/100);
    Mn_st_mass = m_steel0 * (Mn_st/100);
    P_st_mass = m_steel0 * (P_st/100);
    C_st_mass = m_steel0 * (C_st/100);

    dSi = Si_in - Si_st_mass;
    dMn = Mn_in - Mn_st_mass;
    dP = P_in - P_st_mass;
    dC = C_in - C_st_mass;

    m_SiO2_ox = dSi * (M_SiO2 / M_Si);
    m_MnO_ox = dMn * (M_MnO / M_Mn);
    m_P2O5_ox = dP * (M_P2O5 / (2*M_P));

    m_CO = alpha_CO * dC * (M_CO / M_C);
    m_CO2 = (1 - alpha_CO) * dC * (M_CO2 / M_C);

    denom = x_CaO_lime - B * x_SiO2_lime;
    if denom <= 0
        fprintf('Помилка: неможливо забезпечити основність.');
        break;
    end

    m_lime = (B * m_SiO2_ox) / denom;
    m_CaO_total = m_lime * x_CaO_lime;
    m_SiO2_total = m_SiO2_ox + m_lime * x_SiO2_lime;

    m_slag = (m_CaO_total + m_SiO2_total + m_MnO_ox + m_P2O5_ox) / k_main;

    m_FeO_slag = w_FeO * m_slag;
    m_Fe_to_slag = m_FeO_slag * (M_Fe / M_FeO);

    % Розрахунок уточненої маси сталі
    m_steel_new = m_charge - (dSi + dMn + dP + dC) - m_Fe_to_slag - m_dust;

    if abs(m_steel_new - m_steel0) < eps_steel
        m_steel = m_steel_new;
        Yield = (m_steel / m_charge);
        break;
    end

    m_steel0 = m_steel_new;

end

fprintf('===========================================\n');
fprintf('МАТЕРІАЛЬНИЙ БАЛАНС (Варіант %d)\n', variant_no);
fprintf('===========================================\n');

fprintf('1. ШИХТА (База: %.0f кг):\n', m_charge); % Інформація про шихту
fprintf('   Чавун: %8.2f кг\n', m_hm); % Вивід маси чавуну
fprintf('   Брухт: %8.2f кг\n', m_scr); % Вивід маси брухту

fprintf('-------------------------------------------\n');
fprintf('2. ОКИСНЕНІ ДОМІШКИ (кг):\n'); % Секція результатів по елементах
fprintf('   Si: %6.3f  |  Mn: %6.3f  |  P: %6.3f\n', dSi, dMn, dP);

fprintf('-------------------------------------------\n');
fprintf('3. УТВОРЕНІ ОКСИДИ (кг):\n'); % Секція результатів по оксидах
fprintf('   SiO2: %6.3f | MnO: %6.3f | P2O5: %6.3f\n', m_SiO2_ox, m_MnO_ox, m_P2O5_ox);

fprintf('-------------------------------------------\n');
fprintf('4. ГАЗОВА ФАЗА (кг):\n'); % Секція результатів по газах
fprintf('   CO:  %8.2f | CO2: %8.2f\n', m_CO, m_CO2);

fprintf('-------------------------------------------\n');
fprintf('5. ШЛАК ТА ВАПНО:\n'); % Секція флюсів та шлаку
fprintf('   Вапно (для B=%.1f): %8.2f кг\n', B, m_lime); % Витрата вапна
fprintf('   Маса шлаку:        %8.2f кг\n', m_slag); % Отримана маса шлаку

fprintf('-------------------------------------------\n');
fprintf('5. ВТРАТИ ЗАЛІЗА:\n'); % Секція втрат металу
fprintf('   У шлак (чисте Fe): %8.2f кг\n', m_Fe_to_slag); % Втрати заліза у шлак
fprintf('   З пилом:           %8.2f кг\n', m_dust); % Втрати заліза з пилом

fprintf('-------------------------------------------\n');
fprintf('6. РЕЗУЛЬТАТ:\n'); % Підсумкові показники
fprintf('   Маса сталі:        %8.2f кг\n', m_steel); % Кінцева маса сталі
fprintf('   Вихід придатного:  %8.2f %%\n', Yield); % Відсоток виходу металу
fprintf('===========================================\n');
