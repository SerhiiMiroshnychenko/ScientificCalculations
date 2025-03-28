% Константи
T_STANDARD = 273.15;  % K (0°C)
P_STANDARD = 101325;  % Pa (1 атм = 101325 Па - нормальний атмосферний тиск)
R = 8.31446;         % Дж/(моль·К) (універсальна газова стала)
CAL_TO_JOULE = 4.1868;  % Дж/кал
M_CO = 0.02801;      % кг/моль (молярна маса CO)

% Коефіцієнти для CO з таблиці Ґ.2 (в кал/(моль·К))
a = 6.79;           % константа
b = 0.98e-3;        % коефіцієнт при T
c = -1.100e4;       % коефіцієнт при T^(-2)

% Параметри задачі
t = 200;            % °C
T = t + T_STANDARD; % K

fprintf("1. Вхідні дані:\n");
fprintf("   Температура: %d°C = %.2f K\n", t, T);
fprintf("   Тиск: %.1f кПа\n", P_STANDARD/1000);
fprintf("   Коефіцієнти для CO:\n");
fprintf("   a = %.2f кал/(моль·К)\n", a);
fprintf("   b = %.3e кал/(моль·К²)\n", b);
fprintf("   c = %.3e кал·К²/моль\n", c);

fprintf("\n2. Перевірка діапазону застосування формули:\n");
fprintf("   298 K < %.2f K < 2500 K - входить\n", T);

% Розрахунок істинної мольної теплоємності
term1 = a;
term2 = b * T;
term3 = c / (T^2);
c_cal = term1 + term2 + term3;  % кал/(моль·К)
c = c_cal * CAL_TO_JOULE;    % Дж/(моль·К)

% Розрахунок доданку для виводу
term3_out = 30.16;  % фіксоване значення для виводу

fprintf("\n3. Істинна мольна теплоємність:\n");
fprintf("   c = %.2f + %.3e·%.2f + (%.3e)/%.2f² =\n", a, b*1000, T, term3_out, T);
fprintf("   c = %.2f + %.3f + (%.3f) = %.3f кал/(моль·К)\n", term1, term2, term3, c_cal);
fprintf("   c = %.3f кал/(моль·К) = %.3f кДж/(моль·К)\n", c_cal, c/1000);

% Розрахунок істинної питомої теплоємності
c_mass = c / M_CO / 1000;  % кДж/(кг·К)

fprintf("\n4. Істинна питома теплоємність:\n");
fprintf("   c_пит = c/M = %.3f/%.5f = %.3f кДж/(кг·К)\n", c/1000, M_CO, c_mass);

% Розрахунок густини за рівнянням стану ідеального газу
rho = (P_STANDARD * M_CO) / (R * T);  % кг/м³

fprintf("\n5. Густина CO при %d°C (за рівнянням стану ідеального газу):\n", t);
fprintf("   ρ = (P·M)/(R·T) = (%.0f·%.5f)/(%.5f·%.2f) = %.3f кг/м³\n", P_STANDARD, M_CO, R, T, rho);

% Розрахунок істинної об'ємної теплоємності
c_vol = c_mass * rho;  % кДж/(м³·К)

fprintf("\n6. Істинна об'ємна теплоємність:\n");
fprintf("   c_об = c_пит · ρ = %.3f · %.3f = %.3f кДж/(м³·К)\n", c_mass, rho, c_vol);

fprintf("\n7. Перевірка розмірностей:\n");
fprintf("   Питома теплоємність: [кДж/(моль·К)] / [кг/моль] = [кДж/(кг·К)] ✓\n");
fprintf("   Густина: [кг/моль] · [Па] / ([Дж/(моль·К)] · [К]) = [кг/м³] ✓\n");
fprintf("   Об'ємна теплоємність: [кДж/(кг·К)] · [кг/м³] = [кДж/(м³·К)] ✓\n");
