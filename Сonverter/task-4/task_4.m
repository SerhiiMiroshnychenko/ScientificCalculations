% Константи
T_STANDARD = 273.15;  % К (0°C)
P_STANDARD = 101325;  % Па (1 атм)
R_CAL = 1.986;        % кал/(моль·К)
R = 8.31446261815;    % Дж/(моль·К)
CAL_TO_JOULE = 4.1868; % Дж/кал
M_O2 = 0.032;         % кг/моль (молярна маса O₂)

% Коефіцієнти для O₂ з таблиці 1.1 (в кал/(моль·К))
a = 7.16;
b = 1.0e-3;
c = -0.40e5;

% Параметри задачі
t1 = 0;             % °C
t2 = 1000;          % °C
t_avg = (t1 + t2)/2; % °C (середня температура)

T1 = t1 + T_STANDARD; % К
T2 = t2 + T_STANDARD; % К
T_avg = t_avg + T_STANDARD; % К

fprintf("1. Вхідні дані:\n");
fprintf("   Температурний інтервал: %d°C - %d°C\n", t1, t2);
fprintf("   T1 = %.2f К\n", T1);
fprintf("   T2 = %.2f К\n", T2);
fprintf("   T_сер = %.2f К\n", T_avg);
fprintf("   Тиск: %.1f кПа\n", P_STANDARD/1000);
fprintf("   Коефіцієнти для O₂:\n");
fprintf("   a = %.2f кал/(моль·К)\n", a);
fprintf("   b = %.3e кал/(моль·К²)\n", b);
fprintf("   c = %.3e кал·К²/моль\n", c);

% Розрахунок середньої мольної теплоємності при сталому об'ємі
cv_cal_analytical = (a - R_CAL) + b*(T1 + T2)/2 + c*(-1/T2 + 1/T1)/(T2 - T1);  % кал/(моль·К)
cv_analytical = cv_cal_analytical * CAL_TO_JOULE;  % Дж/(моль·К)

fprintf("\n2. Середня мольна теплоємність при сталому об'ємі (аналітичний метод):\n");
fprintf("   cv_сер = %.3f кал/(моль·К) = %.3f Дж/(моль·К)\n", cv_cal_analytical, cv_analytical);

% Числовий метод (метод трапецій)
steps = 100000;
T = linspace(T1, T2, steps);
cv_values = (a - R_CAL) + b*T + c./(T.*T);
cv_cal_trapz = trapz(T, cv_values) / (T2 - T1);
cv_trapz = cv_cal_trapz * CAL_TO_JOULE;

fprintf("\n3. Середня мольна теплоємність при сталому об'ємі (метод трапецій):\n");
fprintf("   cv_сер = %.3f кал/(моль·К) = %.3f Дж/(моль·К)\n", cv_cal_trapz, cv_trapz);

% Середня питома теплоємність
cv_mass = cv_analytical / M_O2;

fprintf("\n4. Середня питома теплоємність:\n");
fprintf("   cv_пит = %.2f Дж/(кг·К)\n", cv_mass);

% Густина при середній температурі
rho = (P_STANDARD * M_O2) / (R * T_avg);

fprintf("\n5. Густина O₂ при середній температурі %.1f°C:\n", t_avg);
fprintf("   ρ = %.4f кг/м³\n", rho);

% Середня об'ємна теплоємність
cv_vol = cv_mass * rho;

fprintf("\n6. Середня об'ємна теплоємність:\n");
fprintf("   cv_об = %.2f Дж/(м³·К)\n", cv_vol);

fprintf("\n7. Перевірка розмірностей:\n");
fprintf("   Мольна теплоємність: [кал/(моль·К)] · [Дж/кал] = [Дж/(моль·К)] ✓\n");
fprintf("   Питома теплоємність: [Дж/(моль·К)] / [кг/моль] = [Дж/(кг·К)] ✓\n");
fprintf("   Об'ємна теплоємність: [Дж/(кг·К)] · [кг/м³] = [Дж/(м³·К)] ✓\n");
