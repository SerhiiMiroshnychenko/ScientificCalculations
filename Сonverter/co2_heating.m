% Константи
R = 8.31446261815;  % Дж/(моль·К)
CAL_TO_JOULE = 4.1868;  % Дж/кал
P_STANDARD = 101325;  % Па (1 атм)
T_STANDARD = 273.15;  % К (0°C)

% Коефіцієнти для розрахунку теплоємності
a = 10.55;  % кал/(моль·К)
b = 2.16e-3;  % кал/(моль·К²)
c = -2.04e-5;  % кал·К²/(моль)

% Діапазон температур
T_MIN = 298;  % К
T_MAX = 2500;  % К

% Параметри процесу
volume = 100;  % м³
t1 = 25;  % °C
t2 = 1000;  % °C

% Перетворення температур
T1 = t1 + T_STANDARD;  % T1 = 25 + 273.15 = 298.15 К
T2 = t2 + T_STANDARD;  % T2 = 1000 + 273.15 = 1273.15 К

% 1. Виведення температур
fprintf('1. Температури:\n');
fprintf('   T1 = %.0f°C = %.2f К\n', t1, T1);
fprintf('   T2 = %.0f°C = %.2f К\n', t2, T2);

% Перевірка діапазону температур
if T1 < T_MIN || T1 > T_MAX
    fprintf('\nУВАГА: Температура %.2f К виходить за межі діапазону %d-%d К\n', T1, T_MIN, T_MAX);
    fprintf('Результати можуть бути неточними!\n');
end
if T2 < T_MIN || T2 > T_MAX
    fprintf('\nУВАГА: Температура %.2f К виходить за межі діапазону %d-%d К\n', T2, T_MIN, T_MAX);
    fprintf('Результати можуть бути неточними!\n');
end

% 2. Розрахунок кількості речовини
n = (P_STANDARD * volume) / (R * T1);
fprintf('\n2. Кількість речовини:\n');
fprintf('   n = %.2f моль\n', n);

% Перетворення коефіцієнтів
R_cal = R / CAL_TO_JOULE;
a_v = a - R_cal;
b_v = b;
c_v = c;

a_v = a_v * CAL_TO_JOULE;  % Дж/(моль·К)
b_v = b_v * CAL_TO_JOULE;  % Дж/(моль·К²)
c_v = c_v * CAL_TO_JOULE;  % Дж·К²/(моль)

% Аналітичний метод
Q_analytical = n * (...
    a_v * (T2 - T1) + ...                  % Інтеграл від константи a_v
    b_v * (T2^2 - T1^2) / 2 + ...         % Інтеграл від лінійного члена b_v·T
    c_v * (-1/T2 + 1/T1));                % Інтеграл від оберненого квадрата c_v·T⁻²

% Числовий метод
steps = 1000;
T = linspace(T1, T2, steps);
cv = @(T) a_v + b_v*T + c_v./(T.^2);
Q_numerical = n * trapz(T, cv(T));

% 3. Виведення результатів
fprintf('\n3. Результати розрахунку теплоти:\n');
fprintf('\tАналітичний метод: %.2f кДж\n', Q_analytical/1000);
fprintf('\tЧисловий метод: %.2f кДж\n', Q_numerical/1000);
fprintf('\tРізниця між методами: %.4f кДж\n', abs(Q_analytical - Q_numerical)/1000);
relative_error = abs(Q_analytical - Q_numerical)/abs(Q_analytical)*100;
fprintf('\tВідносна похибка: %.6f%%\n', relative_error);
