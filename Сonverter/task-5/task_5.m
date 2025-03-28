% Константи
T_STANDARD = 273.15; % К (0°C)

% Коефіцієнти для теплоємності заліза
a = 0.4613; % кДж/(кг·К)
b = 2.12e-4; % кДж/(кг·К²)
c = -6.87e-7; % кДж/(кг·К³)

% Вхідні дані
t1 = 100; % °C
t2 = 200; % °C

% Переведення температур в Кельвіни
T1 = t1 + T_STANDARD; % К
T2 = t2 + T_STANDARD; % К


% 1. Аналітичний метод
c_analytical = (a * (T2 - T1) + (b/2) * (T2^2 - T1^2) + (c/3) * (T2^3 - T1^3)) / (T2 - T1);

fprintf('\n2. Середня питома теплоємність (аналітичний метод):\n');
fprintf('   c_сер = %.4f кДж/(кг·К)\n', c_analytical);

% 2. Метод трапецій
steps = 100000; % кількість кроків інтегрування
T = linspace(T1, T2, steps); % рівномірне розбиття інтервалу [T1, T2]
c_values = a + b*T + c*T.^2; % теплоємність як функція від T
c_trapz = trapz(T, c_values) / (T2 - T1);

fprintf('\n3. Середня питома теплоємність (метод трапецій):\n');
fprintf('   c_сер = %.4f кДж/(кг·К)\n', c_trapz);

% 3. Метод Сімпсона
c_simpson = integral(@(T) a + b*T + c*T.^2, T1, T2) / (T2 - T1);

fprintf('\n4. Середня питома теплоємність (метод Сімпсона):\n');
fprintf('   c_сер = %.4f кДж/(кг·К)\n', c_simpson);


