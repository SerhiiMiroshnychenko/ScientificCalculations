% Скрипт для визначення температури агломерату з часом
% при охолодженні від 400 С до нормальних умов (20 °C)
% і візуалізація результату

% 1. ВХІДНІ ДАННІ
% Обмеження по часу
t_start = 0; % Час початку охолодження, хв.
t_stop = 100; % Час закінчення процесу, хв.
t10 = 10; % Час для визначення температури, 10 хв.
t30 = 30; % Час для визначення температури, 30 хв.

%  Кількість кроків по часу
step_number = 1000;

% Задання температури
T0 = 400; % змінна для початкової температури, T(0)=400 °C.

% 3. РОЗВ'ЯЗАННЯ
% змінна для інтервалу часу t=0 до 100 хвилин
tRange = linspace(t_start, t_stop, step_number);

% Виклик ode45, щоб розв'язати диференціальне рівняння
[tSol, TSol] = ode45(@sinterODEfun, tRange, T0);

% Пошук температури через 10 хвилин
[~, idx10] = min(abs(tSol - t10));
T10 = TSol(idx10);

% Пошук температури через 30 хвилин
[~, idx30] = min(abs(tSol - t30));
T30 = TSol(idx30);

% 4. ВІЗУАЛІЗАЦІЯ

% Графік охолодження від 400 °C до нормальних умов
plot(tSol, TSol, "r", 'DisplayName', 'зміна температури від часу')
hold on;

% Додавання точки та підпису температури через 10 хвилин
scatter(t10, T10, 'r', 'filled');
text(t10+2, T10, sprintf('%.1f °C - температура через 10 хв охолодження', T10), ...
    'Color', 'red', 'HorizontalAlignment', 'left');

% Додавання точки та підпису температури через 30 хвилин
scatter(t30, T30, 'r', 'filled');
text(t30+2, T30, sprintf('%.1f °C - температура через 30 хв охолодження', T30), ...
    'Color', 'red', 'HorizontalAlignment', 'left');

% Додавання пунктирних ліній
plot([t_start t10], [T10 T10], 'r--', 'LineWidth', 0.7);
plot([t10 t10], [t_start T10], 'r--', 'LineWidth', 0.7);
plot([t_start t30], [T30 T30], 'r--', 'LineWidth', 0.7);
plot([t30 t30], [t_start T30], 'r--', 'LineWidth', 0.7);

% Опис та налаштування графіка
title("Охолодження агломерату (Matlab)")
legend("зміна температури від часу")
xlabel("час (хвилини)")
ylabel("температура (°C)")
grid off

% Встановлення межі осі x для початку графіка з 0
xlim([t_start t_stop])
ylim([t_start T0])
hold off

% 2. ВИЗНАЧЕННЯ ФУНКЦІЇ
function dTdt = sinterODEfun(t,T) % Локальна функція що імплементує Зако́н Нью́тона — Рі́хмана
    a = 0.058; % коефіцієнт тепловіддачі, Вт/м²/К.
    Tn = 20; % температура при нормальних умовах, °C.
    dTdt = - a * (T - Tn);
end
