% Визначення температури об'єкта з часом при охолодженні від 400 С до нормальних умов (20 °C) і візуалізація результату
% Вхідні дані:
t_start = 0; t_stop = 100; t10 = 10; t30 = 30;  % Час початку, закінчення та для визначення температури відповідно, хв.
T0 = 400; step_number = 1000; % Початкова температура, °C та кількість кроків.

tRange = linspace(t_start, t_stop, step_number); % змінна для інтервалу часу t=0 до 100 хвилин
[tSol, TSol] = ode45(@sinterODEfun, tRange, T0); % Виклик ode45, щоб розв'язати диференціальне рівняння
[~, idx10] = min(abs(tSol - t10)); T10 = TSol(idx10); % Пошук температури через 10 хвилин
[~, idx30] = min(abs(tSol - t30)); T30 = TSol(idx30); % Пошук температури через 30 хвилин

plot(tSol, TSol, "r", 'DisplayName', 'зміна температури від часу') % Графік охолодження від 400 °C до нормальних умов
hold on; scatter(t10, T10, 'r', 'filled'); scatter(t30, T30, 'r', 'filled');
text(t10+2, T10, sprintf('%.1f °C - температура через 10 хв', T10), 'Color', 'red', 'HorizontalAlignment', 'left');
text(t30+2, T30, sprintf('%.1f °C - температура через 30 хв', T30), 'Color', 'red', 'HorizontalAlignment', 'left');
plot([t_start t10], [T10 T10], 'r--', 'LineWidth', 0.7); plot([t10 t10], [t_start T10], 'r--', 'LineWidth', 0.7);
plot([t_start t30], [T30 T30], 'r--', 'LineWidth', 0.7); plot([t30 t30], [t_start T30], 'r--', 'LineWidth', 0.7);
title("Охолодження об'єктa (Matlab)") % Опис та налаштування графіка
legend("зміна температури від часу"); xlabel("час (хвилини)"); ylabel("температура (°C)")
xlim([t_start t_stop]); ylim([t_start T0]); hold off;

function dTdt = sinterODEfun(t,T) % Локальна функція що імплементує Зако́н Нью́тона — Рі́хмана
    a = 0.058;  % Коефіцієнт тепловіддачі, Вт/м²/К.
    Tn = 20;  % Температура при нормальних умовах, °C.
    dTdt = - a * (T - Tn);
end














