% Аналіз залежності вмісту вуглецю від розміру фракцій
% Експеримент №1

% Розміри фракцій (середнє значення для кожного діапазону)
x = [10.5; 10.4; 8.5; 5.3; 3.1; 0.5];  % приблизні середні значення діапазонів

% Значення вмісту вуглецю для експерименту №1
C = [1.79; 2.88; 3.11; 3.66; 4.35; 4.42];

% Створюємо фігуру для графіка
figure('Position', [100 100 800 600]);

% Будуємо експериментальні точки
plot(x, C, 'bo', 'MarkerSize', 10, 'LineWidth', 1.5, 'DisplayName', 'Експериментальні дані')
hold on

% Створюємо точки для побудови графіків
x_fit = linspace(min(x), max(x), 100);

% Знаходимо коефіцієнти для різних типів апроксимації
p1 = polyfit(x, C, 1);  % лінійна
p2 = polyfit(x, C, 2);  % квадратична
p3 = polyfit(x, C, 3);  % кубічна
p4 = polyfit(x, C, 4);  % 4-го ступеню

% Додаткові типи апроксимації
fo_exp = fitoptions('exp1');
fo_exp.StartPoint = [3.0 -0.1];  % Початкові значення для a*exp(b*x)

fo_log = fitoptions('Method', 'NonlinearLeastSquares');
fo_log.StartPoint = [-0.5 4];  % Початкові значення для a*log(x)+b

fo_pow = fitoptions('power1');
fo_pow.StartPoint = [3.0 -0.1];  % Початкові значення для a*x^b

f_exp = fit(x, C, 'exp1', fo_exp);
f_log = fit(x, C, 'a*log(x)+b', fo_log);
f_pow = fit(x, C, 'power1', fo_pow);

% Будуємо криві апроксимації
plot(x_fit, polyval(p1, x_fit), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Лінійна')
plot(x_fit, polyval(p2, x_fit), 'm-.', 'LineWidth', 1.5, 'DisplayName', 'Квадратична')
plot(x_fit, polyval(p3, x_fit), 'c-', 'LineWidth', 1.5, 'DisplayName', 'Кубічна')
plot(x_fit, polyval(p4, x_fit), 'y-', 'LineWidth', 1.5, 'DisplayName', '4-го ступеню')
plot(x_fit, f_exp(x_fit), '--', 'Color', [0.5 0 0.5], 'LineWidth', 1.5, 'DisplayName', 'Експоненціальна')
plot(x_fit, f_log(x_fit), '-.', 'Color', [0 0.5 0], 'LineWidth', 1.5, 'DisplayName', 'Логарифмічна')
plot(x_fit, f_pow(x_fit), ':', 'Color', [0.8 0.4 0], 'LineWidth', 1.5, 'DisplayName', 'Степенева')

% Налаштування графіка
title('Залежність вмісту вуглецю від розміру фракції (Експеримент №1)', 'FontSize', 14)
xlabel('Розмір фракції, мм', 'FontSize', 12)
ylabel('Вміст вуглецю, %', 'FontSize', 12)
grid on
legend('Location', 'best', 'FontSize', 10)
grid minor

% Виводимо рівняння
fprintf('\nРівняння залежності вмісту вуглецю (Експеримент №1):\n')
fprintf('1. Поліноміальні апроксимації:\n')
fprintf('Лінійне: C = %.4fx + %.4f\n', p1(1), p1(2))
fprintf('Квадратичне: C = %.4fx^2 + %.4fx + %.4f\n', p2(1), p2(2), p2(3))
fprintf('Кубічне: C = %.4fx^3 + %.4fx^2 + %.4fx + %.4f\n', p3(1), p3(2), p3(3), p3(4))
fprintf('4-го ступеню: C = %.4fx^4 + %.4fx^3 + %.4fx^2 + %.4fx + %.4f\n', p4(1), p4(2), p4(3), p4(4), p4(5))

fprintf('\n2. Додаткові типи апроксимації:\n')
fprintf('Експоненціальна: C = %.4f * exp(%.4fx)\n', f_exp.a, f_exp.b)
fprintf('Логарифмічна: C = %.4f * log(x) + %.4f\n', f_log.a, f_log.b)
fprintf('Степенева: C = %.4f * x^(%.4f)\n', f_pow.a, f_pow.b)

% Розрахунок R^2
y_mean = mean(C);
SS_tot = sum((C - y_mean).^2);

y_pred_1 = polyval(p1, x);
y_pred_2 = polyval(p2, x);
y_pred_3 = polyval(p3, x);
y_pred_4 = polyval(p4, x);
y_pred_exp = f_exp(x);
y_pred_log = f_log(x);
y_pred_pow = f_pow(x);

R2_1 = 1 - sum((C - y_pred_1).^2)/SS_tot;
R2_2 = 1 - sum((C - y_pred_2).^2)/SS_tot;
R2_3 = 1 - sum((C - y_pred_3).^2)/SS_tot;
R2_4 = 1 - sum((C - y_pred_4).^2)/SS_tot;
R2_exp = 1 - sum((C - y_pred_exp).^2)/SS_tot;
R2_log = 1 - sum((C - y_pred_log).^2)/SS_tot;
R2_pow = 1 - sum((C - y_pred_pow).^2)/SS_tot;

fprintf('\nКоефіцієнти детермінації (R^2):\n')
fprintf('1. Для поліноміальних апроксимацій:\n')
fprintf('Лінійна: %.4f\n', R2_1)
fprintf('Квадратична: %.4f\n', R2_2)
fprintf('Кубічна: %.4f\n', R2_3)
fprintf('4-го ступеню: %.4f\n', R2_4)

fprintf('\n2. Для додаткових типів апроксимації:\n')
fprintf('Експоненціальна: %.4f\n', R2_exp)
fprintf('Логарифмічна: %.4f\n', R2_log)
fprintf('Степенева: %.4f\n', R2_pow)

% Зберігаємо графік
savefig('carbon_analysis_exp1.fig')
print('carbon_analysis_exp1', '-dpng', '-r300')
