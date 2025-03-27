% Вхідні дані
% Масові частки компонентів, %
x_cao = 45;    % CaO
x_sio2 = 40;   % SiO2
x_al2o3 = 15;  % Al2O3

% Температура, °C
T = 500+273.15;

% Переведення масових часток у десятковий вигляд
x_cao_dec = x_cao / 100;
x_sio2_dec = x_sio2 / 100;
x_al2o3_dec = x_al2o3 / 100;

% Перевірка суми масових часток
x_sum = x_cao_dec + x_sio2_dec + x_al2o3_dec;
fprintf('\n1. Сума масових часток: %.4f\n', x_sum);

% Функції для розрахунку теплоємності компонентів
c_cao = @(T) 0.749 + 3.78e-4*T - 1.533e-7*T^2;
c_sio2 = @(T) 0.794 + 9.4e-4*T - 7.15e-7*T^2;
c_al2o3 = @(T) 0.786 + 5.97e-4*T - 2.98e-7*T^2;
c_slag = @(T) 0.694 + 8.95e-4*T - 1.18e-6*T^2 + 5.72e-10*T^3;

% Розрахунок теплоємностей компонентів при заданій температурі
c_cao_T = c_cao(T);
c_sio2_T = c_sio2(T);
c_al2o3_T = c_al2o3(T);

fprintf('\n2. Теплоємності компонентів при %d°C, кДж/(кг·°С):\n', T);
fprintf('   CaO: %.4f\n', c_cao_T);
fprintf('   SiO2: %.4f\n', c_sio2_T);
fprintf('   Al2O3: %.4f\n', c_al2o3_T);

% Метод адитивності
c_additive = x_cao_dec * c_cao_T + x_sio2_dec * c_sio2_T + x_al2o3_dec * c_al2o3_T;

fprintf('\n3. Внески компонентів у теплоємність суміші (метод адитивності), кДж/(кг·°С):\n');
fprintf('   CaO: %.4f\n', x_cao_dec * c_cao_T);
fprintf('   SiO2: %.4f\n', x_sio2_dec * c_sio2_T);
fprintf('   Al2O3: %.4f\n', x_al2o3_dec * c_al2o3_T);

fprintf('\n4. Середня питома теплоємність (метод адитивності):\n');
fprintf('   c = %.4f кДж/(кг·°С)\n', c_additive);

% Метод з урахуванням температурної залежності
c_temp = c_slag(T);

fprintf('\n5. Середня питома теплоємність (з урахуванням температурної залежності):\n');
fprintf('   c = %.4f кДж/(кг·°С)\n', c_temp);

% Порівняння методів
abs_diff = abs(c_additive - c_temp);
rel_diff = abs_diff / c_temp * 100;

fprintf('\n6. Порівняння методів:\n');
fprintf('   Абсолютна різниця: %.4f кДж/(кг·°С)\n', abs_diff);
fprintf('   Відносна різниця: %.2f%%\n', rel_diff);

fprintf('\n7. Перевірка достовірності:\n');
fprintf('   - Сума масових часток = 1: %s\n', ternary(abs(x_sum - 1) < 1e-10, '✓', '✗'));
fprintf('   - Розмірності збережені: ✓\n');

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
