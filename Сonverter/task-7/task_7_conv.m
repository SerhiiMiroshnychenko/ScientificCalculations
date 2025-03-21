% Вхідні дані
% Масові частки компонентів, %
x_fe2o3 = 34.1;  % Fe2O3
x_h2o = 7.5;     % H2O
x_sio2 = 58.4;   % SiO2 (порожня порода)

% Теплоємності компонентів, кДж/(кг·°С)
c_fe2o3 = 0.61;  % Fe2O3
c_h2o = 4.2;     % H2O
c_sio2 = 1.17;   % SiO2

% Переведення масових часток у десятковий вигляд
x_fe2o3_dec = x_fe2o3 / 100;
x_h2o_dec = x_h2o / 100;
x_sio2_dec = x_sio2 / 100;

% Перевірка суми масових часток
x_sum = x_fe2o3_dec + x_h2o_dec + x_sio2_dec;
fprintf('\n1. Сума масових часток: %.4f\n', x_sum);

% Розрахунок питомої теплоємності суміші
c_mix = x_fe2o3_dec * c_fe2o3 + x_h2o_dec * c_h2o + x_sio2_dec * c_sio2;

fprintf('\n2. Внески компонентів у теплоємність суміші, кДж/(кг·°С):\n');
fprintf('   Fe2O3: %.5f\n', x_fe2o3_dec * c_fe2o3);
fprintf('   H2O: %.5f\n', x_h2o_dec * c_h2o);
fprintf('   SiO2: %.5f\n', x_sio2_dec * c_sio2);

fprintf('\n3. Питома теплоємність суміші:\n');
fprintf('   c = %.5f кДж/(кг·°С)\n', c_mix);

% Переведення в одиниці SI
c_mix_SI = c_mix * 1000;  % кДж -> Дж

fprintf('\n4. Питома теплоємність в одиницях SI:\n');
fprintf('   c = %.2f Дж/(кг·К)\n', c_mix_SI);

fprintf('\n5. Перевірка достовірності:\n');
range_check = min([c_fe2o3, c_h2o, c_sio2]) <= c_mix && c_mix <= max([c_fe2o3, c_h2o, c_sio2]);
sum_check = abs(x_sum - 1) < 1e-10;

fprintf('   - Результат в межах теплоємностей компонентів (%.2f - %.2f кДж/(кг·°С)): %s\n', min([c_fe2o3, c_h2o, c_sio2]), max([c_fe2o3, c_h2o, c_sio2]), ternary(range_check, '✓', '✗'));
fprintf('   - Сума масових часток = 1: %s\n', ternary(sum_check, '✓', '✗'));
fprintf('   - Розмірності збережені та переведені в SI: ✓\n');

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
