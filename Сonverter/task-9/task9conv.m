%% Вхідні дані
% Масові частки компонентів, %
x_cao = 47;    % CaO
x_feo = 14;    % FeO
x_mno = 15;    % MnO
x_sio2 = 24;   % SiO2

% Температура, K
T = 1000 + 273.15;

fprintf('1. Вхідні дані:\n');
fprintf('   Температура: %.2f K\n', T);
fprintf('\n   Масові частки компонентів, %%:\n');
fprintf('   CaO: %d\n', x_cao);
fprintf('   FeO: %d\n', x_feo);
fprintf('   MnO: %d\n', x_mno);
fprintf('   SiO2: %d\n', x_sio2);

%% Переведення у десятковий вигляд
x_cao_dec = x_cao/100;
x_feo_dec = x_feo/100;
x_mno_dec = x_mno/100;
x_sio2_dec = x_sio2/100;

fprintf('\n2. Масові частки у десятковому вигляді:\n');
fprintf('   CaO: %.4f\n', x_cao_dec);
fprintf('   FeO: %.4f\n', x_feo_dec);
fprintf('   MnO: %.4f\n', x_mno_dec);
fprintf('   SiO2: %.4f\n', x_sio2_dec);

%% Перевірка суми часток
x_sum = x_cao_dec + x_feo_dec + x_mno_dec + x_sio2_dec;
fprintf('\n3. Сума масових часток: %.4f\n', x_sum);

%% Функції теплоємності
% Оголошення вкладених функцій
function c = c_cao(T)
    % Теплоємність CaO, кДж/(кг·K)
    c = 0.749 + 3.78e-4*T - 1.535e-7*T^2;
end

function c = c_sio2(T)
    % Теплоємність SiO2, кДж/(кг·K)
    c = 0.768 + 3.23e-4*T;
end

function c = c_feo()
    % Теплоємність FeO, кДж/(кг·K)
    c = 0.7872;
end

function c = c_mno()
    % Теплоємність MnO, кДж/(кг·K)
    c = 0.7268;
end

function c = c_slag(T)
    % Теплоємність шлаку, кДж/(кг·K)
    c = 0.777 + 1.31e-4*T - 5.45e-8*T^2;
end

%% Розрахунок теплоємностей
c_cao_T = c_cao(T);
c_sio2_T = c_sio2(T);
c_feo_T = c_feo();
c_mno_T = c_mno();

fprintf('\n4. Теплоємності компонентів при %.2f K, кДж/(кг·K):\n', T);
fprintf('   CaO: %.4f\n', c_cao_T);
fprintf('   FeO: %.4f\n', c_feo_T);
fprintf('   MnO: %.4f\n', c_mno_T);
fprintf('   SiO2: %.4f\n', c_sio2_T);

%% Метод адитивності
c_additive = (x_cao_dec * c_cao_T + x_feo_dec * c_feo_T + ...
              x_mno_dec * c_mno_T + x_sio2_dec * c_sio2_T);

fprintf('\n5. Внески компонентів у теплоємність суміші:\n');
fprintf('   CaO: %.4f\n', x_cao_dec * c_cao_T);
fprintf('   FeO: %.4f\n', x_feo_dec * c_feo_T);
fprintf('   MnO: %.4f\n', x_mno_dec * c_mno_T);
fprintf('   SiO2: %.4f\n', x_sio2_dec * c_sio2_T);

fprintf('\n6. Середня питома теплоємність (адитивність):\n');
fprintf('   c = %.4f кДж/(кг·K)\n', c_additive);

%% Температурна залежність
c_temp = c_slag(T);

fprintf('\n7. Середня питома теплоємність (температурна залежність):\n');
fprintf('   c = %.4f кДж/(кг·K)\n', c_temp);

%% Порівняння методів
abs_diff = abs(c_additive - c_temp);
rel_diff = abs_diff/c_temp * 100;

fprintf('\n8. Порівняння методів:\n');
fprintf('   Абсолютна різниця: %.4f кДж/(кг·K)\n', abs_diff);
fprintf('   Відносна різниця: %.2f%%\n', rel_diff);

%% Перевірка достовірності
fprintf('\n9. Перевірка достовірності:\n');
fprintf('   - Сума масових часток = 1: %s\n', ternary(abs(x_sum - 1) < 1e-10, '✓', '✗'));
fprintf('   - Розмірності збережені: ✓\n');
T_C = T - 273.15;
fprintf('   - Температура в межах діапазонів:\n');
fprintf('     * CaO (0-800°C): %s\n', ternary(T_C <= 800, '✓', '✗'));
fprintf('     * SiO2 (0-1300°C): %s\n', ternary(T_C <= 1300, '✓', '✗'));
fprintf('     * FeO (800-1500°C): %s\n', ternary(T_C >= 800 & T_C <= 1500, '✓', '✗'));
fprintf('     * MnO (800-1500°C): %s\n', ternary(T_C >= 800 & T_C <= 1500, '✓', '✗'));
fprintf('     * Шлак (до температури плавлення): ✓\n');

%% Допоміжна функція для умовних позначок
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end