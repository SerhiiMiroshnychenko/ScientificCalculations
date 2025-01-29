% Визначаємо основні фізичні параметри системи
a = 1e-6; % Коефіцієнт температуропровідності залізорудного агломерату, м²/с
dx = 0.01; % Крок сітки по x, м (10 мм)
dy = 0.01; % Крок сітки по y, м (10 мм)

% Визначаємо розміри розрахункової сітки
sizex = 250; % Кількість точок по x (2500 мм / 10 мм)
sizey = 40;  % Кількість точок по y (400 мм / 10 мм)

% Задаємо параметри часової еволюції
tStart = 0;      % Початковий час, с
time_step = 40;  % Крок по часу для перевірки температури, с
max_time = 500000; % Максимальний час розрахунку, с (≈5.8 діб)

% Граничні умови
T_ambient = 20;  % Температура навколишнього середовища, °C
T_initial = 400; % Початкова температура шихти після агломерації, °C

% Допоміжні функції
function result = check_temperature(u, target_temp, sizey, sizex, tolerance)
    u_reshaped = reshape(u, sizey, sizex);
    result = all(abs(u_reshaped(:) - target_temp) <= tolerance);
end

function time_str = format_time(total_seconds)
    hours = total_seconds / 3600;
    time_str = sprintf('%.1f секунд (%.2f годин)', total_seconds, hours);
end

function dudt = f_2D_flattened(t, u, sizey, sizex, a, dx, dy)
    % Перетворюємо одновимірний масив назад у двовимірний
    u = reshape(u, sizey, sizex);

    % Створюємо масив для похідних
    unew = zeros(sizey, sizex);

    % Розраховуємо похідні для всіх внутрішніх точок
    for i = 2:sizey-1
        for j = 2:sizex-1
            unew(i,j) = (u(i+1,j) - 2*u(i,j) + u(i-1,j)) * a / dx^2 + ...
                        (u(i,j+1) - 2*u(i,j) + u(i,j-1)) * a / dy^2;
        end
    end

    % Повертаємо розгорнутий одновимірний масив
    dudt = unew(:);
end

% Створення масиву температур та встановлення початкових умов
T = ones(sizey, sizex) * T_initial;

% Встановлюємо граничні умови
T(1, :) = T_ambient;   % Нижня границя
T(end, :) = T_ambient; % Верхня границя
T(:, 1) = T_ambient;   % Ліва границя
T(:, end) = T_ambient; % Права границя

fprintf('\nРозрахунок охолодження шару залізорудної агломераційної шихти...\n');
fprintf('Розмір шару: %.0f x %.0f мм\n', sizex * dx * 1000, sizey * dy * 1000);
fprintf('Кількість точок сітки: %d x %d\n', sizex, sizey);
fprintf('Початкова температура: %d°C\n', T_initial);
fprintf('Температура навколишнього середовища: %d°C\n', T_ambient);

% Ініціалізуємо змінні для зберігання проміжних результатів
current_time = tStart;
solutions = {};
times = [];

while current_time < max_time
    tspan = [current_time, current_time + time_step];
    if current_time == tStart
        y0 = T(:);
    else
        y0 = solutions{end}.y(:,end);
    end

    % Розв'язуємо систему рівнянь
    [t, y] = ode45(@(t,y) f_2D_flattened(t, y, sizey, sizex, a, dx, dy), tspan, y0);

    sol.t = t;
    sol.y = y';
    solutions{end+1} = sol;
    times = [times; t];

    % Перевіряємо чи досягнута цільова температура
    current_temp = reshape(y(end,:)', sizey, sizex);
    if check_temperature(y(end,:)', T_ambient, sizey, sizex, 1.0)
        fprintf('\nШар охолонув до температури %d±1°C за %s\n', ...
            T_ambient, format_time(current_time));
        break;
    end

    current_time = current_time + time_step;

    % Виводимо інформацію про прогрес охолодження
    if mod(current_time, 3600) == 0
        max_temp = max(current_temp(:));
        min_temp = min(current_temp(:));
        avg_temp = mean(current_temp(:));
        fprintf('Час: %s, температура: мін = %.1f°C, середня = %.1f°C, макс = %.1f°C\n', ...
            format_time(current_time), min_temp, avg_temp, max_temp);
    end
end

% Візуалізація результатів
[x_list, y_list] = meshgrid((0:sizex-1)*dx, (0:sizey-1)*dy);

viz_indices = [1, ...                    % Початок
               floor(length(times)/3), ... % 1/3 часу
               floor(2*length(times)/3),... % 2/3 часу
               length(times)];             % Кінець

titles = {'Початковий розподіл температури', ...
          'Розподіл температури через 1/3 часу охолодження', ...
          'Розподіл температури через 2/3 часу охолодження', ...
          'Кінцевий розподіл температури'};

for idx = 1:length(viz_indices)
    plot_idx = viz_indices(idx);
    figure('Position', [100, 100, 600, 400]);

    % Отримуємо дані для поточного часового кроку
    current_sol_idx = ceil(plot_idx / length(solutions{1}.t));
    current_t_idx = mod(plot_idx-1, length(solutions{1}.t)) + 1;
    current_data = reshape(solutions{current_sol_idx}.y(:,current_t_idx), sizey, sizex);

    % Встановлюємо діапазон температур
    if idx == 1
        clim = [T_ambient, T_initial];
    else
        clim = [T_ambient, max(max(current_data(:)), T_ambient + 1)];
    end

    % Створюємо контурний графік
    contourf(x_list, y_list, current_data, 20, 'LineStyle', 'none');
    colorbar;
    caxis(clim);
    colormap('jet');

    title(sprintf('%s\nt = %s', titles{idx}, ...
          format_time(times(plot_idx))), 'FontSize', 8);
    xlabel('x, м', 'FontSize', 8);
    ylabel('y, м', 'FontSize', 8);

    % Встановлюємо однакові пропорції для осей
    axis equal;
    set(gca, 'FontSize', 7);
end