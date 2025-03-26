import csv
from datetime import datetime
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.colors as mcolors
import numpy as np

# Додаємо підтримку українських символів через mplfonts
try:
    from mplfonts import use_font

    # Використовуємо шрифт, який підтримує кирилицю
    use_font('Times New Roman')
    has_mplfonts = True
except ImportError:
    print("Для коректного відображення українських символів потрібно встановити бібліотеку mplfonts:")
    print("pip install mplfonts")
    print("mplfonts init")
    has_mplfonts = False

# Налаштування для наукового стилю
plt.style.use('science')

# Переконуємося, що LaTeX вимкнено
plt.rcParams['text.usetex'] = False

# Налаштовуємо безпечне відображення символу мінус
plt.rcParams['axes.unicode_minus'] = False

# Отримуємо кольори з TABLEAU_COLORS для більш професійного вигляду
colors = list(mcolors.TABLEAU_COLORS.values())


def _read_csv_data(data_file):
    # Read CSV data directly
    data = []
    with open(data_file, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                # Обрізаємо мікросекунди з дат
                if 'date_order' in row:
                    date_str = row['date_order'].split('.')[0]  # Видаляємо мікросекунди
                    row['date_order'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue

            try:
                if 'partner_create_date' in row:
                    date_str = row['partner_create_date'].split('.')[0]  # Видаляємо мікросекунди
                    row['partner_create_date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue

            # Ensure required fields are present
            if not all(field in row for field in
                       ['order_id', 'partner_id', 'date_order', 'state', 'amount_total',
                        'partner_create_date', 'user_id', 'payment_term_id']):
                print(f"Missing required fields in row: {row}")
                continue

            data.append(row)

    print(f"Successfully read {len(data)} rows from CSV")
    return data


def _prepare_partner_age_success_data(data_file):
    """Підготовка даних для графіка успішності партнерів за віком"""

    print("\nПочинаємо підготовку даних _prepare_partner_age_success_data")

    # Read CSV data
    data = _read_csv_data(data_file)

    # Розраховуємо вік партнера для кожного замовлення
    partner_age_data = []
    print("\nОбробка даних про вік партнерів...")

    missing_create_date = 0
    missing_order_date = 0
    processed_rows = 0
    error_rows = 0

    for row in data:
        if not row.get('partner_create_date'):
            missing_create_date += 1
            continue
        if not row.get('date_order'):
            missing_order_date += 1
            continue

        try:
            # Якщо дати в строковому форматі, конвертуємо їх
            if isinstance(row['partner_create_date'], str):
                row['partner_create_date'] = datetime.strptime(row['partner_create_date'], '%Y-%m-%d %H:%M:%S')
            if isinstance(row['date_order'], str):
                row['date_order'] = datetime.strptime(row['date_order'], '%Y-%m-%d %H:%M:%S')

            partner_age = max(0, (row['date_order'] - row['partner_create_date']).days)
            partner_age_data.append((partner_age, row['state'] == 'sale'))
            processed_rows += 1
        except Exception as e:
            error_rows += 1

    print(f"\nПідсумок обробки:")
    print(f"- Всього рядків: {len(data)}")
    print(f"- Відсутні дати створення: {missing_create_date}")
    print(f"- Відсутні дати замовлення: {missing_order_date}")
    print(f"- Успішно оброблено: {processed_rows}")
    print(f"- Помилки: {error_rows}")

    if not partner_age_data:
        print("УВАГА: Не знайдено коректних даних про вік партнерів")
        return None

    print(f"Знайдено {len(partner_age_data)} коректних замовлень для аналізу")
    # Сортуємо за віком партнера
    partner_age_data.sort(key=lambda x: x[0])

    total_orders = len(partner_age_data)
    print(f"Всього замовлень для аналізу: {total_orders}")

    # Визначаємо кількість груп
    num_groups = min(30, total_orders // 50)
    if num_groups < 5:
        num_groups = 5
    print(f"Кількість груп: {num_groups}")

    # Розраховуємо розмір кожної групи
    group_size = total_orders // num_groups
    remainder = total_orders % num_groups
    print(f"Розмір групи: {group_size}, залишок: {remainder}")

    # Ініціалізуємо результат
    result = {
        'ranges': [],
        'rates': [],
        'orders_count': []
    }

    # Розбиваємо на групи
    start_idx = 0
    for i in range(num_groups):
        current_group_size = group_size + (1 if i < remainder else 0)
        if current_group_size == 0:
            break

        end_idx = start_idx + current_group_size
        group_orders = partner_age_data[start_idx:end_idx]

        # Рахуємо статистику для групи
        min_age = group_orders[0][0]
        max_age = group_orders[-1][0]
        successful = sum(1 for _, is_success in group_orders if is_success)
        success_rate = (successful / len(group_orders)) * 100

        print(f"\nГрупа {i}:")
        print(f"- Замовлень: {len(group_orders)}")
        print(f"- Рівень успішності: {success_rate:.1f}%")
        print(f"- Діапазон віку: {min_age}-{max_age} днів")

        # Форматуємо діапазон
        if max_age >= 365:
            range_str = f'{min_age / 365:.1f}р-{max_age / 365:.1f}р'
        elif max_age >= 30:
            range_str = f'{min_age / 30:.0f}м-{max_age / 30:.0f}м'
        else:
            range_str = f'{min_age}д-{max_age}д'

        # Додаємо дані до результату
        result['ranges'].append(range_str)
        result['rates'].append(success_rate)
        result['orders_count'].append(len(group_orders))

        start_idx = end_idx

    print("\nУспішно підготовлено дані про вік та успішність партнерів")
    return result


def _create_partner_age_success_chart(data, output_path):
    """Створення графіка успішності партнерів за віком"""

    # Встановлюємо розміри у пропорціях як у figures.py
    textwidth = 10  # ширина для наукової статті
    aspect_ratio = 0.6  # співвідношення сторін для наукового вигляду
    scale = 1.0
    figwidth = textwidth * scale
    figheight = figwidth * aspect_ratio

    plt.figure(figsize=(figwidth, figheight))
    ax = plt.gca()

    # Встановлюємо тонкі лінії для осей як у figures.py
    width = 0.5
    ax.spines["left"].set_linewidth(width)
    ax.spines["bottom"].set_linewidth(width)
    ax.spines["right"].set_linewidth(width)
    ax.spines["top"].set_linewidth(width)
    ax.tick_params(width=width)

    # Фільтруємо точки з нульовою кількістю ордерів
    x_points = []
    y_points = []
    counts = []
    for i, (rate, count) in enumerate(zip(data['rates'], data['orders_count'])):
        if count > 0:
            x_points.append(i)
            y_points.append(rate)
            counts.append(count)

    # Створюємо маркери в стилі figures.py
    markers = ["+", "x", "d", ".", "*"]

    # Використовуємо червоний і зелений кольори для точок залежно від успішності
    color_below_50 = colors[3]  # червоний колір з TABLEAU_COLORS
    color_above_50 = '#00cc00'  # зелений колір як у оригінальному скрипті

    # Розділяємо точки на групи за рівнем успішності
    below_50_indices = [i for i, rate in enumerate(y_points) if rate < 50]
    above_50_indices = [i for i, rate in enumerate(y_points) if rate >= 50]

    # Розраховуємо розміри точок в залежності від кількості замовлень (але менші ніж раніше для наукового вигляду)
    sizes_below = [max(40, min(80, counts[i] / 5)) for i in below_50_indices]
    sizes_above = [max(40, min(80, counts[i] / 5)) for i in above_50_indices]

    # Малюємо точки окремо для різних груп
    if below_50_indices:
        plt.scatter([x_points[i] for i in below_50_indices],
                    [y_points[i] for i in below_50_indices],
                    s=sizes_below, alpha=0.8, c=color_below_50,
                    marker='o', label='Успішність < 50%')

    if above_50_indices:
        plt.scatter([x_points[i] for i in above_50_indices],
                    [y_points[i] for i in above_50_indices],
                    s=sizes_above, alpha=0.8, c=color_above_50,
                    marker='o', label='Успішність > 50%')

    # Розраховуємо середню кількість ордерів на точку
    avg_orders = sum(counts) // len(counts) if counts else 0

    plt.title(
        f'Рівень успішності за віком партнера\n(кожна точка представляє ~{avg_orders} замовлень)',
        pad=20, fontsize=14)
    plt.xlabel('Вік партнера (д=днів, м=місяців, р=років)', fontsize=14)
    plt.ylabel('Рівень успішності (%)', fontsize=14)

    # Налаштовуємо осі
    plt.ylim(0, 100)

    # Показуємо всі мітки, якщо їх менше 10, інакше кожну другу
    if len(data['ranges']) <= 10:
        plt.xticks(range(len(data['ranges'])), data['ranges'],
                   rotation=45, ha='right', fontsize=14)
    else:
        # Використовуємо кожну другу мітку
        plt.xticks(range(len(data['ranges']))[::2],
                   [data['ranges'][i] for i in range(0, len(data['ranges']), 2)],
                   rotation=45, ha='right', fontsize=14)

    # Додаємо легенду в стилі figures.py
    legend = plt.legend(fancybox=False, edgecolor="black", fontsize=14)
    legend.get_frame().set_linewidth(0.5)

    # Налаштовуємо розмір шрифту для осі Y
    plt.yticks(fontsize=14)

    # Додаємо сітку для кращої читабельності
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)

    # Зберігаємо з більшим dpi для кращої якості
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300)

    # Створюємо також PDF версію для використання в LaTeX
    plt.savefig(f"{output_path}.pdf")

    print(f"Графік збережено до {output_path}.png та {output_path}.pdf")
    plt.close()


def create_partner_age_success_chart(csv_path, output_path):
    """
    Створює графік аналізу історії клієнтів

    Args:
        csv_path (str): Шлях до CSV файлу з даними
        output_path (str): Базовий шлях для збереження графіка (без розширення)

    Returns:
        bool: True у разі успіху
    """
    try:
        # Підготовка даних
        data = _prepare_partner_age_success_data(csv_path)
        if not data:
            print("Error preparing data")
            return False

        # Створення графіка
        _create_partner_age_success_chart(data, output_path)
        return True

    except Exception as e:
        print(f"Error creating chart: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Приклад використання
    csv_path = 'data_collector_simple.csv'
    output_path = 'partner_age_success_chart'
    create_partner_age_success_chart(csv_path, output_path)

