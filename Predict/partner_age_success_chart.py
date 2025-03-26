import csv
import base64
from io import StringIO, BytesIO
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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
    """Prepare data for partner age-success rate chart"""

    print("\nStarting _prepare_partner_age_success_data")

    # Read CSV data
    data = _read_csv_data(data_file)

    # Розраховуємо вік партнера для кожного замовлення
    partner_age_data = []
    print("\nProcessing partner age data...")

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

            partner_age = (row['date_order'] - row['partner_create_date']).days
            partner_age_data.append((partner_age, row['state'] == 'sale'))
            processed_rows += 1
        except Exception as e:
            error_rows += 1

    print(f"\nProcessing summary:")
    print(f"- Total rows: {len(data)}")
    print(f"- Missing create date: {missing_create_date}")
    print(f"- Missing order date: {missing_order_date}")
    print(f"- Successfully processed: {processed_rows}")
    print(f"- Errors: {error_rows}")

    if not partner_age_data:
        print("WARNING: No valid partner age data found")
        return None

    print(f"Found {len(partner_age_data)} valid orders for analysis")
    # Сортуємо за віком партнера
    partner_age_data.sort(key=lambda x: x[0])

    total_orders = len(partner_age_data)
    print(f"Total orders for analysis: {total_orders}")

    # Визначаємо кількість груп
    num_groups = min(30, total_orders // 50)
    if num_groups < 5:
        num_groups = 5
    print(f"Number of groups: {num_groups}")

    # Розраховуємо розмір кожної групи
    group_size = total_orders // num_groups
    remainder = total_orders % num_groups
    print(f"Group size: {group_size}, remainder: {remainder}")

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

        print(f"\nGroup {i}:")
        print(f"- Orders: {len(group_orders)}")
        print(f"- Success rate: {success_rate:.1f}%")
        print(f"- Age range: {min_age}-{max_age} days")

        # Форматуємо діапазон
        if max_age >= 365:
            range_str = f'{min_age / 365:.1f}y-{max_age / 365:.1f}y'
        elif max_age >= 30:
            range_str = f'{min_age / 30:.0f}m-{max_age / 30:.0f}m'
        else:
            range_str = f'{min_age}d-{max_age}d'

        # Додаємо дані до результату
        result['ranges'].append(range_str)
        result['rates'].append(success_rate)
        result['orders_count'].append(len(group_orders))

        start_idx = end_idx

    print("\nSuccessfully prepared partner age success data")
    return result


def _create_partner_age_success_chart(data, output_path):
    """Create chart showing success rate by partner age"""

    plt.figure(figsize=(15, 8))

    # Фільтруємо точки з нульовою кількістю ордерів
    x_points = []
    y_points = []
    counts = []
    for i, (rate, count) in enumerate(zip(data['rates'], data['orders_count'])):
        if count > 0:
            x_points.append(i)
            y_points.append(rate)
            counts.append(count)

    # Створюємо градієнт кольорів від червоного до зеленого в залежності від success rate
    colors = ['#ff4d4d' if rate < 50 else '#00cc00' for rate in y_points]
    sizes = [max(80, min(150, count / 2)) for count in counts]  # Розмір точки залежить від кількості замовлень

    # Малюємо точки
    scatter = plt.scatter(x_points, y_points, s=sizes, alpha=0.6, c=colors)

    # Розраховуємо середню кількість ордерів на точку
    avg_orders = sum(counts) // len(counts) if counts else 0

    plt.title(
        f'Success Rate by Partner Age\n(each point represents ~{avg_orders} orders, point size shows relative number in range)',
        pad=20, fontsize=12)
    plt.xlabel('Partner Age (d=days, m=months, y=years)', fontsize=10)
    plt.ylabel('Success Rate (%)', fontsize=10)

    # Налаштовуємо осі
    plt.ylim(-5, 105)

    # Показуємо всі мітки, якщо їх менше 10, інакше кожну другу
    if len(data['ranges']) <= 10:
        plt.xticks(range(len(data['ranges'])), data['ranges'],
                   rotation=45, ha='right')
    else:
        plt.xticks(range(len(data['ranges']))[::2],
                   [data['ranges'][i] for i in range(0, len(data['ranges']), 2)],
                   rotation=45, ha='right')

    plt.grid(True, linestyle='--', alpha=0.7)

    # Додаємо горизонтальні лінії
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=100, color='gray', linestyle='-', alpha=0.3)

    # Додаємо легенду
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#ff4d4d', markersize=10,
                   label='Success Rate < 50%'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#00cc00', markersize=10,
                   label='Success Rate ≥ 50%')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    # Зберігаємо графік
    plt.savefig(f"{output_path}.png", format='png',
                bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_path}.svg", format='svg',
                bbox_inches='tight')

    plt.close()
    print(f"Графік успішно збережено у файли:")
    print(f"- PNG: {output_path}.png")
    print(f"- SVG: {output_path}.svg")
    return True


def create_partner_age_success_chart(csv_path, output_path):
    """
    Створює графік аналізу історії клієнтів

    Args:
        csv_path (str): Шлях до CSV файлу з даними
        output_path (str): Базовий шлях для збереження графіка (без розширення)

    Returns:
        bool: True у разі успіху
    """
    partner_age_success_data = _prepare_partner_age_success_data(csv_path)
    _create_partner_age_success_chart(partner_age_success_data, output_path)


if __name__ == '__main__':
    # Приклад використання
    csv_path = 'data_collector_simple.csv'
    output_path = 'partner_age_success_chart'
    create_partner_age_success_chart(csv_path, output_path)
