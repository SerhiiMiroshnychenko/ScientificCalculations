import csv
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mplfonts import use_font


use_font('Times New Roman')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False


COLOR_LOW = '#87CEEB'
COLOR_MED = '#4169E1'
COLOR_HIGH = '#000080'
COLOR_TREND = '#4682B4'

COLOR_FAIL = '#87CEEB'
COLOR_SUCCESS = '#000080'


def _parse_datetime(value):
    if not value:
        return None
    try:
        date_str = value.split('.')[0]
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except Exception:
        return None


def _read_orders(csv_path):
    print('Початок читання CSV')

    orders = []
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if 'date_order' in row:
                row['date_order'] = _parse_datetime(row.get('date_order'))
            if 'create_date' in row:
                row['create_date'] = _parse_datetime(row.get('create_date'))
            orders.append(row)

    print(f'Зчитано рядків: {len(orders)}')
    return orders


def _is_success(row):
    return row.get('state') == 'sale'


def _to_float(value):
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _to_int(value):
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _scatter_raw_success(
    x_values,
    y_success,
    title,
    xlabel,
    output_path,
    x_tick_labels=None,
    x_tick_positions=None,
):
    if not x_values or not y_success:
        print(f'Немає даних для графіка: {output_path}')
        return False

    if len(x_values) != len(y_success):
        print(f'Неконсистентні дані для графіка: {output_path}')
        return False

    rng = np.random.default_rng(42)
    y = np.array([1.0 if ok else 0.0 for ok in y_success], dtype=float)
    y = y + rng.uniform(-0.06, 0.06, size=len(y))

    colors = [COLOR_SUCCESS if ok else COLOR_FAIL for ok in y_success]

    plt.figure(figsize=(15, 8))
    plt.scatter(x_values, y, s=14, alpha=0.25, c=colors, edgecolors='none')

    plt.title(f'{title}\n(кожна точка = 1 замовлення)', pad=20, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Успішність (0/1)', fontsize=14)

    plt.yticks([0, 1], ['0', '1'], fontsize=14)
    plt.ylim(-0.2, 1.2)
    plt.grid(True, linestyle='--', alpha=0.35)

    if x_tick_positions is not None and x_tick_labels is not None:
        plt.xticks(x_tick_positions, x_tick_labels, rotation=0, ha='center', fontsize=14)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_FAIL, markersize=8, label='Неуспішне (state != sale)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_SUCCESS, markersize=8, label='Успішне (state = sale)'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.savefig(f'{output_path}.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_path}.svg', format='svg', bbox_inches='tight')
    plt.close()

    print(f'Збережено: {output_path}.png та {output_path}.svg')
    return True


def build_order_messages_chart(orders, output_path):
    x = []
    y = []
    for row in orders:
        messages_count = _to_int(row.get('messages_count'))
        if messages_count is None or messages_count < 0:
            continue
        x.append(messages_count)
        y.append(_is_success(row))

    return _scatter_raw_success(
        x_values=x,
        y_success=y,
        title='Успішність замовлення залежно від кількості повідомлень (order_messages)',
        xlabel='messages_count',
        output_path=output_path,
    )


def build_order_changes_chart(orders, output_path):
    x = []
    y = []
    for row in orders:
        changes_count = _to_int(row.get('changes_count'))
        if changes_count is None or changes_count < 0:
            continue
        x.append(changes_count)
        y.append(_is_success(row))

    return _scatter_raw_success(
        x_values=x,
        y_success=y,
        title='Успішність замовлення залежно від кількості змін (order_changes)',
        xlabel='changes_count',
        output_path=output_path,
    )


def build_order_amount_chart(orders, output_path):
    x = []
    y = []
    for row in orders:
        amount = _to_float(row.get('total_amount'))
        if amount is None or amount <= 0:
            continue
        x.append(amount)
        y.append(_is_success(row))

    return _scatter_raw_success(
        x_values=x,
        y_success=y,
        title='Успішність замовлення залежно від суми (order_amount)',
        xlabel='total_amount',
        output_path=output_path,
    )


def build_discount_total_chart(orders, output_path):
    x = []
    y = []
    for row in orders:
        discount = _to_float(row.get('discount_total'))
        if discount is None:
            continue
        x.append(discount)
        y.append(_is_success(row))

    return _scatter_raw_success(
        x_values=x,
        y_success=y,
        title='Успішність замовлення залежно від знижки (discount_total)',
        xlabel='discount_total',
        output_path=output_path,
    )


def build_day_of_week_chart(orders, output_path):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_to_x = {day: i for i, day in enumerate(weekdays)}

    x = []
    y = []
    rng = np.random.default_rng(42)

    for row in orders:
        day = row.get('day_of_week')
        if day not in day_to_x:
            continue
        x.append(day_to_x[day] + float(rng.uniform(-0.12, 0.12)))
        y.append(_is_success(row))

    return _scatter_raw_success(
        x_values=x,
        y_success=y,
        title='Успішність замовлення залежно від дня тижня (day_of_week)',
        xlabel='day_of_week',
        output_path=output_path,
        x_tick_positions=list(range(len(weekdays))),
        x_tick_labels=weekdays,
    )


def build_all(csv_path='data_collector_extended.csv'):
    orders = _read_orders(csv_path)

    build_order_messages_chart(orders, output_path='messages_success_chart')
    build_order_changes_chart(orders, output_path='changes_success_chart')
    build_order_amount_chart(orders, output_path='amount_success_chart')
    build_discount_total_chart(orders, output_path='discount_success_chart')
    build_day_of_week_chart(orders, output_path='dayofweek_success_chart')


if __name__ == '__main__':
    build_all()
