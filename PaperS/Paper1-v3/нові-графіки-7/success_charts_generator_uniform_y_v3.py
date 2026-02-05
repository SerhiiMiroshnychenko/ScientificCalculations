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
mpl.rcParams['font.size'] = 14


COLOR_FAIL = '#FFD700'
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
    print('Starting CSV read')

    orders = []
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if 'date_order' in row:
                row['date_order'] = _parse_datetime(row.get('date_order'))
            if 'create_date' in row:
                row['create_date'] = _parse_datetime(row.get('create_date'))
            orders.append(row)

    print(f'Rows read: {len(orders)}')
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


def _uniform_success_band_y(y_success, seed=42, band_padding=0.02):
    rng = np.random.default_rng(seed)

    low_min = 0.0 + band_padding
    low_max = 0.5 - band_padding
    high_min = 0.5 + band_padding
    high_max = 1.0 - band_padding

    y = np.empty(len(y_success), dtype=float)
    for i, ok in enumerate(y_success):
        if ok:
            y[i] = float(rng.uniform(high_min, high_max))
        else:
            y[i] = float(rng.uniform(low_min, low_max))
    return y


def _scatter_raw_success(
    x_values,
    y_success,
    xlabel,
    output_path,
    x_tick_labels=None,
    x_tick_positions=None,
):
    if not x_values or not y_success:
        print(f'No data for chart: {output_path}')
        return False

    if len(x_values) != len(y_success):
        print(f'Inconsistent data for chart: {output_path}')
        return False

    y = _uniform_success_band_y(y_success=y_success, seed=42, band_padding=0.02)
    colors = [COLOR_SUCCESS if ok else COLOR_FAIL for ok in y_success]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(x_values, y, s=14, alpha=0.25, c=colors, edgecolors='none')

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Success (0/1)', fontsize=14)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(['0', '1'], fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.35)

    if x_tick_positions is not None and x_tick_labels is not None:
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, rotation=0, ha='center', fontsize=14)

    if any(isinstance(v, datetime) for v in x_values):
        fig.autofmt_xdate()

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=COLOR_FAIL,
            markersize=8,
            label='Unsuccessful (state != sale)',
        ),
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=COLOR_SUCCESS,
            markersize=8,
            label='Successful (state = sale)',
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=14,
    )

    fig.subplots_adjust(bottom=0.22)
    fig.savefig(f'{output_path}.png', format='png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{output_path}.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {output_path}.png and {output_path}.svg')
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
        xlabel='messages_count',
        output_path=output_path,
    )


def build_hour_of_day_chart(orders, output_path):
    x = []
    y = []
    for row in orders:
        hour_of_day = _to_int(row.get('hour_of_day'))
        if hour_of_day is None or hour_of_day < 0 or hour_of_day > 23:
            continue
        x.append(hour_of_day)
        y.append(_is_success(row))

    return _scatter_raw_success(
        x_values=x,
        y_success=y,
        xlabel='hour_of_day',
        output_path=output_path,
        x_tick_positions=list(range(0, 24)),
        x_tick_labels=[str(h) for h in range(0, 24)],
    )


def build_customer_relationship_days_chart(orders, output_path):
    x = []
    y = []
    for row in orders:
        days = _to_int(row.get('customer_relationship_days'))
        if days is None or days < 0:
            continue
        x.append(days)
        y.append(_is_success(row))

    return _scatter_raw_success(
        x_values=x,
        y_success=y,
        xlabel='customer_relationship_days',
        output_path=output_path,
    )


def build_previous_orders_count_chart(orders, output_path):
    x = []
    y = []
    for row in orders:
        previous_orders_count = _to_int(row.get('previous_orders_count'))
        if previous_orders_count is None or previous_orders_count < 0:
            continue
        x.append(previous_orders_count)
        y.append(_is_success(row))

    return _scatter_raw_success(
        x_values=x,
        y_success=y,
        xlabel='previous_orders_count',
        output_path=output_path,
    )


def build_date_order_chart(orders, output_path):
    start_dt = datetime(2017, 7, 1, 0, 0, 0)
    end_dt = datetime(2025, 1, 31, 23, 59, 59)

    x = []
    y = []
    for row in orders:
        date_order = row.get('date_order')
        if not isinstance(date_order, datetime):
            continue
        if date_order < start_dt or date_order > end_dt:
            continue
        x.append(date_order)
        y.append(_is_success(row))

    return _scatter_raw_success(
        x_values=x,
        y_success=y,
        xlabel='date_order',
        output_path=output_path,
    )


def build_all(csv_path='data_collector_extended.csv'):
    orders = _read_orders(csv_path)

    build_customer_relationship_days_chart(
        orders,
        output_path='customer_relationship_days_success_chart',
    )
    build_date_order_chart(orders, output_path='date_order_success_chart')
    build_previous_orders_count_chart(
        orders,
        output_path='previous_orders_count_success_chart',
    )
    build_hour_of_day_chart(orders, output_path='hour_of_day_success_chart')
    build_order_messages_chart(orders, output_path='messages_success_chart')


if __name__ == '__main__':
    build_all()
