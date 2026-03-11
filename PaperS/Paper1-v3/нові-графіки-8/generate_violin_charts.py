import csv
import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from mplfonts import use_font

# Налаштування стилів у стилі попередніх скриптів
try:
    use_font('Times New Roman')
except Exception as e:
    print(f"Не вдалося застосувати шрифт Times New Roman через mplfonts: {e}")

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.size'] = 14

COLOR_FAIL = '#FFEDA0'  # Блідо-жовтий (ненасичений, щоб було видно box plot)
COLOR_SUCCESS = '#9ECAE1'  # Блідо-блакитний (ненасичений)


def _parse_datetime(value):
    if not value:
        return None
    try:
        date_str = value.split('.')[0]
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except Exception:
        return None


def _read_orders(csv_path):
    print(f'Starting CSV read: {csv_path}')
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


def build_violin_chart(orders, feature_key, ylabel, output_name, is_datetime=False, ylim_top=None, log_scale=False,
                       bw_adjust=1.0):
    x_labels = []
    y_values = []

    start_dt = datetime(2017, 7, 1, 0, 0, 0)
    end_dt = datetime(2025, 1, 31, 23, 59, 59)

    for row in orders:
        ok = _is_success(row)
        cat = 'Successful\n(state = sale)' if ok else 'Unsuccessful\n(state != sale)'

        if is_datetime:
            val = row.get(feature_key)
            if isinstance(val, datetime):
                if val >= start_dt and val <= end_dt:
                    y_values.append(val.timestamp())
                    x_labels.append(cat)
        else:
            val = _to_float(row.get(feature_key))
            if val is not None and val >= 0:
                y_values.append(val)
                x_labels.append(cat)

    if not y_values:
        print(f"No valid data for {feature_key}")
        return

    df = pd.DataFrame({'Success Status': x_labels, feature_key: y_values})

    order = ['Unsuccessful\n(state != sale)', 'Successful\n(state = sale)']

    fig, ax = plt.subplots(figsize=(10, 8))

    try:
        sns.violinplot(
            data=df,
            x='Success Status',
            y=feature_key,
            order=order,
            palette={'Unsuccessful\n(state != sale)': COLOR_FAIL, 'Successful\n(state = sale)': COLOR_SUCCESS},
            inner="box",
            linewidth=1.2,
            cut=0,
            bw_adjust=bw_adjust,
            ax=ax
        )
    except TypeError:
        # Для старих версій seaborn
        sns.violinplot(
            data=df,
            x='Success Status',
            y=feature_key,
            order=order,
            palette={'Unsuccessful\n(state != sale)': COLOR_FAIL, 'Successful\n(state = sale)': COLOR_SUCCESS},
            inner="box",
            linewidth=1.2,
            cut=0,
            bw=bw_adjust,
            ax=ax
        )

    ax.set_xlabel('')

    if log_scale:
        min_y = min(y_values)
        if min_y <= 0:
            ax.set_yscale('symlog', linthresh=1.0)
            ax.set_ylim(bottom=-0.1)
        else:
            ax.set_yscale('log')
            ax.set_ylim(bottom=min_y * 0.8)
        ylabel += ' (Log Scale)'

    ax.set_ylabel(ylabel, fontsize=16, labelpad=10)

    if ylim_top is not None and not log_scale:
        ax.set_ylim(-ylim_top * 0.05, ylim_top)

    if is_datetime:
        # Жорстко фіксуємо тіки на 1 січня кожного року від 2017 до 2025
        years = range(2017, 2026)
        ticks = [datetime(y, 1, 1).timestamp() for y in years]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(y) for y in years])
        # Ставимо жорсткі межі Y, щоб графік не розтягувало на випадкові місяці
        ax.set_ylim(datetime(2017, 1, 1).timestamp(), datetime(2025, 12, 31).timestamp())
    elif feature_key == 'hour_of_day':
        ax.set_yticks(range(0, 25, 2))
        ax.set_ylim(-1, 24)

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_name}.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_name}.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_name}.png / .svg")


def build_all(csv_path):
    orders = _read_orders(csv_path)
    base_dir = os.path.dirname(csv_path)

    build_violin_chart(
        orders,
        feature_key='customer_relationship_days',
        ylabel='Customer Relationship (Days)',
        output_name=os.path.join(base_dir, 'customer_relationship_days_violin')
    )

    build_violin_chart(
        orders,
        feature_key='date_order',
        ylabel='Order Date',
        output_name=os.path.join(base_dir, 'date_order_violin'),
        is_datetime=True
    )

    build_violin_chart(
        orders,
        feature_key='previous_orders_count',
        ylabel='Previous Orders Count',
        output_name=os.path.join(base_dir, 'previous_orders_count_violin'),
        ylim_top=100
    )

    build_violin_chart(
        orders,
        feature_key='previous_orders_count',
        ylabel='Previous Orders Count',
        output_name=os.path.join(base_dir, 'previous_orders_count_violin_log'),
        log_scale=True
    )

    build_violin_chart(
        orders,
        feature_key='hour_of_day',
        ylabel='Hour of Day',
        output_name=os.path.join(base_dir, 'hour_of_day_violin'),
        bw_adjust=2.5 # Оптимальне згладжування для годин
    )

    build_violin_chart(
        orders,
        feature_key='messages_count',
        ylabel='Messages Count',
        output_name=os.path.join(base_dir, 'messages_count_violin'),
        ylim_top=50
    )

    build_violin_chart(
        orders,
        feature_key='messages_count',
        ylabel='Messages Count',
        output_name=os.path.join(base_dir, 'messages_count_violin_log'),
        log_scale=True
    )


if __name__ == '__main__':
    target_csv = r"data_collector_extended.csv"
    build_all(target_csv)
