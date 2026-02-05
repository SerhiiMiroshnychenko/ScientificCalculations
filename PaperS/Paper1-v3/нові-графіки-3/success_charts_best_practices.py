import csv
import math
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mplfonts import use_font


use_font('Times New Roman')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False


COLOR_LINE = '#003E6B'
COLOR_POINTS = '#00629B'
COLOR_BAND = '#00629B'
COLOR_SCATTER_SUCCESS = '#000080'
COLOR_SCATTER_FAIL = '#87CEEB'


def _parse_datetime(value):
    if not value:
        return None
    try:
        date_str = value.split('.')[0]
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except Exception:
        return None


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


def _wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0, 0.0

    p_hat = successes / n
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z ** 2) / (4.0 * (n ** 2)))

    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return p_hat, low, high


def _quantile_bins(points, max_bins=20, min_bin_size=200):
    """Розбиття на біни за квантилями без фіксованих меж.

    points: list[(x, success_bool)] уже відсортований за x.

    Повертає список бінів, де кожен бін — список точок.
    """
    total = len(points)
    if total == 0:
        return []

    bins = max(5, min(max_bins, total // min_bin_size))
    if bins <= 0:
        bins = 1

    base = total // bins
    rem = total % bins

    out = []
    start = 0
    for i in range(bins):
        size = base + (1 if i < rem else 0)
        if size <= 0:
            break
        end = start + size
        chunk = points[start:end]
        if chunk:
            out.append(chunk)
        start = end

    return out


def _plot_probability_by_x(
    points,
    title,
    xlabel,
    output_path,
    show_raw_scatter=True,
    raw_scatter_max_points=30000,
    xscale=None,
):
    """Візуалізація P(success|x) по квантильних бінах + Wilson CI.

    points: list[(x, success_bool)]
    """
    if not points:
        print(f'Немає даних для графіка: {output_path}')
        return False

    points.sort(key=lambda t: t[0])

    bins = _quantile_bins(points, max_bins=20, min_bin_size=200)

    x_centers = []
    p_hats = []
    ci_low = []
    ci_high = []
    n_per_bin = []

    for b in bins:
        xs = [t[0] for t in b]
        n = len(b)
        s = sum(1 for _, ok in b if ok)
        p_hat, low, high = _wilson_ci(s, n)

        x_centers.append(float(np.median(xs)))
        p_hats.append(p_hat)
        ci_low.append(low)
        ci_high.append(high)
        n_per_bin.append(n)

    plt.figure(figsize=(15, 8))

    if show_raw_scatter:
        rng = np.random.default_rng(42)
        if len(points) > raw_scatter_max_points:
            idx = rng.choice(len(points), size=raw_scatter_max_points, replace=False)
            sample = [points[i] for i in idx]
        else:
            sample = points

        xs = [t[0] for t in sample]
        ys = [1.0 if t[1] else 0.0 for t in sample]
        ys = np.array(ys, dtype=float) + rng.uniform(-0.05, 0.05, size=len(sample))
        colors = [COLOR_SCATTER_SUCCESS if ok else COLOR_SCATTER_FAIL for _, ok in sample]

        plt.scatter(xs, ys, s=6, alpha=0.08, c=colors, edgecolors='none')

    # CI band
    plt.fill_between(x_centers, ci_low, ci_high, color=COLOR_BAND, alpha=0.15, linewidth=0)

    # Line + points
    plt.plot(x_centers, p_hats, color=COLOR_LINE, linewidth=2)
    plt.scatter(x_centers, p_hats, color=COLOR_POINTS, s=40, zorder=3)

    # Annotate n per bin (optional, light)
    for x0, p0, n in zip(x_centers, p_hats, n_per_bin):
        plt.text(x0, min(1.04, p0 + 0.03), f'n={n}', fontsize=9, ha='center', va='bottom', alpha=0.7)

    plt.title(title, pad=16, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('P(успіх)', fontsize=14)

    plt.ylim(-0.05, 1.05)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.35)

    if xscale:
        plt.xscale(xscale)

    legend_elements = [
        plt.Line2D([0], [0], color=COLOR_LINE, linewidth=2, label='Оцінка P(успіх|x) (квантильні біни)'),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BAND, alpha=0.15, label='Wilson 95% CI'),
    ]
    if show_raw_scatter:
        legend_elements.extend(
            [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_SCATTER_SUCCESS, markersize=6, label='Успіх (state=sale)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_SCATTER_FAIL, markersize=6, label='Неуспіх'),
            ]
        )

    plt.legend(handles=legend_elements, loc='best')

    plt.savefig(f'{output_path}.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_path}.svg', format='svg', bbox_inches='tight')
    plt.close()

    print(f'Збережено: {output_path}.png та {output_path}.svg')
    return True


def build_messages_probability_chart(orders, output_path='bp_messages_probability'):
    points = []
    for row in orders:
        x = _to_int(row.get('messages_count'))
        if x is None or x < 0:
            continue
        points.append((x, _is_success(row)))

    return _plot_probability_by_x(
        points=points,
        title='P(успіх) залежно від кількості повідомлень (order_messages)',
        xlabel='messages_count',
        output_path=output_path,
        show_raw_scatter=True,
        xscale=None,
    )


def build_changes_probability_chart(orders, output_path='bp_changes_probability'):
    points = []
    for row in orders:
        x = _to_int(row.get('changes_count'))
        if x is None or x < 0:
            continue
        points.append((x, _is_success(row)))

    return _plot_probability_by_x(
        points=points,
        title='P(успіх) залежно від кількості змін (order_changes)',
        xlabel='changes_count',
        output_path=output_path,
        show_raw_scatter=True,
        xscale=None,
    )


def build_amount_probability_chart(orders, output_path='bp_amount_probability'):
    points = []
    for row in orders:
        x = _to_float(row.get('total_amount'))
        if x is None or x <= 0:
            continue
        points.append((x, _is_success(row)))

    return _plot_probability_by_x(
        points=points,
        title='P(успіх) залежно від суми замовлення (order_amount)',
        xlabel='total_amount',
        output_path=output_path,
        show_raw_scatter=True,
        xscale='log',
    )


def build_discount_probability_chart(orders, output_path='bp_discount_probability'):
    points = []
    for row in orders:
        x = _to_float(row.get('discount_total'))
        if x is None:
            continue
        # Дозволяємо 0 і позитивні. Якщо є від'ємні — пропускаємо.
        if x < 0:
            continue
        points.append((x, _is_success(row)))

    return _plot_probability_by_x(
        points=points,
        title='P(успіх) залежно від знижки (discount_total)',
        xlabel='discount_total',
        output_path=output_path,
        show_raw_scatter=True,
        xscale=None,
    )


def build_day_of_week_lollipop(orders, output_path='bp_dayofweek_lollipop'):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    totals = {d: 0 for d in weekdays}
    successes = {d: 0 for d in weekdays}

    for row in orders:
        d = row.get('day_of_week')
        if d not in totals:
            continue
        totals[d] += 1
        if _is_success(row):
            successes[d] += 1

    rates = []
    lows = []
    highs = []
    ns = []

    for d in weekdays:
        n = totals[d]
        s = successes[d]
        p_hat, low, high = _wilson_ci(s, n)
        rates.append(p_hat)
        lows.append(low)
        highs.append(high)
        ns.append(n)

    y_pos = np.arange(len(weekdays))

    plt.figure(figsize=(15, 8))

    # Lollipop lines
    for y0, p in zip(y_pos, rates):
        plt.plot([0, p], [y0, y0], color=COLOR_LINE, alpha=0.4, linewidth=2)

    # Points
    plt.scatter(rates, y_pos, color=COLOR_POINTS, s=80, zorder=3)

    # CI as horizontal error bars
    xerr = [
        [p - low for p, low in zip(rates, lows)],
        [high - p for p, high in zip(rates, highs)],
    ]
    plt.errorbar(rates, y_pos, xerr=xerr, fmt='none', ecolor=COLOR_BAND, elinewidth=2, alpha=0.7, capsize=4)

    for p, y0, n in zip(rates, y_pos, ns):
        plt.text(min(1.02, p + 0.02), y0, f'n={n}', va='center', fontsize=10, alpha=0.8)

    plt.yticks(y_pos, weekdays, fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlim(-0.02, 1.05)

    plt.grid(True, axis='x', linestyle='--', alpha=0.35)
    plt.xlabel('P(успіх)', fontsize=14)
    plt.title('P(успіх) за днями тижня (day_of_week) + Wilson 95% CI', pad=16, fontsize=14)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_POINTS, markersize=8, label='Оцінка частки успіху'),
        plt.Line2D([0], [0], color=COLOR_BAND, linewidth=2, label='Wilson 95% CI'),
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.savefig(f'{output_path}.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_path}.svg', format='svg', bbox_inches='tight')
    plt.close()

    print(f'Збережено: {output_path}.png та {output_path}.svg')
    return True


def build_all(csv_path='data_collector_extended.csv'):
    orders = _read_orders(csv_path)

    build_messages_probability_chart(orders)
    build_changes_probability_chart(orders)
    build_amount_probability_chart(orders)
    build_discount_probability_chart(orders)
    build_day_of_week_lollipop(orders)


if __name__ == '__main__':
    build_all()
