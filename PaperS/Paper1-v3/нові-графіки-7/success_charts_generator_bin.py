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


COLOR_LOW = '#87CEEB'
COLOR_MED = '#4169E1'
COLOR_HIGH = '#000080'
COLOR_TREND = '#4682B4'


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


def _rate_color(rate):
    if rate < 50:
        return COLOR_LOW
    if rate <= 80:
        return COLOR_MED
    return COLOR_HIGH


def _split_into_quantile_groups(sorted_points, min_orders_per_group, max_groups, min_groups):
    total_points = len(sorted_points)
    if total_points <= 0:
        return []

    num_groups = min(max_groups, max(min_groups, total_points // min_orders_per_group))
    group_size = total_points // num_groups
    remainder = total_points % num_groups

    groups = []
    start_idx = 0
    for i in range(num_groups):
        current_group_size = group_size + (1 if i < remainder else 0)
        if current_group_size <= 0:
            break

        end_idx = start_idx + current_group_size
        group_points = sorted_points[start_idx:end_idx]
        if group_points:
            groups.append(group_points)

        start_idx = end_idx

    return groups


def _make_scatter_success_by_grouped_ranges(
    xlabel,
    ranges,
    rates,
    counts,
    output_path,
    show_trend_line=False,
    y_limit_top=105,
    legend_loc='upper right',
    x_tick_rotation=0,
    x_tick_ha='center',
):
    plt.figure(figsize=(15, 8))

    x_points = []
    y_points = []
    counts_nonzero = []

    for i, (rate, count) in enumerate(zip(rates, counts)):
        if count > 0:
            x_points.append(i)
            y_points.append(rate)
            counts_nonzero.append(count)

    colors = [_rate_color(r) for r in y_points]
    sizes = [max(80, min(150, c / 2)) for c in counts_nonzero]

    plt.scatter(x_points, y_points, s=sizes, alpha=0.6, c=colors)

    if show_trend_line and len(x_points) > 1:
        z = np.polyfit(x_points, y_points, 1)
        p = np.poly1d(z)
        plt.plot(
            x_points,
            p(x_points),
            color=COLOR_TREND,
            linestyle='--',
            alpha=0.8,
            linewidth=2,
            label='Trend line',
        )

    avg_orders = sum(counts_nonzero) // len(counts_nonzero) if counts_nonzero else 0

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)

    plt.ylim(-5, y_limit_top)

    if len(ranges) <= 10:
        plt.xticks(range(len(ranges)), ranges, rotation=x_tick_rotation, ha=x_tick_ha, fontsize=14)
    else:
        plt.xticks(
            range(len(ranges))[::2],
            [ranges[i] for i in range(0, len(ranges), 2)],
            rotation=x_tick_rotation,
            ha=x_tick_ha,
            fontsize=14,
        )

    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=80, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=100, color='gray', linestyle='-', alpha=0.3)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LOW, markersize=10, label='Success Rate < 50%'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MED, markersize=10, label='Success Rate 50-80%'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_HIGH, markersize=10, label='Success Rate > 80%'),
    ]

    if show_trend_line:
        legend_elements.append(plt.Line2D([0], [0], color=COLOR_TREND, linestyle='--', linewidth=2, label='Trend line'))

    plt.legend(handles=legend_elements, loc=legend_loc, fontsize=14)

    plt.savefig(f'{output_path}.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_path}.svg', format='svg', bbox_inches='tight')
    plt.close()

    print(f'Збережено: {output_path}.png та {output_path}.svg')


def build_order_messages_chart(orders, output_path):
    points = []
    for row in orders:
        try:
            messages_count = int(row.get('messages_count', ''))
        except Exception:
            continue
        if messages_count < 0:
            continue
        points.append((messages_count, _is_success(row)))

    if not points:
        print('Немає валідних даних для order_messages (messages_count)')
        return False

    points.sort(key=lambda x: x[0])
    groups = _split_into_quantile_groups(points, min_orders_per_group=15, max_groups=20, min_groups=5)

    ranges = []
    rates = []
    counts = []
    for g in groups:
        min_v = g[0][0]
        max_v = g[-1][0]
        success_cnt = sum(1 for _, ok in g if ok)
        rate = (success_cnt / len(g)) * 100
        range_str = f'{min_v}' if min_v == max_v else f'{min_v}-{max_v}'
        ranges.append(range_str)
        rates.append(rate)
        counts.append(len(g))

    _make_scatter_success_by_grouped_ranges(
        xlabel='Messages Count Range',
        ranges=ranges,
        rates=rates,
        counts=counts,
        output_path=output_path,
        show_trend_line=False,
        legend_loc='lower right',
    )
    return True


def build_order_changes_chart(orders, output_path):
    points = []
    for row in orders:
        try:
            changes_count = int(row.get('changes_count', ''))
        except Exception:
            continue
        if changes_count < 0:
            continue
        points.append((changes_count, _is_success(row)))

    if not points:
        print('Немає валідних даних для order_changes (changes_count)')
        return False

    points.sort(key=lambda x: x[0])
    groups = _split_into_quantile_groups(points, min_orders_per_group=1, max_groups=6, min_groups=6)

    ranges = []
    rates = []
    counts = []
    for g in groups:
        min_v = g[0][0]
        max_v = g[-1][0]
        success_cnt = sum(1 for _, ok in g if ok)
        rate = (success_cnt / len(g)) * 100
        range_str = f'{min_v}' if min_v == max_v else f'{min_v}-{max_v}'
        ranges.append(range_str)
        rates.append(rate)
        counts.append(len(g))

    _make_scatter_success_by_grouped_ranges(
        xlabel='Changes Count Range',
        ranges=ranges,
        rates=rates,
        counts=counts,
        output_path=output_path,
        show_trend_line=True,
        legend_loc='lower right',
    )
    return True


def build_order_amount_chart(orders, output_path):
    points = []
    for row in orders:
        try:
            amount = float(row.get('total_amount', ''))
        except Exception:
            continue
        if amount <= 0:
            continue
        points.append((amount, _is_success(row)))

    if not points:
        print('Немає валідних даних для order_amount (total_amount)')
        return False

    points.sort(key=lambda x: x[0])
    groups = _split_into_quantile_groups(points, min_orders_per_group=20, max_groups=30, min_groups=5)

    ranges = []
    rates = []
    counts = []
    for g in groups:
        min_v = g[0][0]
        max_v = g[-1][0]
        success_cnt = sum(1 for _, ok in g if ok)
        rate = (success_cnt / len(g)) * 100

        if max_v >= 1_000_000:
            range_str = f'{min_v / 1_000_000:.1f}M-{max_v / 1_000_000:.1f}M'
        elif max_v >= 1_000:
            range_str = f'{min_v / 1_000:.0f}K-{max_v / 1_000:.0f}K'
        else:
            range_str = f'{min_v:.0f}-{max_v:.0f}'

        ranges.append(range_str)
        rates.append(rate)
        counts.append(len(g))

    _make_scatter_success_by_grouped_ranges(
        xlabel='Order Amount Range',
        ranges=ranges,
        rates=rates,
        counts=counts,
        output_path=output_path,
        show_trend_line=False,
        legend_loc='upper right',
        x_tick_rotation=45,
        x_tick_ha='right',
    )
    return True


def build_discount_total_chart(orders, output_path):
    pos_points = []
    zero_orders = []

    for row in orders:
        try:
            discount = float(row.get('discount_total', ''))
        except Exception:
            continue

        if discount > 0:
            pos_points.append((discount, _is_success(row)))
        elif discount == 0:
            zero_orders.append((row.get('customer_id'), _is_success(row)))

    if not pos_points and not zero_orders:
        print('Немає валідних даних для discount_total')
        return False

    pos_points.sort(key=lambda x: x[0])

    # Дані для лівого графіка: no-discount (субгрупи) і знижки (одна точка)
    zero_total = len(zero_orders)
    zero_success = sum(1 for _, ok in zero_orders if ok)
    pos_total = len(pos_points)
    pos_success = sum(1 for _, ok in pos_points if ok)

    # Субгрупи no-discount: групуємо клієнтів за успішністю (0..100, крок 10) і підсумовуємо кількість замовлень
    zero_subgroups = []
    if zero_total > 0:
        customer_stats = {}
        for cust_id, ok in zero_orders:
            if cust_id not in customer_stats:
                customer_stats[cust_id] = {'total': 0, 'success': 0}
            customer_stats[cust_id]['total'] += 1
            if ok:
                customer_stats[cust_id]['success'] += 1

        bucket_to_orders = {k: 0 for k in range(0, 101, 10)}
        for stats in customer_stats.values():
            total = stats['total']
            rate = (stats['success'] / total) * 100 if total > 0 else 0.0
            bucket = int(round(rate / 10.0) * 10)
            bucket = max(0, min(100, bucket))
            bucket_to_orders[bucket] += total

        for bucket in sorted(bucket_to_orders.keys()):
            cnt = bucket_to_orders[bucket]
            if cnt > 0:
                zero_subgroups.append({'rate': float(bucket), 'count': cnt})

    # Дані для правого графіка: бінінг позитивних знижок
    pos_ranges = []
    pos_rates = []
    pos_counts = []

    if pos_points:
        groups = _split_into_quantile_groups(pos_points, min_orders_per_group=20, max_groups=30, min_groups=5)
        for g in groups:
            min_v = g[0][0]
            max_v = g[-1][0]
            success_cnt = sum(1 for _, ok in g if ok)
            rate = (success_cnt / len(g)) * 100

            if max_v >= 1_000_000:
                range_str = f'{min_v / 1_000_000:.1f}M-{max_v / 1_000_000:.1f}M'
            elif max_v >= 1_000:
                range_str = f'{min_v / 1_000:.0f}K-{max_v / 1_000:.0f}K'
            else:
                range_str = f'{min_v:.0f}-{max_v:.0f}'

            pos_ranges.append(range_str)
            pos_rates.append(rate)
            pos_counts.append(len(g))

    # Малюємо 2 підграфіки як у твоєму discount_success_chart.py
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 8))
    discount_fontsize = 16

    # LEFT: no discount vs with discount
    labels = ['No discount', 'With discount']
    x = np.arange(2)

    if zero_subgroups:
        n = len(zero_subgroups)
        offsets = [0.0] if n == 1 else list(np.linspace(-0.8, 0.8, n))
        for off, sg in zip(offsets, zero_subgroups):
            r = sg['rate']
            c = sg['count']
            color = _rate_color(r)
            size = max(120, min(260, c))
            ax_left.scatter(x[0] + off, r, s=size, c=color, alpha=0.7)
            ax_left.text(x[0] + off, r + 3, f'{r:.1f}%\n(n={c})', ha='center', va='bottom', fontsize=discount_fontsize)
    else:
        r0 = (zero_success / zero_total * 100) if zero_total > 0 else 0.0
        c0 = zero_total
        ax_left.scatter(x[0], r0, s=max(120, min(260, c0)), c=_rate_color(r0), alpha=0.7)
        ax_left.text(x[0], r0 + 3, f'{r0:.1f}%\n(n={c0})', ha='center', va='bottom', fontsize=discount_fontsize)

    r1 = (pos_success / pos_total * 100) if pos_total > 0 else 0.0
    c1 = pos_total
    ax_left.scatter(x[1], r1, s=max(120, min(260, c1)), c=_rate_color(r1), alpha=0.7)
    ax_left.text(x[1], r1 + 3, f'{r1:.1f}%\n(n={c1})', ha='center', va='bottom', fontsize=discount_fontsize)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, fontsize=discount_fontsize)
    ax_left.tick_params(axis='y', labelsize=discount_fontsize)
    ax_left.set_ylim(-5, 112)
    ax_left.set_ylabel('Success Rate (%)', fontsize=discount_fontsize)
    ax_left.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax_left.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_left.set_xlim(-1.1, 1.4)

    # RIGHT: grouped positive discounts
    x_points = []
    y_points = []
    sizes = []
    colors = []
    for i, (rate, count) in enumerate(zip(pos_rates, pos_counts)):
        if count <= 0:
            continue
        x_points.append(i)
        y_points.append(rate)
        sizes.append(max(80, min(150, count / 2)))
        colors.append(_rate_color(rate))

    ax_right.scatter(x_points, y_points, s=sizes, alpha=0.6, c=colors)
    avg_orders = (sum(pos_counts) // len(pos_counts)) if pos_counts else 0
    ax_right.set_xlabel('Discount Amount Range', fontsize=discount_fontsize)
    ax_right.set_ylabel('Success Rate (%)', fontsize=discount_fontsize)
    ax_right.set_ylim(-5, 112)

    if len(pos_ranges) <= 10:
        ax_right.set_xticks(range(len(pos_ranges)))
        ax_right.set_xticklabels(pos_ranges, rotation=0, ha='center', fontsize=discount_fontsize)
    else:
        ax_right.set_xticks(range(len(pos_ranges))[::2])
        ax_right.set_xticklabels(
            [pos_ranges[i] for i in range(0, len(pos_ranges), 2)],
            rotation=0,
            ha='center',
            fontsize=discount_fontsize,
        )

    ax_right.tick_params(axis='y', labelsize=discount_fontsize)
    ax_right.grid(True, linestyle='--', alpha=0.7)
    ax_right.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax_right.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax_right.axhline(y=80, color='gray', linestyle='--', alpha=0.3)
    ax_right.axhline(y=100, color='gray', linestyle='-', alpha=0.3)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LOW, markersize=10, label='Success Rate < 50%'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MED, markersize=10, label='Success Rate 50-80%'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_HIGH, markersize=10, label='Success Rate > 80%'),
    ]
    ax_right.legend(handles=legend_elements, loc='lower right', fontsize=discount_fontsize)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    left_bbox = ax_left.get_position()
    right_bbox = ax_right.get_position()
    fig.text(
        (left_bbox.x0 + left_bbox.x1) / 2,
        left_bbox.y0 - 0.08,
        '(a)',
        ha='center',
        va='top',
        fontsize=discount_fontsize,
    )
    fig.text(
        (right_bbox.x0 + right_bbox.x1) / 2,
        right_bbox.y0 - 0.08,
        '(b)',
        ha='center',
        va='top',
        fontsize=discount_fontsize,
    )

    fig.savefig(f'{output_path}.png', format='png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{output_path}.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    print(f'Збережено: {output_path}.png та {output_path}.svg')
    return True


def build_day_of_week_chart(orders, output_path):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    totals = {d: 0 for d in weekdays}
    successes = {d: 0 for d in weekdays}

    for row in orders:
        day = row.get('day_of_week')
        if day not in totals:
            continue
        totals[day] += 1
        if _is_success(row):
            successes[day] += 1

    ranges = []
    rates = []
    counts = []
    for d in weekdays:
        total = totals[d]
        rate = (successes[d] / total * 100) if total > 0 else 0.0
        ranges.append(d)
        rates.append(rate)
        counts.append(total)

    plt.figure(figsize=(15, 8))

    x_points = []
    y_points = []
    counts_nonzero = []
    for i, (rate, count) in enumerate(zip(rates, counts)):
        if count > 0:
            x_points.append(i)
            y_points.append(rate)
            counts_nonzero.append(count)

    colors = [_rate_color(r) for r in y_points]
    sizes = [max(100, min(260, c)) for c in counts_nonzero]

    plt.scatter(x_points, y_points, s=sizes, alpha=0.6, c=colors)

    avg_orders = sum(counts_nonzero) // len(counts_nonzero) if counts_nonzero else 0

    plt.xlabel('Day of Week', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)

    plt.ylim(-5, 105)
    plt.xticks(range(len(ranges)), ranges, rotation=0, ha='center', fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=80, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=100, color='gray', linestyle='-', alpha=0.3)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LOW, markersize=10, label='Success Rate < 50%'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MED, markersize=10, label='Success Rate 50-80%'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_HIGH, markersize=10, label='Success Rate > 80%'),
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=14)

    plt.savefig(f'{output_path}.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_path}.svg', format='svg', bbox_inches='tight')
    plt.close()

    print(f'Збережено: {output_path}.png та {output_path}.svg')
    return True


def build_all(csv_path='data_collector_extended.csv'):
    orders = _read_orders(csv_path)

    build_order_messages_chart(orders, output_path='messages_success_chart')
    build_order_changes_chart(orders, output_path='changes_success_chart')
    build_order_amount_chart(orders, output_path='amount_success_chart')
    build_discount_total_chart(orders, output_path='discount_success_chart')
    build_day_of_week_chart(orders, output_path='dayofweek_success_chart')


if __name__ == '__main__':
    build_all()
