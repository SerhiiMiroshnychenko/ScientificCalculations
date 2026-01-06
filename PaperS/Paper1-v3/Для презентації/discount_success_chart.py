import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mplfonts import use_font
import csv
from datetime import datetime
from io import StringIO

# Configure Arial font for English labels
use_font('Arial')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False  # Disable LaTeX

def _read_csv_data(csv_path):
    """Read CSV data from file"""
    print("Starting _read_csv_data")

    try:
        csv_file = open(csv_path, 'r', encoding='utf-8')
        reader = csv.DictReader(csv_file)
        data = []
        for row in reader:
            try:
                # Strip microseconds from dates
                if 'date_order' in row:
                    date_str = row['date_order'].split('.')[0]
                    row['date_order'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue

            try:
                if 'create_date' in row:
                    date_str = row['create_date'].split('.')[0]
                    row['create_date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue

            # Ensure required fields are present
            if not all(field in row for field in
                       ['order_id', 'customer_id', 'date_order', 'state', 'discount_total']):
                print(f"Missing required fields in row: {row}")
                continue

            data.append(row)

        csv_file.close()
        print(f"Successfully read {len(data)} rows from CSV")
        return data

    except Exception as e:
        print(f"Error reading CSV data: {str(e)}")
        return []

def _prepare_discount_success_data(csv_path):
    """Prepare data for discount success chart.

    Returns a structure with two datasets:
    - binary: No discount vs With discount
    - grouped: only positive discounts split into ranges
    """
    print("Starting _prepare_discount_success_data")

    try:
        data = _read_csv_data(csv_path)
        if not data:
            print("No data available")
            return None

        # Collect orders
        pos_points = []  # list of (discount, is_success) for positive discounts
        zero_orders = []  # list of (customer_id, is_success) for zero-discount orders
        for row in data:
            try:
                discount = float(row['discount_total'])
                if discount > 0:  # positive discounts grouped by ranges
                    is_successful = row['state'] == 'sale'
                    pos_points.append((discount, is_successful))
                elif discount == 0:  # collect zero-discount orders per customer
                    is_successful = row['state'] == 'sale'
                    zero_orders.append((row.get('customer_id'), is_successful))
            except (ValueError, TypeError):
                continue

        if not pos_points and len(zero_orders) == 0:
            print("No valid data points found")
            return None

        # Sort by discount amount for positive discounts
        pos_points.sort(key=lambda x: x[0])

        total_points = len(pos_points)
        print(f"Total orders with positive discounts: {total_points}")

        # Determine number of groups similar to amount chart
        num_groups = min(30, total_points // 20)  # At least ~20 orders per group
        if num_groups < 5:
            num_groups = 5

        # Compute group sizes for positive discounts
        group_size = total_points // num_groups if num_groups > 0 else 0
        remainder = total_points % num_groups if num_groups > 0 else 0

        result = {
            'binary': {
                'labels': [],
                'rates': [],
                'counts': [],
            },
            'grouped': {
                'ranges': [],
                'rates': [],
                'orders_count': [],
            }
        }

        # 1) Binary dataset
        zero_total = len(zero_orders)
        zero_success = sum(1 for _, f in zero_orders if f)
        pos_total = len(pos_points)
        pos_success = sum(1 for _, f in pos_points if f)

        if zero_total + pos_total > 0:
            # Add binary labels and metrics
            result['binary']['labels'] = ['No discount', 'With discount']
            result['binary']['counts'] = [zero_total, pos_total]
            result['binary']['rates'] = [
                (zero_success / zero_total * 100) if zero_total > 0 else 0.0,
                (pos_success / pos_total * 100) if pos_total > 0 else 0.0,
            ]
            # Build success-rate buckets (0%,10%,...,100%) for zero-discount orders based on per-customer success rates
            zero_subgroups = []
            if zero_total > 0:
                # Aggregate zero-discount orders per customer
                customer_stats = {}
                for cust_id, is_success in zero_orders:
                    if cust_id not in customer_stats:
                        customer_stats[cust_id] = {'total': 0, 'success': 0}
                    customer_stats[cust_id]['total'] += 1
                    if is_success:
                        customer_stats[cust_id]['success'] += 1

                # Bucket customers by rounded-to-10 success rate and sum their order counts
                bucket_to_orders = {k: 0 for k in range(0, 101, 10)}
                for stats in customer_stats.values():
                    total = stats['total']
                    rate = (stats['success'] / total) * 100 if total > 0 else 0.0
                    bucket = int(round(rate / 10.0) * 10)
                    bucket = max(0, min(100, bucket))
                    bucket_to_orders[bucket] += total

                # Build subgroup list sorted by rate ascending
                for bucket in sorted(bucket_to_orders.keys()):
                    cnt = bucket_to_orders[bucket]
                    if cnt > 0:
                        zero_subgroups.append({'rate': float(bucket), 'count': cnt, 'label': f'No discount {bucket}%'})
            result['binary']['zero_subgroups'] = zero_subgroups

        # 2) Grouped dataset for positive discounts only
        if pos_points:
            pos_ranges, pos_rates, pos_counts = [], [], []
            start_idx = 0
            for i in range(num_groups):
                current_group_size = group_size + (1 if i < remainder else 0)
                if current_group_size == 0:
                    break

                end_idx = start_idx + current_group_size
                group_points = pos_points[start_idx:end_idx]
                if not group_points:
                    continue

                # Group statistics
                min_discount = group_points[0][0]
                max_discount = group_points[-1][0]
                successful_count = sum(1 for _, is_success in group_points if is_success)
                success_rate = (successful_count / len(group_points)) * 100

                # Range label formatting
                if max_discount >= 1_000_000:
                    range_str = f'{min_discount / 1_000_000:.1f}M-{max_discount / 1_000_000:.1f}M'
                elif max_discount >= 1_000:
                    range_str = f'{min_discount / 1_000:.0f}K-{max_discount / 1_000:.0f}K'
                else:
                    range_str = f'{min_discount:.0f}-{max_discount:.0f}'

                pos_ranges.append(range_str)
                pos_rates.append(success_rate)
                pos_counts.append(len(group_points))

                start_idx = end_idx

            result['grouped']['ranges'] = pos_ranges
            result['grouped']['rates'] = pos_rates
            result['grouped']['orders_count'] = pos_counts

        grouped_ranges = result.get('grouped', {}).get('ranges', [])
        print(f"Created {len(grouped_ranges)} groups")
        return result

    except Exception as e:
        print(f"Error preparing discount success data: {str(e)}")
        return None

def _create_discount_success_chart(data):
    """Create a figure with two subplots:
    - Left: binary (No discount vs With discount)
    - Right: grouped positive discounts
    """
    if not data:
        return False

    try:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 8))

        # Left subplot: binary (No discount split into 5 subgroup points)
        if data.get('binary'):
            labels = data['binary']['labels']
            rates = data['binary']['rates']
            counts = data['binary']['counts']
            x = np.arange(len(labels))  # 0,1
            # Plot zero-discount subgroups around x=0
            subgroups = data['binary'].get('zero_subgroups', [])
            if subgroups:
                n = len(subgroups)
                # spread points wider around x=0 to reduce overlap even more
                offsets = [0.0] if n == 1 else list(np.linspace(-0.8, 0.8, n))
                for off, sg in zip(offsets, subgroups):
                    r = sg['rate']
                    c = sg['count']
                    color = '#87CEEB' if r < 50 else ('#4169E1' if r <= 80 else '#000080')
                    size = max(120, min(260, c))
                    ax_left.scatter(x[0] + off, r, s=size, c=color, alpha=0.7)
                    ax_left.text(x[0] + off, r + 3, f"{r:.1f}%\n(n={c})", ha='center', va='bottom', fontsize=9)
            else:
                r0, c0 = rates[0], counts[0]
                color = '#87CEEB' if r0 < 50 else ('#4169E1' if r0 <= 80 else '#000080')
                size = max(120, min(260, c0))
                ax_left.scatter(x[0], r0, s=size, c=color, alpha=0.7)
                ax_left.text(x[0], r0 + 3, f"{r0:.1f}%\n(n={c0})", ha='center', va='bottom', fontsize=9)

            # Single with-discount point at x=1
            r1 = rates[1] if len(rates) > 1 else 0.0
            c1 = counts[1] if len(counts) > 1 else 0
            color1 = '#87CEEB' if r1 < 50 else ('#4169E1' if r1 <= 80 else '#000080')
            size1 = max(120, min(260, c1))
            ax_left.scatter(x[1], r1, s=size1, c=color1, alpha=0.7)
            ax_left.text(x[1], r1 + 3, f"{r1:.1f}%\n(n={c1})", ha='center', va='bottom', fontsize=9)
            ax_left.set_title('Success Rate: No discount vs With discount', fontsize=12, pad=12)
            ax_left.set_xticks(x)
            ax_left.set_xticklabels(labels)
            ax_left.set_ylim(-5, 112)
            ax_left.set_ylabel('Success Rate (%)', fontsize=10)
            ax_left.grid(True, linestyle='--', alpha=0.3, axis='y')
            # Add vertical line to separate discount/no discount blocks
            ax_left.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            # widen x-limits to fully show spread
            ax_left.set_xlim(-1.1, 1.4)

        # Right subplot: grouped positive discounts (scatter)
        grouped = data.get('grouped', {})
        ranges = grouped.get('ranges', [])
        rates = grouped.get('rates', [])
        counts = grouped.get('orders_count', [])

        x_points, y_points, sizes, colors = [], [], [], []
        for i, (rate, count) in enumerate(zip(rates, counts)):
            if count > 0:
                x_points.append(i)
                y_points.append(rate)
                sizes.append(max(80, min(150, count / 2)))
                if rate < 50:
                    colors.append('#87CEEB')
                elif rate <= 80:
                    colors.append('#4169E1')
                else:
                    colors.append('#000080')

        ax_right.scatter(x_points, y_points, s=sizes, alpha=0.6, c=colors)
        avg_orders = (sum(counts) // len(counts)) if counts else 0
        ax_right.set_title(
            f'Success Rate by Discount Amount\n(each point ~{avg_orders} orders; size shows relative count)',
            fontsize=12, pad=12)
        ax_right.set_xlabel('Discount Amount Range', fontsize=10)
        ax_right.set_ylabel('Success Rate (%)', fontsize=10)
        ax_right.set_ylim(-5, 112)
        if len(ranges) <= 10:
            ax_right.set_xticks(range(len(ranges)))
            ax_right.set_xticklabels(ranges, rotation=45, ha='right')
        else:
            ax_right.set_xticks(range(len(ranges))[::2])
            ax_right.set_xticklabels([ranges[i] for i in range(0, len(ranges), 2)], rotation=45, ha='right')
        ax_right.grid(True, linestyle='--', alpha=0.7)
        ax_right.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax_right.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        ax_right.axhline(y=80, color='gray', linestyle='--', alpha=0.3)
        ax_right.axhline(y=100, color='gray', linestyle='-', alpha=0.3)

        # Legend for right subplot
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB', markersize=10, label='Success Rate < 50%'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4169E1', markersize=10, label='Success Rate 50-80%'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#000080', markersize=10, label='Success Rate > 80%')
        ]
        ax_right.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        return True

    except Exception as e:
        print(f"Error creating discount-success chart: {str(e)}")
        return False

def create_discount_success_chart(csv_path, output_path):
    """Create and save the chart for success rate by discount total"""
    data = _prepare_discount_success_data(csv_path)

    if not data:
        print("Failed to prepare data")
        return False

    success = _create_discount_success_chart(data)

    if success:
        plt.savefig(f"{output_path}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{output_path}.svg", format='svg', bbox_inches='tight')
        plt.close()

        print("Chart successfully saved to files:")
        print(f"- PNG: {output_path}.png")
        print(f"- SVG: {output_path}.svg")

        print("\nSummary Statistics:")
        binary_counts = data.get('binary', {}).get('counts', [])
        grouped_counts = data.get('grouped', {}).get('orders_count', [])
        total_orders = sum(binary_counts) if binary_counts else sum(grouped_counts)
        num_groups = len(data.get('grouped', {}).get('ranges', []))
        avg_per_group = (sum(grouped_counts) // len(grouped_counts)) if grouped_counts else 0
        print(f"Total orders analyzed: {total_orders}")
        print(f"Number of groups created: {num_groups}")
        print(f"Average orders per group: {avg_per_group}")
        return True

    print("Failed to create chart")
    return False


if __name__ == '__main__':
    # Example usage
    csv_path = 'data_collector_extended.csv'
    output_path = 'discount_success_chart'
    create_discount_success_chart(csv_path, output_path)


