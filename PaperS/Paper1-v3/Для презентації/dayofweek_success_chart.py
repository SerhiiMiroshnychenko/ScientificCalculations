import matplotlib.pyplot as plt
import matplotlib as mpl
from mplfonts import use_font
import csv
from datetime import datetime

# Configure Arial font for English labels
use_font('Arial')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False


def _read_csv_data(csv_path):
    """Read CSV data from file"""
    print("Starting _read_csv_data")

    try:
        csv_file = open(csv_path, 'r', encoding='utf-8')
        reader = csv.DictReader(csv_file)
        data = []
        for row in reader:
            try:
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

            if not all(field in row for field in ['order_id', 'date_order', 'state', 'day_of_week']):
                print(f"Missing required fields in row: {row}")
                continue

            data.append(row)

        csv_file.close()
        print(f"Successfully read {len(data)} rows from CSV")
        return data

    except Exception as e:
        print(f"Error reading CSV data: {str(e)}")
        return []


def _prepare_dayofweek_success_data(csv_path):
    """Prepare success rate per day of week with order counts"""
    print("Starting _prepare_dayofweek_success_data")

    try:
        data = _read_csv_data(csv_path)
        if not data:
            print("No data available")
            return None

        # Fixed order of weekdays
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        totals = {d: 0 for d in weekdays}
        successes = {d: 0 for d in weekdays}

        for row in data:
            day = row.get('day_of_week')
            if day not in totals:
                continue
            totals[day] += 1
            if row.get('state') == 'sale':
                successes[day] += 1

        ranges = []
        rates = []
        counts = []
        for d in weekdays:
            total = totals[d]
            success_rate = (successes[d] / total * 100) if total > 0 else 0.0
            ranges.append(d)
            rates.append(success_rate)
            counts.append(total)

        return {
            'ranges': ranges,
            'rates': rates,
            'orders_count': counts,
        }

    except Exception as e:
        print(f"Error preparing day-of-week success data: {str(e)}")
        return None


def _create_dayofweek_success_chart(data):
    """Create chart showing success rate by day of week"""
    if not data:
        return False

    try:
        plt.figure(figsize=(15, 8))

        x_points = []
        y_points = []
        counts = []
        for i, (rate, count) in enumerate(zip(data['rates'], data['orders_count'])):
            if count > 0:
                x_points.append(i)
                y_points.append(rate)
                counts.append(count)

        colors = []
        for rate in y_points:
            if rate < 50:
                colors.append('#87CEEB')
            elif rate <= 80:
                colors.append('#4169E1')
            else:
                colors.append('#000080')

        sizes = [max(100, min(260, count)) for count in counts]

        plt.scatter(x_points, y_points, s=sizes, alpha=0.6, c=colors)

        avg_orders = sum(counts) // len(counts) if counts else 0

        plt.title(
            f'Success Rate by Day of Week\n(each point represents ~{avg_orders} orders, point size shows relative number per day)',
            pad=20, fontsize=12)
        plt.xlabel('Day of Week', fontsize=10)
        plt.ylabel('Success Rate (%)', fontsize=10)

        plt.ylim(-5, 105)

        plt.xticks(range(len(data['ranges'])), data['ranges'], rotation=0, ha='center')

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=80, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=100, color='gray', linestyle='-', alpha=0.3)

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB', markersize=10, label='Success Rate < 50%'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4169E1', markersize=10, label='Success Rate 50-80%'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#000080', markersize=10, label='Success Rate > 80%')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        return True

    except Exception as e:
        print(f"Error creating day-of-week success chart: {str(e)}")
        return False


def create_dayofweek_success_chart(csv_path, output_path):
    """Create and save the chart for success rate by day of week"""
    data = _prepare_dayofweek_success_data(csv_path)

    if not data:
        print("Failed to prepare data")
        return False

    success = _create_dayofweek_success_chart(data)

    if success:
        plt.savefig(f"{output_path}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{output_path}.svg", format='svg', bbox_inches='tight')
        plt.close()

        print("Chart successfully saved to files:")
        print(f"- PNG: {output_path}.png")
        print(f"- SVG: {output_path}.svg")

        print("\nSummary Statistics:")
        total_orders = sum(data['orders_count'])
        print(f"Total orders analyzed: {total_orders}")
        print(f"Number of days: {len(data['ranges'])}")
        print(f"Average orders per day: {total_orders // len(data['orders_count']) if data['orders_count'] else 0}")
        return True

    print("Failed to create chart")
    return False


if __name__ == '__main__':
    # Example usage
    csv_path = 'data_collector_extended.csv'
    output_path = 'dayofweek_success_chart'
    create_dayofweek_success_chart(csv_path, output_path)


