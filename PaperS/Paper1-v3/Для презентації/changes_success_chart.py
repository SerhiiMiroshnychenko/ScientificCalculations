import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mplfonts import use_font
import csv
from datetime import datetime
from io import StringIO

# Configure Arial font for English labels
use_font('Times New Roman')
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
                       ['order_id', 'customer_id', 'date_order', 'state', 'changes_count']):
                print(f"Missing required fields in row: {row}")
                continue

            data.append(row)

        csv_file.close()
        print(f"Successfully read {len(data)} rows from CSV")
        return data

    except Exception as e:
        print(f"Error reading CSV data: {str(e)}")
        return []

def _prepare_changes_success_data(csv_path):
    """Prepare data for changes success chart - group orders by changes count"""
    print("Starting _prepare_changes_success_data")

    try:
        data = _read_csv_data(csv_path)
        if not data:
            print("No data available")
            return None

        # Prepare data points - each order as a separate point
        data_points = []
        for row in data:
            try:
                changes_count = int(row['changes_count'])
                if changes_count >= 0:  # Exclude negative values
                    is_successful = row['state'] == 'sale'
                    data_points.append((changes_count, is_successful))
            except (ValueError, TypeError):
                continue

        if not data_points:
            print("No valid data points found")
            return None

        # Sort by changes count
        data_points.sort(key=lambda x: x[0])

        total_points = len(data_points)
        print(f"Total orders with valid changes count: {total_points}")

        # Determine number of groups (reduce if few orders)
        num_groups = 6  # Fixed number of groups

        # Calculate group size
        group_size = total_points // num_groups
        remainder = total_points % num_groups

        # Initialize result
        result = {
            'ranges': [],
            'rates': [],
            'orders_count': []
        }

        # Split into groups
        start_idx = 0
        for i in range(num_groups):
            # Add +1 to group size for first remainder groups
            current_group_size = group_size + (1 if i < remainder else 0)
            if current_group_size == 0:
                break

            end_idx = start_idx + current_group_size
            group_points = data_points[start_idx:end_idx]

            # Calculate group statistics
            min_changes = group_points[0][0]
            max_changes = group_points[-1][0]
            successful_count = sum(1 for _, is_success in group_points if is_success)
            success_rate = (successful_count / len(group_points)) * 100

            # Format range
            if min_changes == max_changes:
                range_str = f'{min_changes}'
            else:
                range_str = f'{min_changes}-{max_changes}'

            # Add data to result
            result['ranges'].append(range_str)
            result['rates'].append(success_rate)
            result['orders_count'].append(len(group_points))

            start_idx = end_idx

        print(f"Created {len(result['ranges'])} groups")
        return result

    except Exception as e:
        print(f"Error preparing changes success data: {str(e)}")
        return None

def _create_changes_success_chart(data):
    """Create chart showing success rate by changes count"""
    if not data:
        return False

    try:
        plt.figure(figsize=(15, 8))

        # Filter points with zero order count
        x_points = []
        y_points = []
        counts = []
        for i, (rate, count) in enumerate(zip(data['rates'], data['orders_count'])):
            if count > 0:
                x_points.append(i)
                y_points.append(rate)
                counts.append(count)

        # Create color palette with more contrasting colors
        # Less than 50% - light blue (#87CEEB)
        # 50-80% - medium blue (#4169E1)
        # More than 80% - dark blue (#000080)
        colors = []
        for rate in y_points:
            if rate < 50:
                colors.append('#87CEEB')  # Light blue
            elif rate <= 80:
                colors.append('#4169E1')  # Medium blue
            else:
                colors.append('#000080')  # Dark blue

        sizes = [max(80, min(150, count / 2)) for count in counts]  # Point size depends on order count

        # Draw points
        scatter = plt.scatter(x_points, y_points, s=sizes, alpha=0.6, c=colors)

        # Add trend line (blue approximation line)
        if len(x_points) > 1:
            z = np.polyfit(x_points, y_points, 1)
            p = np.poly1d(z)
            plt.plot(x_points, p(x_points), color='#4682B4', linestyle='--', alpha=0.8, linewidth=2, label='Trend line')

        # Calculate average orders per point
        avg_orders = sum(counts) // len(counts) if counts else 0

        plt.title(
            f'Success Rate by Changes Count\n(each point represents ~{avg_orders} orders)',
            pad=20, fontsize=14)
        plt.xlabel('Changes Count Range', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)

        # Configure axes
        plt.ylim(-5, 105)  # Add some space at top and bottom

        # Show all labels if less than 10, otherwise every other
        if len(data['ranges']) <= 10:
            plt.xticks(range(len(data['ranges'])), data['ranges'],
                       rotation=45, ha='right', fontsize=14)
        else:
            plt.xticks(range(len(data['ranges']))[::2],
                       [data['ranges'][i] for i in range(0, len(data['ranges']), 2)],
                       rotation=45, ha='right', fontsize=14)

        plt.yticks(fontsize=14)

        plt.grid(True, linestyle='--', alpha=0.7)

        # Add horizontal lines
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=80, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=100, color='gray', linestyle='-', alpha=0.3)

        # Add legend with new colors and trend line
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#87CEEB', markersize=10,
                       label='Success Rate < 50%'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#4169E1', markersize=10,
                       label='Success Rate 50-80%'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#000080', markersize=10,
                       label='Success Rate > 80%'),
            plt.Line2D([0], [0], color='#4682B4', linestyle='--', linewidth=2,
                       label='Trend line')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        return True

    except Exception as e:
        print(f"Error creating changes-success chart: {str(e)}")
        return False

def create_changes_success_chart(csv_path, output_path):
    """
    Creates a scatter plot showing the relationship between changes count and success rate

    Args:
        csv_path (str): Path to CSV file with data
        output_path (str): Base path for saving the chart (without extension)

    Returns:
        bool: True on success
    """
    # Prepare data
    data = _prepare_changes_success_data(csv_path)

    if not data:
        print("Failed to prepare data")
        return False

    # Create the chart
    success = _create_changes_success_chart(data)

    if success:
        # Save chart
        plt.savefig(f"{output_path}.png", format='png',
                    bbox_inches='tight', dpi=300)
        plt.savefig(f"{output_path}.svg", format='svg',
                    bbox_inches='tight')

        plt.close()

        print(f"Chart successfully saved to files:")
        print(f"- PNG: {output_path}.png")
        print(f"- SVG: {output_path}.svg")

        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Total orders analyzed: {sum(data['orders_count'])}")
        print(f"Number of groups created: {len(data['ranges'])}")
        print(f"Average orders per group: {sum(data['orders_count']) // len(data['orders_count'])}")

        return True
    else:
        print("Failed to create chart")
        return False


if __name__ == '__main__':
    # Example usage
    csv_path = 'data_collector_extended.csv'
    output_path = 'changes_success_chart'
    create_changes_success_chart(csv_path, output_path)
