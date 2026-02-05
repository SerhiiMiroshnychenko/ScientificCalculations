import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mplfonts import use_font
import csv
from datetime import datetime
from io import StringIO

# Налаштовуємо шрифт Arial для англійських символів
use_font('Times New Roman')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False  # Вимикаємо LaTeX

def _read_csv_data(csv_path):
    """Read CSV data from file"""
    print("Starting _read_csv_data")
    
    try:
        # Read CSV data
        csv_file = open(csv_path, 'r', encoding='utf-8')
        reader = csv.DictReader(csv_file)
        data = []
        for row in reader:
            try:
                # Обрізаємо мікросекунди з дат
                if 'date_order' in row:
                    date_str = row['date_order'].split('.')[0]  # Видаляємо мікросекунди
                    row['date_order'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue

            try:
                if 'create_date' in row:
                    date_str = row['create_date'].split('.')[0]  # Видаляємо мікросекунди
                    row['create_date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue

            # Ensure required fields are present - адаптуємо під реальну структуру CSV
            if not all(field in row for field in
                       ['order_id', 'customer_id', 'date_order', 'state', 'total_amount']):
                print(f"Missing required fields in row: {row}")
                continue

            data.append(row)

        csv_file.close()
        print(f"Successfully read {len(data)} rows from CSV")
        return data

    except Exception as e:
        print(f"Error reading CSV data: {str(e)}")
        return []

def _prepare_amount_success_data(csv_path):
    """Prepare data for amount success chart - групую замовлення за сумою"""
    print("Starting _prepare_amount_success_data")
    
    try:
        # Read CSV data
        data = _read_csv_data(csv_path)
        if not data:
            print("No data available")
            return None

        # Prepare data points - кожне замовлення як окрема точка
        data_points = []
        for row in data:
            amount = float(row['total_amount'])
            if amount > 0:  # Виключаємо записи з нульовою сумою
                is_successful = row['state'] == 'sale'
                data_points.append((amount, is_successful))

        if not data_points:
            print("No valid data points found")
            return None

        # Sort by amount
        data_points.sort(key=lambda x: x[0])

        total_points = len(data_points)
        print(f"Total orders with valid amounts: {total_points}")

        # Визначаємо кількість груп (зменшуємо якщо замовлень мало)
        num_groups = min(30, total_points // 20)  # Мінімум 20 замовлень на групу
        if num_groups < 5:  # Якщо груп менше 5, встановлюємо мінімум 5 груп
            num_groups = 5

        # Розраховуємо розмір кожної групи
        group_size = total_points // num_groups
        remainder = total_points % num_groups

        # Ініціалізуємо результат
        result = {
            'ranges': [],
            'rates': [],
            'orders_count': []
        }

        # Розбиваємо на групи
        start_idx = 0
        for i in range(num_groups):
            # Додаємо +1 до розміру групи для перших remainder груп
            current_group_size = group_size + (1 if i < remainder else 0)
            if current_group_size == 0:
                break

            end_idx = start_idx + current_group_size
            group_points = data_points[start_idx:end_idx]

            # Рахуємо статистику для групи
            min_amount = group_points[0][0]
            max_amount = group_points[-1][0]
            successful_count = sum(1 for _, is_success in group_points if is_success)
            success_rate = (successful_count / len(group_points)) * 100

            # Форматуємо діапазон
            if max_amount >= 1000000:
                range_str = f'{min_amount / 1000000:.1f}M-{max_amount / 1000000:.1f}M'
            elif max_amount >= 1000:
                range_str = f'{min_amount / 1000:.0f}K-{max_amount / 1000:.0f}K'
            else:
                range_str = f'{min_amount:.0f}-{max_amount:.0f}'

            # Додаємо дані до результату
            result['ranges'].append(range_str)
            result['rates'].append(success_rate)
            result['orders_count'].append(len(group_points))

            start_idx = end_idx

        print(f"Created {len(result['ranges'])} groups")
        return result

    except Exception as e:
        print(f"Error preparing amount success data: {str(e)}")
        return None

def _create_amount_success_chart(data):
    """Create chart showing success rate by order amount"""
    if not data:
        return False

    try:
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

        # Створюємо кольорову гамму з більш контрастними кольорами
        # Менше 50% - темно-синій (#003E6B)
        # 50-80% - середньо-синій (#00629B) 
        # Більше 80% - світло-синій (#0A5D7F)
        colors = []
        for rate in y_points:
            if rate < 50:
                colors.append('#87CEEB')  # Світло-синій
            elif rate <= 80:
                colors.append('#4169E1')  # Середньо-синій
            else:
                colors.append('#000080')  # Темно-синій
        
        sizes = [max(80, min(150, count / 2)) for count in counts]  # Розмір точки залежить від кількості замовлень

        # Малюємо точки
        scatter = plt.scatter(x_points, y_points, s=sizes, alpha=0.6, c=colors)

        # Розраховуємо середню кількість ордерів на точку
        avg_orders = sum(counts) // len(counts) if counts else 0

        plt.title(
            f'Success Rate by Order Amount\n(each point represents ~{avg_orders} orders)',
            pad=20, fontsize=14)
        plt.xlabel('Order Amount Range', fontsize=14)
        plt.ylabel('Success Rate (%)', fontsize=14)

        # Налаштовуємо осі
        plt.ylim(-5, 105)  # Додаємо трохи простору зверху і знизу

        # Показуємо всі мітки, якщо їх менше 10, інакше кожну другу
        if len(data['ranges']) <= 10:
            plt.xticks(range(len(data['ranges'])), data['ranges'],
                       rotation=45, ha='right', fontsize=14)
        else:
            plt.xticks(range(len(data['ranges']))[::2],
                       rotation=45, ha='right', fontsize=14)

        plt.yticks(fontsize=14)

        plt.grid(True, linestyle='--', alpha=0.7)

        # Додаємо горизонтальні лінії
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=80, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=100, color='gray', linestyle='-', alpha=0.3)

        # Додаємо легенду з новими кольорами
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#87CEEB', markersize=10,
                       label='Success Rate < 50%'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#4169E1', markersize=10,
                       label='Success Rate 50-80%'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#000080', markersize=10,
                       label='Success Rate > 80%')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        return True

    except Exception as e:
        print(f"Error creating amount-success chart: {str(e)}")
        return False

def create_amount_success_chart(csv_path, output_path):
    """
    Creates a scatter plot showing the relationship between order amount and success rate

    Args:
        csv_path (str): Path to CSV file with data
        output_path (str): Base path for saving the chart (without extension)

    Returns:
        bool: True on success
    """
    # Prepare data
    data = _prepare_amount_success_data(csv_path)
    
    if not data:
        print("Failed to prepare data")
        return False
    
    # Create the chart
    success = _create_amount_success_chart(data)
    
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
    output_path = 'amount_success_chart'
    create_amount_success_chart(csv_path, output_path)
