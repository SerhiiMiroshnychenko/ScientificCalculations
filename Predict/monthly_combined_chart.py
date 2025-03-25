"""
Скрипт для створення комбінованого місячного графіку з аналізом замовлень.
Показує:
- Загальна кількість замовлень
- Відсоток успішності
- Відносний вік клієнтів
"""

import csv
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict


def create_monthly_combined_chart(csv_path, output_path):
    try:
        # Читаємо CSV файл
        data = []
        with open(csv_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # Перетворюємо строкові дати в об'єкти datetime
                row['date_order'] = datetime.strptime(row['date_order'].split()[0], '%Y-%m-%d')
                row['partner_create_date'] = datetime.strptime(row['partner_create_date'].split()[0], '%Y-%m-%d')
                data.append(row)

        if not data:
            print("Немає даних для аналізу")
            return False

        # Знаходимо найранішу дату
        date_from = min(row['date_order'] for row in data)

        # Групуємо дані по місяцях
        monthly_data = defaultdict(lambda: {
            'total_orders': 0,
            'successful_orders': 0,
            'customer_ages': []
        })

        for row in data:
            order_date = row['date_order']
            month_key = order_date.strftime('%Y-%m')

            # Рахуємо замовлення
            monthly_data[month_key]['total_orders'] += 1

            # Рахуємо успішні замовлення
            if row['state'] in ['done', 'sale']:
                monthly_data[month_key]['successful_orders'] += 1

            # Розраховуємо відносний вік клієнта
            customer_since = row['partner_create_date']
            total_time = (order_date - date_from).days / 30.0  # Загальний час у місяцях
            customer_age = (order_date - customer_since).days / 30.0  # Вік у місяцях
            relative_age = (customer_age / total_time * 100) if total_time > 0 else 0
            # Якщо відносний вік від'ємний - прирівнюємо до 0
            relative_age = max(0, relative_age)
            monthly_data[month_key]['customer_ages'].append(relative_age)

        # Сортуємо місяці
        sorted_months = sorted(monthly_data.keys())

        # Готуємо дані для графіку
        months = []
        orders_count = []
        success_rates = []
        avg_relative_ages = []

        for month in sorted_months:
            data = monthly_data[month]
            total = data['total_orders']
            successful = data['successful_orders']
            ages = data['customer_ages']

            months.append(datetime.strptime(month, '%Y-%m'))
            orders_count.append(total)
            success_rates.append((successful / total * 100) if total > 0 else 0)
            avg_relative_ages.append(sum(ages) / len(ages) if ages else 0)

        # Створюємо графік та основну вісь
        fig, ax1 = plt.subplots(figsize=(15, 8))

        # Основна вісь - Загальна кількість замовлень (синій)
        color1 = 'blue'
        ax1.set_xlabel('Місяць')
        ax1.set_ylabel('Загальна кількість замовлень', color=color1)
        ax1.plot(months, orders_count, color=color1, marker='o', markersize=8,
                 label='Загальна кількість замовлень', alpha=0.7, linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color1)

        # Друга вісь - Відсоток успішності (помаранчевий)
        ax2 = ax1.twinx()
        color2 = 'orange'
        ax2.set_ylabel('Відсоток успішності (%)', color=color2)
        ax2.plot(months, success_rates, color=color2, marker='s', markersize=8,
                 label='Відсоток успішності (%)', alpha=0.7, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Третя вісь - Відносний вік клієнтів (зелений)
        ax3 = ax1.twinx()
        # Зміщуємо третю вісь
        ax3.spines["right"].set_position(("axes", 1.1))
        color3 = 'green'
        ax3.set_ylabel('Відносний вік клієнтів (%)', color=color3)
        ax3.plot(months, avg_relative_ages, color=color3, marker='^', markersize=8,
                 label='Відносний вік клієнтів (%)\n(% часу від першого замовлення)', alpha=0.7, linestyle=':')
        ax3.tick_params(axis='y', labelcolor=color3)

        # Форматуємо вісь X
        ax1.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=6))
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Додаємо легенду з поясненням
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
                   loc='center', bbox_to_anchor=(0.15, 0.9))

        plt.title(
            'Комбінований місячний аналіз\nВідносний вік клієнтів показує середній вік клієнта як відсоток часу, що минув від першого замовлення')
        plt.grid(True)
        plt.tight_layout()

        # Зберігаємо у PNG форматі
        png_path = output_path + '.png'
        plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

        # Зберігаємо у SVG форматі
        svg_path = output_path + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')

        plt.close()
        print(f"Графік успішно збережено у файли:")
        print(f"- PNG: {png_path}")
        print(f"- SVG: {svg_path}")
        return True

    except Exception as e:
        print(f'Помилка при створенні графіку: {e.__class__}: {e}')
        return False
    finally:
        plt.close('all')


if __name__ == '__main__':
    csv_path = 'data_collector_simple.csv'
    output_path = 'monthly_combined_chart'
    create_monthly_combined_chart(csv_path, output_path)
