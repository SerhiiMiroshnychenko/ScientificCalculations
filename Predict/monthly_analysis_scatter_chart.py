"""
Скрипт для створення графіку розсіювання місячної аналітики замовлень.
Читає дані з CSV файлу та створює графік з трьома метриками:
- Загальна кількість замовлень
- Кількість успішних замовлень
- Відсоток успішності
"""

import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def create_monthly_scatter_chart(csv_path, output_path):
    try:
        # Підготовка місячних даних
        monthly_data = {}

        # Читання CSV файлу
        with open(csv_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)

            # Збір даних по місяцях
            for row in reader:
                date = datetime.strptime(row['date_order'].split()[0], '%Y-%m-%d')
                month_key = date.strftime('%Y-%m')

                if month_key not in monthly_data:
                    monthly_data[month_key] = {
                        'orders': 0,
                        'successful': 0,
                        'rate': 0,
                        'date': date.replace(day=1)
                    }

                monthly_data[month_key]['orders'] += 1
                if row['state'] == 'sale':
                    monthly_data[month_key]['successful'] += 1

        # Розрахунок відсотку успішності
        for month in monthly_data:
            total = monthly_data[month]['orders']
            successful = monthly_data[month]['successful']
            monthly_data[month]['rate'] = (successful / total * 100) if total > 0 else 0

        # Сортування місяців хронологічно
        sorted_months = sorted(monthly_data.keys())

        if not sorted_months:
            print("Немає даних для відображення")
            return False

        # Створення масивів даних у хронологічному порядку
        dates = [monthly_data[month]['date'] for month in sorted_months]
        orders_data = [monthly_data[month]['orders'] for month in sorted_months]
        successful_data = [monthly_data[month]['successful'] for month in sorted_months]
        rate_data = [monthly_data[month]['rate'] for month in sorted_months]

        # Створення графіку розсіювання
        fig, ax1 = plt.subplots(figsize=(15, 8))

        # Створення другої осі Y
        ax2 = ax1.twinx()

        # Графіки для кількості замовлень (ліва вісь)
        scatter1 = ax1.scatter(dates, orders_data, color='skyblue', s=100, label='Всього замовлень')
        scatter2 = ax1.scatter(dates, successful_data, color='gold', s=100, label='Успішні замовлення')

        # Графік для відсотків (права вісь)
        scatter3 = ax2.scatter(dates, rate_data, color='purple', s=100, label='Відсоток успішності (%)')

        # Форматування осі X
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Налаштування лівої осі (кількість)
        ax1.set_xlabel('Місяць')
        ax1.set_ylabel('Кількість')
        ax1.tick_params(axis='y', labelcolor='black')

        # Налаштування правої осі (відсотки)
        ax2.set_ylabel('Відсоток успішності (%)')
        ax2.tick_params(axis='y', labelcolor='purple')

        # Об'єднання легенд з обох осей
        handles = [scatter1, scatter2, scatter3]
        labels = ['Всього замовлень', 'Успішні замовлення', 'Відсоток успішності (%)']
        ax1.legend(handles, labels, loc='upper left')

        plt.title('Місячний аналіз замовлень (Графік розсіювання)')
        plt.grid(True)
        plt.tight_layout()

        # Збереження графіку
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
    output_path = 'monthly_analysis_scatter_chart'
    create_monthly_scatter_chart(csv_path, output_path)
