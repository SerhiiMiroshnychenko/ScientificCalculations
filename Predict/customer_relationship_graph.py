"""
Скрипт для створення графіку аналізу терміну співпраці з клієнтами.
Створює комбінований графік, який показує:
- Кількість клієнтів за категоріями терміну співпраці
- Кількість замовлень за категоріями
- Відсоток успішності замовлень за категоріями
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_relationship_category(months):
    """Визначає категорію терміну співпраці на основі кількості місяців"""
    if months < 2:
        return 'Нові'
    elif 2 <= months < 6:
        return '2-6 місяців'
    elif 6 <= months < 12:
        return '6-12 місяців'
    elif 12 <= months < 24:
        return '1-2 роки'
    else:
        return '2+ роки'


def create_customer_relationship_chart(csv_path, output_path):
    """
    Створює графік аналізу терміну співпраці з клієнтами.

    Параметри:
    ----------
    csv_path : str
        Шлях до CSV файлу з даними
    output_path : str
        Базовий шлях для збереження графіку (без розширення)

    Повертає:
    --------
    bool
        True якщо графік успішно створено, False якщо виникла помилка
    """
    # Читаємо дані з CSV файлу
    df = pd.read_csv(csv_path)

    # Конвертуємо дні співпраці в місяці
    df['relationship_months'] = pd.to_numeric(df['customer_relationship_days'],
                                              errors='coerce').fillna(0) / 30

    # Визначаємо порядок категорій
    category_order = ['Нові', '2-6 місяців', '6-12 місяців', '1-2 роки', '2+ роки']

    # Застосовуємо категоризацію
    df['relationship_category'] = df['relationship_months'].apply(get_relationship_category)

    # Визначаємо успішність замовлень
    df['is_successful'] = df['state'] == 'sale'

    # Отримуємо останнє замовлення для кожного клієнта
    latest_orders = df.sort_values('date_order').groupby('customer_id').last()

    # Категоризуємо клієнтів на основі їх останнього замовлення
    latest_orders['relationship_category'] = latest_orders['relationship_months'].apply(get_relationship_category)

    # Рахуємо статистику
    category_counts = latest_orders['relationship_category'].value_counts()
    category_counts = category_counts.reindex(category_order)

    orders_counts = df['relationship_category'].value_counts()
    orders_counts = orders_counts.reindex(category_order)

    success_by_category = df.groupby('relationship_category')['is_successful'].mean()
    success_by_category = success_by_category.reindex(category_order)

    # Створюємо графік
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Позиції для стовпчиків
    x = np.arange(len(category_order))
    width = 0.35

    # Стовпчики для кількості клієнтів
    bars1 = ax1.bar(x - width / 2, category_counts.values, width,
                    color='#1f77b4', label='Кількість клієнтів')
    ax1.set_xlabel('Термін співпраці')
    ax1.set_ylabel('Кількість клієнтів', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    # Додаємо значення над стовпчиками клієнтів
    for i, v in enumerate(category_counts.values):
        ax1.text(x[i] - width / 2, v, str(int(v)), ha='center', va='bottom')

    # Друга вісь для кількості замовлень
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    bars2 = ax3.bar(x + width / 2, orders_counts.values, width,
                    color='skyblue', label='Кількість замовлень')
    ax3.set_ylabel('Кількість замовлень', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')

    # Додаємо значення над стовпчиками замовлень
    for i, v in enumerate(orders_counts.values):
        ax3.text(x[i] + width / 2, v, str(int(v)), ha='center', va='bottom')

    # Третя вісь для відсотка успішності
    ax2 = ax1.twinx()
    success_line = ax2.plot(x, success_by_category.values * 100, 'o-',
                            color='gold', linewidth=2, markersize=8,
                            label='Відсоток успішності (%)')
    ax2.set_ylabel('Відсоток успішності (%)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 100)

    # Додаємо значення на лінії успішності
    for i, v in enumerate(success_by_category.values):
        ax2.text(x[i], v * 100 + 2, f'{v * 100:.1f}%', ha='center', va='bottom')

    # Налаштування міток осі X
    ax1.set_xticks(x)
    ax1.set_xticklabels(category_order, rotation=45)

    # Додаємо заголовок
    plt.title('Розподіл клієнтів та замовлень відносно терміну співпраці')

    # Об'єднуємо легенди
    lines1, labels1 = ax1.get_legend_handles_labels()  # Кількість клієнтів
    lines2, labels2 = ax2.get_legend_handles_labels()  # Відсоток успішності
    lines3, labels3 = ax3.get_legend_handles_labels()  # Кількість замовлень

    # Змінюємо порядок елементів легенди відповідно до оригіналу та розміщуємо зверху
    ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
               loc='center', bbox_to_anchor=(0.25, 0.9))

    plt.tight_layout()

    # Зберігаємо графік
    plt.savefig(f"{output_path}.png", format='png',
                bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_path}.svg", format='svg',
                bbox_inches='tight')

    plt.close()
    print(f"Графік успішно збережено у файли:")
    print(f"- PNG: {output_path}.png")
    print(f"- SVG: {output_path}.svg")
    return True


if __name__ == '__main__':
    # Приклад використання
    csv_path = 'data_collector_extended.csv'
    output_path = 'customer_relationship_graph'
    create_customer_relationship_chart(csv_path, output_path)
