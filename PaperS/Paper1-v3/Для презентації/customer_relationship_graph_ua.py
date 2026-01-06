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
import matplotlib as mpl
from datetime import datetime
import scienceplots
from mplfonts import use_font

# Використовуємо науковий стиль із пакету scienceplots
plt.style.use('science')

# Налаштовуємо шрифт Times New Roman для українських символів
use_font('Times New Roman')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False  # Вимикаємо LaTeX, щоб уникнути проблем із символами{{ ... }}


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

    # Встановлюємо розміри у пропорціях для наукової статті
    textwidth = 10  # ширина для наукової статті
    aspect_ratio = 0.6  # співвідношення сторін для наукового вигляду
    scale = 1.0
    figwidth = textwidth * scale
    figheight = figwidth * aspect_ratio

    # Створюємо графік з науковими параметрами
    fig, ax1 = plt.subplots(figsize=(figwidth, figheight))
    
    # Встановлюємо товсті лінії для осей
    width = 1.0  # Збільшена товщина ліній рамки
    ax1.spines["left"].set_linewidth(width)
    ax1.spines["bottom"].set_linewidth(width)
    ax1.spines["right"].set_linewidth(width)
    ax1.spines["top"].set_linewidth(width)

    # Налаштовуємо товщину засічок
    ax1.tick_params(width=width, length=6)
    
    # Вимикаємо проміжні засічки
    ax1.minorticks_off()

    # Позиції для стовпчиків
    x = np.arange(len(category_order))
    width = 0.35

    # Стовпчики для кількості клієнтів
    bars1 = ax1.bar(x - width / 2, category_counts.values, width,
                    color='#1f77b4', label='Кількість клієнтів')
    ax1.set_xlabel('Термін співпраці', fontsize=14)
    ax1.set_ylabel('Кількість клієнтів', color='#1f77b4', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    # Друга вісь для кількості замовлень
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.spines["right"].set_linewidth(width)  # Задаємо товщину лінії для правої осі
    ax3.tick_params(width=width, length=6)  # Налаштовуємо товщину засічок для правої осі
    ax3.minorticks_off()  # Вимикаємо проміжні засічки для правої осі
    
    bars2 = ax3.bar(x + width / 2, orders_counts.values, width,
                    color='skyblue', label='Кількість замовлень')
    ax3.set_ylabel('Кількість замовлень', color='blue', fontsize=14)
    ax3.tick_params(axis='y', labelcolor='blue', labelsize=12)

    # Третя вісь для відсотка успішності
    ax2 = ax1.twinx()
    ax2.spines["right"].set_linewidth(width)  # Задаємо товщину лінії для правої осі
    ax2.spines['right'].set_position(('outward', 0))
    ax2.tick_params(width=width, length=6)  # Налаштовуємо товщину засічок для правої осі
    ax2.minorticks_off()  # Вимикаємо проміжні засічки для правої осі
    
    success_line = ax2.plot(x, success_by_category.values * 100, 'o-',
                            color='gold', linewidth=2, markersize=8,
                            label='Відсоток успішності (%)')
    ax2.set_ylabel('Відсоток успішності (%)', color='black', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax2.set_ylim(0, 100)

    # Додаємо значення над стовпчиками з кольорами відповідних шкал
    for i, v in enumerate(category_counts.values):
        ax1.text(x[i] - width / 2, v, str(int(v)), ha='center', va='bottom', color='#1f77b4', fontsize=12)  # синій як шкала клієнтів

    for i, v in enumerate(orders_counts.values):
        ax3.text(x[i] + width / 2, v, str(int(v)), ha='center', va='bottom', color='blue', fontsize=12)  # синій як шкала замовлень

    # Додаємо значення для лінії відсотка успішності
    for i, v in enumerate(success_by_category.values):
        ax2.text(x[i], v * 100 + 2, f'{v * 100:.1f}%', ha='center', va='bottom', color='black', fontsize=12)  # чорний як шкала успішності

    # Налаштування міток осі X
    ax1.set_xticks(x)
    ax1.set_xticklabels(category_order, rotation=45)

    # Додаємо заголовок
    plt.title('Розподіл клієнтів та замовлень відносно терміну співпраці', fontsize=14, pad=20)

    # Об'єднуємо легенди
    lines1, labels1 = ax1.get_legend_handles_labels()  # Кількість клієнтів
    lines2, labels2 = ax2.get_legend_handles_labels()  # Відсоток успішності
    lines3, labels3 = ax3.get_legend_handles_labels()  # Кількість замовлень

    # Змінюємо порядок елементів легенди відповідно до оригіналу та розміщуємо зверху
    ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
               loc='center', bbox_to_anchor=(0.28, 0.88), fontsize=12)

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
