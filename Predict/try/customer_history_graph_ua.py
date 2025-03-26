import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib as mpl
import scienceplots
from mplfonts import use_font

# Використовуємо науковий стиль із пакету scienceplots
plt.style.use('science')

# Налаштовуємо шрифт Times New Roman для українських символів
use_font('Times New Roman')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False  # Вимикаємо LaTeX, щоб уникнути проблем із символами

def create_customer_history_chart(csv_path, output_path):
    """
    Створює графік аналізу історії клієнтів

    Args:
        csv_path (str): Шлях до CSV файлу з даними
        output_path (str): Базовий шлях для збереження графіка (без розширення)

    Returns:
        bool: True у разі успіху
    """
    # Зчитуємо дані
    df = pd.read_csv(csv_path)

    # Перевірка на коректність даних у колонці processing_time_hours
    df['processing_time_hours'] = pd.to_numeric(df['processing_time_hours'], errors='coerce')

    # Виведення некоректних даних
    invalid_data = df[df['processing_time_hours'].isna()]
    if not invalid_data.empty:
        print("Некоректні дані в колонці 'processing_time_hours':")
        print(invalid_data[['order_id', 'processing_time_hours']].to_string(index=False))

    # Перевірка та конвертація даних у колонці discount_total
    df['discount_total'] = pd.to_numeric(df['discount_total'], errors='coerce')

    # Виведення некоректних даних
    invalid_data = df[df['discount_total'].isna()]
    if not invalid_data.empty:
        print("Некоректні дані в колонці 'discount_total':")
        print(invalid_data[['order_id', 'discount_total']].to_string(index=False))

    # Заміна некоректних значень на 0
    df['discount_total'] = df['discount_total'].fillna(0)

    # Збереження базової статистики
    df['is_successful'] = df['state'].apply(lambda x: 1 if x == 'sale' else 0)
    
    # Конвертація дат з обробкою мікросекунд
    df['create_date'] = pd.to_datetime(df['create_date'].str.split('.').str[0])
    df['date_order'] = pd.to_datetime(df['date_order'].str.split('.').str[0])
    
    df['avg_response_time_days'] = abs((df['date_order'] - df['create_date']).dt.total_seconds() / (3600 * 24))

    # Встановлюємо розміри у пропорціях для наукової статті
    textwidth = 10  # ширина для наукової статті
    aspect_ratio = 0.6  # співвідношення сторін для наукового вигляду
    scale = 1.0
    figwidth = textwidth * scale
    figheight = figwidth * aspect_ratio

    # Створюємо фігуру з науковими параметрами
    fig, ax1 = plt.subplots(figsize=(figwidth, figheight))

    # Конвертуємо previous_orders_count в числовий формат
    df['previous_orders_count'] = pd.to_numeric(df['previous_orders_count'], errors='coerce').fillna(0).astype(int)

    # Встановлюємо товсті лінії для осей
    width = 1.0  # Збільшена товщина ліній рамки
    ax1.spines["left"].set_linewidth(width)
    ax1.spines["bottom"].set_linewidth(width)
    ax1.spines["right"].set_linewidth(width)
    ax1.spines["top"].set_linewidth(width)

    # Налаштовуємо товщину засічок
    ax1.tick_params(width=width, length=6)

    # Використовуємо previous_orders_count для категоризації
    def get_order_category(row):
        count = int(row['previous_orders_count'])
        if count == 0:
            return 'Нові'
        elif 1 <= count <= 4:
            return '2-5 замовлень'
        elif 5 <= count <= 9:
            return '6-10 замовлень'
        elif 10 <= count <= 19:
            return '11-20 замовлень'
        else:
            return '20+ замовлень'

    # Визначаємо порядок категорій
    category_order = ['Нові', '2-5 замовлень', '6-10 замовлень', '11-20 замовлень', '20+ замовлень']

    # Застосування категоризації до всіх замовлень
    df['customer_category'] = df.apply(get_order_category, axis=1)

    # Визначаємо успішність на основі state == 'sale'
    df['is_successful'] = df['state'] == 'sale'

    # Отримуємо останнє замовлення для кожного клієнта
    latest_orders = df.sort_values('date_order').groupby('customer_id').last()

    # Категоризуємо клієнтів на основі їх останнього замовлення
    latest_orders['customer_category'] = latest_orders.apply(get_order_category, axis=1)

    # Рахуємо кількість клієнтів в кожній категорії
    category_counts = latest_orders['customer_category'].value_counts()
    category_counts = category_counts.reindex(category_order)

    # Рахуємо кількість замовлень в кожній категорії
    orders_counts = df['customer_category'].value_counts()
    orders_counts = orders_counts.reindex(category_order)

    # Рахуємо відсоток успішності для кожної категорії (використовуємо всі замовлення)
    success_by_category = df.groupby('customer_category')['is_successful'].mean()
    success_by_category = success_by_category.reindex(category_order)

    # Створюємо позиції для стовпчиків
    x = np.arange(len(category_counts))
    width = 0.35  # ширина стовпчика

    # Створюємо стовпчики для клієнтів (зліва від центру)
    bars1 = ax1.bar(x - width / 2, category_counts.values, width,
                    color='#1f77b4', label='Кількість клієнтів')
    ax1.set_xlabel('Категорія', fontsize=14)
    ax1.set_ylabel('Кількість клієнтів', color='#1f77b4', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    # Створюємо другу вісь для замовлень
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.spines["right"].set_linewidth(width)  # Задаємо товщину лінії для правої осі
    ax3.tick_params(width=width, length=6)  # Налаштовуємо товщину засічок для правої осі
    # Створюємо стовпчики для замовлень (справа від центру)
    bars2 = ax3.bar(x + width / 2, orders_counts.values, width,
                    color='skyblue', label='Кількість замовлень')
    ax3.set_ylabel('Кількість замовлень', color='blue', fontsize=14)
    ax3.tick_params(axis='y', labelcolor='blue', labelsize=12)

    # Створюємо третю вісь для відсотка успішності
    ax2 = ax1.twinx()
    ax2.spines["right"].set_linewidth(width)  # Задаємо товщину лінії для правої осі
    ax2.spines['right'].set_position(('outward', 0))
    ax2.tick_params(width=width, length=6)  # Налаштовуємо товщину засічок для правої осі
    success_line = ax2.plot(x, success_by_category.values * 100, 'o-',
                            color='gold', linewidth=2, markersize=8)
    ax2.set_ylabel('Відсоток успішності (%)', color='black', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax2.set_ylim(0, 100)

    # Налаштовуємо мітки осі X
    ax1.set_xticks(x)
    ax1.set_xticklabels(category_order, rotation=45)
    ax1.set_title('Розподіл клієнтів та замовлень за кількістю попередніх замовлень', fontsize=14, pad=20)

    # Додаємо підписи значень для клієнтів
    for i, v in enumerate(category_counts.values):
        ax1.text(x[i] - width / 2, v, f'{int(v):,}',
                 ha='center', va='bottom', color='#1f77b4')

    # Додаємо підписи значень для замовлень
    for i, v in enumerate(orders_counts.values):
        ax3.text(x[i] + width / 2, v, f'{int(v):,}',
                 ha='center', va='bottom', color='blue')

    # Додаємо підписи для відсотка успішності
    for i, v in enumerate(success_by_category.values):
        ax2.text(i, v * 100 + 2, f'{v:.1%}',
                 ha='center', va='bottom', color='black')

    # Додаємо легенду
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2 = [Line2D([0], [0], color='gold', marker='o', linewidth=2, markersize=8)]
    lines3, labels3 = ax3.get_legend_handles_labels()

    ax1.legend(lines1 + lines3 + lines2,
               ['Кількість клієнтів', 'Кількість замовлень', 'Відсоток успішності'],
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
    output_path = 'customer_history_graph'
    create_customer_history_chart(csv_path, output_path)
