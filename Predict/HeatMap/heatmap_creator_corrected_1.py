#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для створення теплової карти рангів ознак
за різними алгоритмами оцінки важливості.
Використовує правильні топ-15 ознак за даними з файлу Важливість.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Шлях до файлу з даними
features_file_path = "Рейтинг-місць-ознак-1.md"
output_path = "heatmap_ranks_final_1.png"

# Функція для зчитування даних з MD-файлу
def read_markdown_table(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Знаходимо рядок з заголовками
    header_row = None
    for i, line in enumerate(lines):
        if '| № | Технічна назва ознаки |' in line:
            header_row = i
            break
    
    if header_row is None:
        raise ValueError("Не вдалося знайти заголовок таблиці у файлі.")
    
    # Отримуємо заголовки стовпців
    headers = [col.strip() for col in lines[header_row].strip().strip('|').split('|')]
    
    # Пропускаємо рядок з розділювачами (---|---)
    data_start = header_row + 2
    
    # Збираємо дані з таблиці
    data = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or not line.startswith('|'):
            break
        row_data = [cell.strip() for cell in line.strip('|').split('|')]
        if len(row_data) == len(headers):
            data.append(row_data)
    
    # Створюємо DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    # Перетворюємо числові стовпці в числові типи даних
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col])
    
    return df

# Функція для отримання правильних назв ознак українською мовою
def get_feature_names_ukr():
    # Топ-15 ознак за важливістю з файлу Важливість.txt
    features_ukr = {
        "order_messages": "Кількість повідомлень",
        "partner_success_rate": "Сердній % успішних замовлень клієнта",
        "order_changes": "Кількість змін в замовлені",
        "order_amount": "Сума замовлення",
        "partner_total_messages": "Загальна кількість повідомлень клієнта",
        "partner_total_orders": "Кількість замовлень клієнта",
        "partner_order_age_days": "Термін співпраці",
        "order_lines_count": "Кількість позицій в замовленні",
        "partner_avg_changes": "Середня кількість змін в замовленях клієнта",
        "partner_success_avg_messages": "Середня кількість повідомлень успішних замовлень",
        "create_date_months": "Місяці від найранішої дати",
        "partner_success_avg_amount": "Середня сума успішних замовлень клієнта",
        "partner_success_avg_changes": "Середня кількість змін в успішних замовленях клієнта",
        "partner_fail_avg_messages": "Середня кількість повідомлень невдалих замовлень",
        "partner_fail_avg_changes": "Середня кількість змін в невдалих замовленях клієнта"
    }
    return features_ukr

# Функція для отримання топ-15 ознак у правильному порядку важливості
def get_top_15_features():
    return [
        "order_messages",
        "partner_success_rate", 
        "order_changes",
        "order_amount",
        "partner_total_messages",
        "partner_total_orders",
        "partner_order_age_days",
        "order_lines_count",
        "partner_avg_changes",
        "partner_success_avg_messages",
        "create_date_months",
        "partner_success_avg_amount", 
        "partner_success_avg_changes",
        "partner_fail_avg_messages",
        "partner_fail_avg_changes"
    ]

# Функція для побудови теплової карти
def plot_heatmap(rankings_df, feature_order, feature_names_ukr, save_path=None):
    """
    Будує теплову карту рангів важливості ознак за різними метриками
    
    Args:
        rankings_df (pd.DataFrame): DataFrame з результатами рейтингу
        feature_order (list): Список ознак у правильному порядку
        feature_names_ukr (dict): Словник з українськими назвами ознак
        save_path (str): Шлях для збереження графіка, якщо None - не зберігати
    """
    # Отримуємо назви методів (стовпців)
    method_names = rankings_df.columns[2:].tolist()
    
    # Створюємо новий DataFrame з відсортованими ознаками
    sorted_df = pd.DataFrame()
    
    for feature in feature_order:
        # Знаходимо рядки, що відповідають ознаці
        row = rankings_df[rankings_df['Технічна назва ознаки'] == feature]
        if not row.empty:
            sorted_df = pd.concat([sorted_df, row])
    
    # Створюємо матрицю рангів для теплової карти
    feature_data = []
    feature_names = []
    
    for _, row in sorted_df.iterrows():
        feature_tech_name = row['Технічна назва ознаки']
        if feature_tech_name in feature_names_ukr:
            feature_names.append(feature_names_ukr[feature_tech_name])
            feature_data.append(row[method_names].values)
    
    # Створюємо DataFrame для теплової карти
    heatmap_df = pd.DataFrame(feature_data, index=feature_names, columns=method_names)
    
    # Визначаємо максимальний ранг для шкали кольорів
    max_rank = int(heatmap_df.max().max())
    
    # Створюємо теплову карту
    plt.figure(figsize=(14, 10))
    
    # Використовуємо інвертовану кольорову палітру, щоб менші ранги (1, 2, 3...) були темнішими
    cmap = plt.cm.get_cmap('YlGnBu_r')
    
    # Відображаємо теплову карту з рангами
    heatmap = sns.heatmap(heatmap_df, annot=True, cmap=cmap, fmt='.0f', linewidths=.5,
                         vmin=1, vmax=max_rank)
    
    # Налаштовуємо colorbar: 1 вгорі (темніший), max_rank внизу (світліший)
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([1, max_rank])
    cbar.set_ticklabels(['1 (найважливіша)', f'{int(max_rank)} (найменш важлива)'])
    
    # Обертаємо шкалу, щоб 1 був зверху, а max_rank знизу
    cbar.ax.invert_yaxis()
    
    # Додаємо інформацію про кількість відображених ознак
    title = f'Ранги ознак за різними метриками (топ-{len(feature_names)} з 24 ознак)'
    plt.title(title, fontsize=16, pad=20)
    
    # Налаштування осей
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Теплову карту збережено: {save_path}")
    
    return plt.gcf()

# Запускаємо скрипт
if __name__ == "__main__":
    # Налаштування шрифтів для коректного відображення кирилиці
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    try:
        # Зчитуємо дані з файлу рейтингу місць
        df = read_markdown_table(features_file_path)
        print(f"Дані успішно зчитано з файлу: {features_file_path}")
        
        # Отримуємо порядок ознак і їх українські назви
        feature_order = get_top_15_features()
        feature_names_ukr = get_feature_names_ukr()
        
        # Будуємо і зберігаємо теплову карту
        fig = plot_heatmap(df, feature_order, feature_names_ukr, save_path=output_path)
        
        # Показуємо карту
        plt.show()
        print(f"Відображено топ-15 ознак у правильному порядку!")
        
    except Exception as e:
        print(f"Помилка: {e}")
