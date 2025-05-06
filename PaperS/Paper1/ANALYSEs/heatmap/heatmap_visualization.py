# -*- coding: utf-8 -*-
"""
Візуалізація теплової карти рангів значущості ознак за різними методами
Виправлена версія без дублювання колонки "Середній ранг"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# Шлях до файлу з результатами аналізу значущості ознак
file_path = 'feature_importance_summary.csv'

# Завантаження даних
print("Завантаження даних...")
data = pd.read_csv(file_path)
print(f"Завантажено набір даних розміром {data.shape}")

# Словник для перекладу назв ознак на українську
translation_dict = {
    'order_messages': 'Кількість повідомлень',
    'partner_success_rate': 'Середній % успішних замовлень клієнта',
    'order_changes': 'Кількість змін в замовленні',
    'order_amount': 'Сума замовлення',
    'partner_total_messages': 'Загальна кількість повідомлень клієнта',
    'partner_total_orders': 'Кількість замовлень клієнта',
    'partner_order_age_days': 'Термін співпраці',
    'order_lines_count': 'Кількість позицій в замовленні',
    'partner_avg_changes': 'Середня кількість змін в замовленнях клієнта',
    'partner_success_avg_messages': 'Середня кількість повідомлень успішних замовлень',
    'create_date_months': 'Місяці від найранішої дати',
    'partner_success_avg_amount': 'Середня сума успішних замовлень клієнта',
    'partner_success_avg_changes': 'Середня кількість змін в успішних замовленнях клієнта',
    'partner_fail_avg_messages': 'Середня кількість повідомлень невдалих замовлень',
    'partner_fail_avg_changes': 'Середня кількість змін в невдалих замовленнях клієнта',
    'partner_avg_amount': 'Середня сума замовлень клієнта',
    'hour_of_day': 'Година доби',
    'salesperson': 'Менеджер',
    'partner_fail_avg_amount': 'Середня сума невдалих замовлень клієнта',
    'source': 'Джерело замовлення',
    'month': 'Місяць',
    'quarter': 'Квартал',
    'day_of_week': 'День тижня',
    'discount_total': 'Загальна знижка'
}

# Отримуємо колонки з рангами, ВИКЛЮЧАЮЧИ 'Середній_ранг'
# Це ключове виправлення для запобігання дублюванню
rank_columns = [col for col in data.columns if col.endswith('_ранг') and col != 'Середній_ранг']

# Словник повних англійських назв методів
method_full_names = {
    'AUC': 'Area Under Curve',
    'MI': 'Mutual Information',
    'dCor': 'Distance Correlation',
    'LogReg': 'Logistic Regression',
    'DecTree': 'Decision Tree'
}

# Отримуємо короткі назви методів
method_names = [col.replace('_ранг', '') for col in rank_columns]

# Створюємо DataFrame для теплової карти
heatmap_data = pd.DataFrame()

# Додаємо ранги для кожного методу (крім середнього рангу)
for col in rank_columns:
    method_name = col.replace('_ранг', '')
    heatmap_data[method_name] = data[col]

# Додаємо назви ознак (в т.ч. українською)
heatmap_data['Ознака'] = data['Ознака']
heatmap_data['Ознака_укр'] = heatmap_data['Ознака'].map(lambda x: translation_dict.get(x, x))

# Додаємо середній ранг окремо
heatmap_data['Середній ранг'] = data['Середній_ранг']

# Сортуємо за середнім рангом
heatmap_data = heatmap_data.sort_values('Середній ранг')

# Встановлюємо індекс для теплової карти
heatmap_data = heatmap_data.set_index('Ознака_укр')

# Вибираємо колонки з методами для теплової карти
heatmap_cols = method_names + ['Середній ранг']
plot_data = heatmap_data[heatmap_cols]

# Перейменовуємо ЛИШЕ колонки методів на повні англійські назви, залишаючи "Середній ранг" як є
plot_data = plot_data.rename(columns={method: method_full_names.get(method, method) for method in method_names})


# Функція для встановлення українських шрифтів
def setup_ukrainian_fonts():
    # Спробуємо знайти шрифт з підтримкою кирилиці
    # Список можливих шрифтів
    cyrillic_fonts = ['Arial', 'Verdana', 'Times New Roman', 'DejaVu Sans', 'Liberation Sans']

    # Перевіряємо наявність шрифтів
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # Шукаємо перший доступний шрифт з підтримкою кирилиці
    font_to_use = None
    for font in cyrillic_fonts:
        if font in available_fonts:
            font_to_use = font
            break

    # Якщо знайдено відповідний шрифт, встановлюємо його
    if font_to_use:
        plt.rcParams['font.family'] = font_to_use
    else:
        print("Не знайдено шрифт з підтримкою кирилиці. Можливі проблеми з відображенням українських символів.")

    # Налаштування для відображення кирилиці
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans']
    plt.rcParams['font.size'] = 10


# Встановлюємо українські шрифти
setup_ukrainian_fonts()

# Створюємо фігуру з відповідним розміром
plt.figure(figsize=(14, max(10, len(plot_data) * 0.4)))  # Збільшуємо висоту для кращої читабельності

# Створюємо анотацію для заголовка
plt.annotate('Теплова карта рангів значущості ознак за різними методами',
             xy=(0.5, 1.02), xycoords='figure fraction',
             fontsize=16, ha='center')

# Створюємо теплову карту
# Змінюємо колірну палітру на зелено-жовто-червону для кращого відображення рангів (зелений - найважливіші)
heatmap = sns.heatmap(plot_data, annot=True, cmap="RdYlGn_r", fmt=".1f", linewidths=.5,
                      cbar_kws={'label': 'Ранг (нижче = важливіше)'})

# Налаштовуємо вісі
plt.ylabel('')  # Не потрібно підписувати вісь Y, оскільки там вже є українські назви ознак
plt.xlabel('Метод оцінки')

# Обертаємо підписи на осі X для кращої читабельності
plt.xticks(rotation=45, ha='right')

# Додаємо підпис з поясненням
plt.figtext(0.5, -0.05,
            "Значення в комірках - ранги ознак за кожним методом (менше = важливіше).\n"
            "Кольорове кодування: зелений - найважливіші ознаки, червоний - найменш важливі.",
            ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.9, "pad": 5})

# Налаштовуємо макет
plt.tight_layout()

# Зберігаємо теплову карту
output_file = 'feature_importance_heatmap.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Теплову карту збережено у файл {output_file}")
print("Візуалізація завершена.")
