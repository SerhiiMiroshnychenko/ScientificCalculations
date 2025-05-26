# -*- coding: utf-8 -*-
"""
Візуалізація теплової карти рангів значущості ознак за різними методами
Фінальна версія з точним відтворенням бажаного вигляду
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

# Налаштування наукового стилю
plt.style.use(['science'])

# Шлях до файлу з результатами аналізу значущості ознак
file_path = 'feature_importance_summary.csv'

# Завантаження даних
print("Завантаження даних...")
data = pd.read_csv(file_path)
print(f"Завантажено набір даних розміром {data.shape}")

# Отримуємо колонки з рангами, ВИКЛЮЧАЮЧИ 'Середній_ранг'
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

# Додаємо ранги для кожного методу
for col in rank_columns:
    method_name = col.replace('_ранг', '')
    heatmap_data[method_name] = data[col]

# Додаємо оригінальні назви ознак
heatmap_data['Feature'] = data['Ознака']

# Додаємо середній ранг окремо (англійською)
heatmap_data['Average Rank'] = data['Середній_ранг']

# Сортуємо за середнім рангом
heatmap_data = heatmap_data.sort_values('Average Rank')

# Додаємо колонку з порядковими номерами (загальний ранг)
heatmap_data['Overall Rank'] = range(1, len(heatmap_data) + 1)

# Встановлюємо індекс для теплової карти
heatmap_data = heatmap_data.set_index('Feature')

# Вибираємо колонки з методами для теплової карти
heatmap_cols = method_names + ['Average Rank', 'Overall Rank']
plot_data = heatmap_data[heatmap_cols]

# Перейменовуємо колонки методів на повні назви
plot_data = plot_data.rename(columns={method: method_full_names.get(method, method) for method in method_names})

# Налаштування стилю і шрифтів
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# Налаштування розміру фігури
n_rows = len(plot_data)
figure_height = max(10, n_rows * 0.4)
figure_width = 16

# Створюємо фігуру
fig, ax = plt.subplots(figsize=(figure_width, figure_height))

# Створюємо теплову карту
heatmap = sns.heatmap(
    plot_data, 
    annot=True,
    cmap="RdYlGn_r", 
    fmt=".1f", 
    linewidths=.5,
    cbar_kws={'label': 'Rank (lower = more important)'},
    ax=ax
)

# Видаляємо всі рамки
for spine in ax.spines.values():
    spine.set_visible(False)

# Повністю видаляємо засічки
ax.tick_params(
    axis='both',          # застосовуємо до обох осей
    which='both',         # основні та допоміжні засічки
    bottom=False,         # прибираємо засічки внизу
    top=False,            # прибираємо засічки зверху
    left=False,           # прибираємо засічки ліворуч
    right=False,          # прибираємо засічки праворуч
    labelbottom=True,     # залишаємо підписи внизу
    labeltop=False,       # прибираємо підписи зверху
    labelleft=True,       # залишаємо підписи ліворуч
    labelright=False,     # прибираємо підписи праворуч
    length=0,             # встановлюємо довжину засічок в 0
    width=0               # встановлюємо ширину засічок в 0
)

# Прибираємо рамку і засічки зі шкали кольорів, залишаючи тільки підписи
cbar = heatmap.collections[0].colorbar
cbar.outline.set_visible(False)  # приховуємо рамку шкали

# Створюємо користувацьку шкалу без засічок
cbar.ax.tick_params(size=0, length=0, width=0, direction='out')

# Залишаємо тільки позначки цифр, видаляючи всі засічки
cbar.ax.yaxis.set_ticks_position('none')

# Налаштовуємо вісі
ax.set_ylabel('')
ax.set_xlabel('')

# Обертаємо підписи на осі X для кращої читабельності
plt.xticks(rotation=0, ha='center')  # rotation=0 для горизонтальних підписів

# Встановлюємо заголовок максимально близько до теплової карти
ax.set_title('Feature Importance Rank Heatmap by Different Methods',
            fontsize=16, pad=15)  # використовуємо заголовок осі з мінімальним відступом

# Налаштовуємо макет без додаткового простору вгорі
plt.tight_layout()

# Зберігаємо теплову карту
output_file_png = 'feature_importance_heatmap_final.png'
output_file_pdf = 'feature_importance_heatmap_final.pdf'

plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
plt.savefig(output_file_pdf, bbox_inches='tight')
plt.close()

print(f"Теплову карту збережено у файли:")
print(f"- {output_file_png}")
print(f"- {output_file_pdf}")
print("Візуалізація завершена.")
