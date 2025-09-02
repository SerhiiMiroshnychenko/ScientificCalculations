# -*- coding: utf-8 -*-
"""
Візуалізація теплової карти рангів значущості ознак за різними методами
Оновлена версія з використанням англійської мови
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np

# Шлях до файлу з результатами аналізу значущості ознак
file_path = 'feature_importance_summary.csv'

# Завантаження даних
print("Завантаження даних...")
data = pd.read_csv(file_path)
print(f"Завантажено набір даних розміром {data.shape}")

# Словник перекладів назв ознак на українську (тепер не використовується)
# translation_dict = {...}

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

# Додаємо оригінальні назви ознак
heatmap_data['Feature'] = data['Ознака']

# Додаємо середній ранг окремо (англійською)
heatmap_data['Average Rank'] = data['Середній_ранг']

# Сортуємо за середнім рангом
heatmap_data = heatmap_data.sort_values('Average Rank')

# Додаємо колонку з порядковими номерами (загальний ранг)
heatmap_data['Overall Rank'] = range(1, len(heatmap_data) + 1)

# Встановлюємо індекс для теплової карти (використовуємо оригінальні назви ознак)
heatmap_data = heatmap_data.set_index('Feature')

# Вибираємо колонки з методами для теплової карти
heatmap_cols = method_names + ['Average Rank', 'Overall Rank']
plot_data = heatmap_data[heatmap_cols]

# Перейменовуємо ЛИШЕ колонки методів на повні англійські назви, залишаючи "Середній ранг" як є
plot_data = plot_data.rename(columns={method: method_full_names.get(method, method) for method in method_names})


# Функція для встановлення шрифтів у науковому стилі
def setup_fonts():
    # Використовуємо шрифт Times New Roman розміром 8 пунктів
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 22


# Встановлюємо шрифти
setup_fonts()

# Створюємо фігуру з відповідним розміром
plt.figure(figsize=(27, max(10, len(plot_data) * 0.4)))  # Збільшуємо ширину для додаткової колонки

# Створюємо матрицю з вже відформатованими значеннями
# Це дозволить нам контролювати форматування кожного елемента
annot_data = plot_data.copy()

# Створюємо матрицю рядків для анотацій
annot_texts = pd.DataFrame(index=annot_data.index, columns=annot_data.columns)

# Заповнюємо матрицю відформатованими текстами
for col in annot_data.columns:
    for idx in annot_data.index:
        value = annot_data.loc[idx, col]
        if col == 'Average Rank':
            # Формат з одним знаком після коми для Average Rank
            annot_texts.loc[idx, col] = f"{value:.1f}"
        else:
            # Цілі числа для всіх інших колонок
            annot_texts.loc[idx, col] = f"{int(value)}"

# Імпортуємо необхідні модулі для керування кольоровою шкалою
from matplotlib.colors import ListedColormap

# Створюємо теплову карту з нашими власними анотаціями
# Визначаємо максимальне і мінімальне значення для масштабування кольорів
vmin = plot_data.min().min()  # Мінімальне значення рангу
vmax = plot_data.max().max()  # Максимальне значення рангу

# Створюємо звичайну кольорову гаму та інвертовану для colorbar
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Отримуємо кольори з оригінальної кольорової гами RdYlGn_r
cmap_orig = plt.cm.get_cmap('RdYlGn_r')
colors = cmap_orig(np.linspace(0, 1, 256))

# Створюємо нову кольорову гаму з інвертованими кольорами
inverted_colors = np.flip(colors, axis=0)  # Інвертуємо порядок кольорів
inverted_cmap = ListedColormap(inverted_colors)

# Створюємо участки шкали (5 основних точок)
min_val = 1
max_val = 24
tick_positions = np.linspace(min_val, max_val, 5)  # [1, 7, 13, 19, 24]
tick_labels = [f"{int(val)}" for val in tick_positions]

heatmap = sns.heatmap(
    plot_data,  # оригінальні дані для кольорів
    annot=annot_texts.values,  # вже відформатовані тексти
    cmap="RdYlGn_r",  # Використовуємо оригінальну кольорову схему для даних
    fmt="",  # порожній формат, бо ми вже відформатували тексти
    linewidths=.5,
    annot_kws={"size": 22},
    cbar_kws={
        'label': 'Rank (lower = more important)',
        'ticks': tick_positions  # Використовуємо визначені точки для шкали
    }
)

# Отримуємо доступ до colorbar і інвертуємо його
cbar = heatmap.collections[0].colorbar

# Змінюємо напрямок colorbar на протилежний
cbar.ax.invert_yaxis()

# Налаштовуємо позначки на colorbar
cbar.set_ticks(tick_positions)
cbar.set_ticklabels(tick_labels)

# Налаштовуємо вісі
plt.ylabel('')  # Не потрібно підписувати вісь Y, там вже є назви ознак
plt.xlabel('')

# Встановлюємо горизонтальні підписи на осі X для кращої читабельності
plt.xticks(rotation=0, ha='center')  # rotation=0 для горизонтальних підписів

# Налаштовуємо макет
plt.tight_layout()

# Зберігаємо теплову карту у кількох форматах
output_file_png = 'feature_importance_heatmap_R.png'
output_file_svg = 'feature_importance_heatmap_R.svg'

plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
plt.savefig(f"{output_file_svg}", format='svg',
                bbox_inches='tight')
plt.close()

print(f"Теплову карту збережено у файл {output_file_png}")
print("Візуалізація завершена.")
