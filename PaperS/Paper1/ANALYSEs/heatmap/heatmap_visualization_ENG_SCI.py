# -*- coding: utf-8 -*-
"""
Візуалізація теплової карти рангів значущості ознак за різними методами
Оновлена версія з використанням англійської мови
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

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
    plt.rcParams['font.size'] = 12


# Встановлюємо шрифти
setup_fonts()

# Створюємо фігуру з відповідним розміром
plt.figure(figsize=(16, max(10, len(plot_data) * 0.4)))  # Збільшуємо ширину для додаткової колонки

# Створюємо анотацію для заголовка (англійською)
plt.annotate('Feature Importance Rank Heatmap by Different Methods',
             xy=(0.5, 0.93), xycoords='figure fraction',
             fontsize=16, ha='center')

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

# Створюємо теплову карту з нашими власними анотаціями
heatmap = sns.heatmap(
    plot_data,  # оригінальні дані для кольорів
    annot=annot_texts.values,  # вже відформатовані тексти
    cmap="RdYlGn_r",
    fmt="",  # порожній формат, бо ми вже відформатували тексти
    linewidths=.5,
    annot_kws={"size": 10},
    cbar_kws={'label': 'Rank (lower = more important)'}
)

# Налаштовуємо вісі
plt.ylabel('')  # Не потрібно підписувати вісь Y, там вже є назви ознак
plt.xlabel('')

# Встановлюємо горизонтальні підписи на осі X для кращої читабельності
plt.xticks(rotation=0, ha='center')  # rotation=0 для горизонтальних підписів

# Налаштовуємо макет
plt.tight_layout()

# Зберігаємо теплову карту у кількох форматах
output_file_png = 'feature_importance_heatmap_ENG_SCI.png'
output_file_svg = 'feature_importance_heatmap_ENG_SCI.svg'

plt.savefig(output_file_png, dpi=900, bbox_inches='tight')
plt.savefig(f"{output_file_svg}", format='svg',
                bbox_inches='tight')
plt.close()

print(f"Теплову карту збережено у файл {output_file_png}")
print("Візуалізація завершена.")
