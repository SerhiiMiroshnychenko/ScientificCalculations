import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import textwrap

# Визначаємо файли та методи
methods = [
    ('Прямий XGBoost', 'xgb_importance_aucpr_curve.csv'),
    ('Зворотній XGBoost', 'backward_xgb_aucpr_curve.csv'),
    ('Прямий Greedy', 'greedy_aucpr_curve.csv'),
    ('Зворотній Greedy', 'backward_aucpr_curve.csv'),
    ('RFE', 'rfe_aucpr_curve.csv'),
    ('Boruta', 'boruta_aucpr_curve.csv'),
]

# Зчитуємо оптимальні набори ознак для кожного методу
feature_sets = {}
for method, fname in methods:
    df = pd.read_csv(fname)
    max_auc = df['aucpr'].max()
    max_rows = df[df['aucpr'] == max_auc]
    best_row = max_rows.loc[max_rows['n_features'].idxmin()]
    features = [f.strip() for f in best_row['features'].split(',')]
    feature_sets[method] = set(features)

# Зчитуємо рейтинг ознак з xgb_importance.csv
xgb_imp = pd.read_csv('xgboost_importance.csv')
feature_order = list(xgb_imp['feature'])
# Всі унікальні ознаки, що входять хоча б в один оптимальний набір
all_features = [f for f in feature_order if any(f in s for s in feature_sets.values())]

# Формуємо матрицю для heatmap: 1 якщо ознака входить у набір для методу, 0 — ні
heatmap_data = np.zeros((len(all_features), len(methods)), dtype=int)
for j, (method, _) in enumerate(methods):
    for i, feature in enumerate(all_features):
        if feature in feature_sets[method]:
            heatmap_data[i, j] = 1

# Створюємо DataFrame для heatmap
heatmap_df = pd.DataFrame(heatmap_data, index=all_features, columns=[m[0] for m in methods])

# Кольорова карта: білий (0) — не обрано, світлозелений (1) — обрано
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#ffffff', '#7BA05B'])

# Формуємо підписи методів з переносом слів
maxlen = 12
wrapped_methods = ['\n'.join(textwrap.wrap(m[0], maxlen)) for m in methods]

plt.figure(figsize=(10, max(6, len(all_features)*0.35)))
sns.heatmap(heatmap_df, annot=False, cmap=cmap, cbar=False, linewidths=0.5, linecolor='gray', square=False, xticklabels=wrapped_methods, yticklabels=all_features)
plt.xlabel('Метод')
plt.ylabel('Ознака')
plt.title('Теплова карта: включення ознак у найкращі набори за методами')
plt.tight_layout()
plt.savefig('feature_selection_heatmap.png', dpi=300)
plt.close()

# Гістограма: скільки методів обрали кожну ознаку
feature_counts = heatmap_df.sum(axis=1)
feature_counts_sorted = feature_counts.sort_values(ascending=False)

plt.figure(figsize=(8, max(6, len(all_features)*0.35)))
plt.barh(feature_counts_sorted.index, feature_counts_sorted.values, color='#0077b6')
plt.xlabel('Кількість методів, що обрали ознаку')
plt.ylabel('Ознака')
plt.title('Частота включення ознак у найкращі набори (по методах)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_selection_feature_count_hist.png', dpi=300)
plt.close()

print('Збережено: feature_selection_heatmap.png та feature_selection_feature_count_hist.png') 