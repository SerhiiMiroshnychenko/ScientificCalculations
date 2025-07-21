import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import textwrap
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

methods = [
    ('Прямий XGBoost', 'xgb_importance_aucpr_curve.csv'),
    ('Зворотній XGBoost', 'backward_xgb_aucpr_curve.csv'),
    ('Прямий Greedy', 'greedy_aucpr_curve.csv'),
    ('Зворотній Greedy', 'backward_aucpr_curve.csv'),
    ('RFE', 'rfe_aucpr_curve.csv'),
    ('Boruta', 'boruta_aucpr_curve.csv'),
]
DATASETS = [
    ('dataset1', 'Блакитний'),
    ('dataset2', 'Помаранчевий'),
]

# 1. Збираємо всі ознаки, які входять у найкращі набори хоча б в одному датасеті
feature_sets = {ds: {} for ds, _ in DATASETS}
all_features = set()
for ds, _ in DATASETS:
    for method, fname in methods:
        path = os.path.join(ds, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        max_auc = df['aucpr'].max()
        max_rows = df[df['aucpr'] == max_auc]
        best_row = max_rows.loc[max_rows['n_features'].idxmin()]
        features = [f.strip() for f in best_row['features'].split(',')]
        feature_sets[ds][method] = set(features)
        all_features.update(features)
all_features = list(all_features)

# Додаємо дефолтну важливість ознак з xgboost_importance.csv
xgb_imp = pd.read_csv('xgboost_importance.csv')
imp_dict = dict(zip(xgb_imp['feature'], xgb_imp['importance']))

# 2. Формуємо матрицю для heatmap: 0 — жоден, 1 — тільки dataset1, 2 — тільки dataset2, 3 — обидва
heatmap_data = np.zeros((len(all_features), len(methods)), dtype=int)
for j, (method, _) in enumerate(methods):
    for i, feature in enumerate(all_features):
        in1 = feature in feature_sets['dataset1'].get(method, set())
        in2 = feature in feature_sets['dataset2'].get(method, set())
        if in1 and in2:
            heatmap_data[i, j] = 3
        elif in1:
            heatmap_data[i, j] = 1
        elif in2:
            heatmap_data[i, j] = 2
        else:
            heatmap_data[i, j] = 0

# 3. Сортуємо ознаки: спочатку за сумою включень, далі за дефолтною важливістю (спадання)
counts1 = np.array([[feature in feature_sets['dataset1'].get(method, set()) for method, _ in methods] for feature in all_features]).sum(axis=1)
counts2 = np.array([[feature in feature_sets['dataset2'].get(method, set()) for method, _ in methods] for feature in all_features]).sum(axis=1)
sum_counts = counts1 + counts2
imp_arr = np.array([imp_dict.get(f, 0) for f in all_features])
sort_idx = np.lexsort((-imp_arr, -sum_counts))
all_features_sorted = [all_features[i] for i in sort_idx]
heatmap_data_sorted = heatmap_data[sort_idx, :]
counts1_sorted = counts1[sort_idx]
counts2_sorted = counts2[sort_idx]

# 4. Малюємо heatmap
cmap = ListedColormap(['#ffffff', '#00bfff', '#ff8800', '#3cb371'])  # білий, блакитний, помаранчевий, зелений
bounds = [0, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)
maxlen = 12
wrapped_methods = ['\n'.join(textwrap.wrap(m[0], maxlen)) for m in methods]
plt.figure(figsize=(10, max(6, len(all_features_sorted)*0.35) + 1.5))
ax = plt.gca()
sns.heatmap(heatmap_data_sorted, annot=False, cmap=cmap, norm=norm, cbar=False, linewidths=0.5, linecolor='gray', square=False, xticklabels=wrapped_methods, yticklabels=all_features_sorted, ax=ax)
plt.xlabel('Метод')
plt.ylabel('Ознака')
plt.title('Теплова карта: включення ознак у найкращі набори')
plt.tight_layout(rect=[0,0.08,1,1])

# Додаємо кастомну легенду під графіком
legend_elements = [
    Patch(facecolor='#ffffff', edgecolor='black', label='Не входить'),
    Patch(facecolor='#00bfff', edgecolor='black', label='Тільки датасет 1'),
    Patch(facecolor='#ff8800', edgecolor='black', label='Тільки датасет 2'),
    Patch(facecolor='#3cb371', edgecolor='black', label='Обидва датасети'),
]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=True, fontsize=11)
plt.savefig('feature_selection_heatmap_compare.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Малюємо гістограму
plt.figure(figsize=(8, max(6, len(all_features_sorted)*0.35)))
y = np.arange(len(all_features_sorted))
plt.barh(y, counts1_sorted, height=0.8, color='#00bfff', alpha=0.6, label='Датасет 1', edgecolor='black')
plt.barh(y, counts2_sorted, height=0.4, color='#ff8800', alpha=0.6, label='Датасет 2', edgecolor='black')
plt.yticks(y, all_features_sorted)
plt.xlabel('Кількість методів, що обрали ознаку')
plt.ylabel('Ознака')
plt.title('Частота включення ознак у найкращі набори (по методах)')
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_selection_feature_count_hist_compare.png', dpi=300)
plt.close()

print('Збережено: feature_selection_heatmap_compare.png та feature_selection_feature_count_hist_compare.png')