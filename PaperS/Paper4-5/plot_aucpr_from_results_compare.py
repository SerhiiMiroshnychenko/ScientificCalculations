import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# # Налаштування шрифту Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 14

AUC_MIN = 0.9763

# Файли та методи з різними маркерами
methods = [
    ('Прямий XGBoost', 'xgb_importance_aucpr_curve.csv', '#00bfff', 'o'),
    ('Зворотній XGBoost', 'backward_xgb_aucpr_curve.csv', 'blue', 's'),
    ('Прямий Greedy', 'greedy_aucpr_curve.csv', 'fuchsia', '^'),
    ('Зворотній Greedy', 'backward_aucpr_curve.csv', 'purple', 'D'),
    ('RFE', 'rfe_aucpr_curve.csv', 'green', 'v'),
    ('Boruta', 'boruta_aucpr_curve.csv', 'red', 'p'),
]

# Директрії з результатами
DATASETS = [
    ('Датасет 1', 'dataset1', '-'),
    ('Датасет 2', 'dataset2', '--'),
]

plt.figure(figsize=(10, 6))
for ds_label, ds_dir, linestyle in DATASETS:
    for method_label, fname, color, marker in methods:
        path = os.path.join(ds_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[(df['n_features'] >= 12) & (df['n_features'] <= 24) & (df['aucpr'] >= AUC_MIN)]
        if len(df) == 0:
            continue
        plt.plot(df['n_features'], df['aucpr'], marker=marker, color=color, linestyle=linestyle, label=f'{method_label} ({ds_label})')
        # Підписуємо тільки максимальну точку для кожної лінії
        max_auc = df['aucpr'].max()
        max_rows = df[df['aucpr'] == max_auc]
        best_row = max_rows.loc[max_rows['n_features'].idxmin()]
        plt.annotate(
            f"{best_row['aucpr']:.5f}",
            (best_row['n_features'], best_row['aucpr']),
            textcoords="offset points",
            xytext=(0, 7),
            ha='center',
            fontsize=9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1)
        )

plt.xlabel('Кількість ознак')
plt.ylabel('AUC-PR')
plt.grid(True, zorder=0)
plt.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.ylim(AUC_MIN, plt.ylim()[1] + (plt.ylim()[1] - AUC_MIN) * 0.08)
plt.savefig('compare_aucpr_curve_12_24_from_results_compare.png')
print('Графік збережено у compare_aucpr_curve_12_24_from_results_compare.png') 