import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AUC_MIN = 0.9778

# Оновлені імена файлів, порядок, назви та кольори
files = [
    ('Прямий XGBoost', 'xgb_importance_aucpr_curve.csv', '#00bfff'),
    ('Зворотній XGBoost', 'backward_xgb_aucpr_curve.csv', 'blue'),
    ('Прямий Greedy', 'greedy_aucpr_curve.csv', 'fuchsia'),
    ('Зворотній Greedy', 'backward_aucpr_curve.csv', 'purple'),
    ('RFE', 'rfe_aucpr_curve.csv', 'green'),
    ('Boruta', 'boruta_aucpr_curve.csv', 'red'),
]

plt.figure(figsize=(8, 5))
for label, fname, color in files:
    df = pd.read_csv(fname)
    df = df[(df['n_features'] >= 12) & (df['n_features'] <= 24) & (df['aucpr'] >= AUC_MIN)]
    if len(df) == 0:
        continue
    # Малюємо лінію та всі точки
    plt.plot(df['n_features'], df['aucpr'], marker='o', color=color, label=label)
    # Знаходимо точку з максимальним aucpr (якщо кілька — з найменшою кількістю ознак)
    max_auc = df['aucpr'].max()
    max_rows = df[df['aucpr'] == max_auc]
    best_row = max_rows.loc[max_rows['n_features'].idxmin()]
    # Підписуємо тільки максимальну точку
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
plt.title('Оптимальна кількість ознак за методами')
plt.grid(True, zorder=0)
plt.legend()
plt.tight_layout()
plt.ylim(AUC_MIN, plt.ylim()[1] + (plt.ylim()[1] - AUC_MIN) * 0.08)
plt.savefig('compare_aucpr_curve_12_24_from_results.png')
print('Графік збережено у compare_aucpr_curve_12_24_from_results.png') 