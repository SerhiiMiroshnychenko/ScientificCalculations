# -*- coding: utf-8 -*-
"""
Нові візуалізації важливості ознак у стилі прикладу (heatmap + bar chart).
- Кольорова гама теплокарти: YlGnBu_r (темно-синій = кращий ранг).
- Підписи у візуалізаціях — англійською.
- Вхід: CSV 'feature_importance_summary.csv' у поточній папці (можна змінити шлях нижче).
Очікувані колонки у CSV:
  - 'Ознака' — назва ознаки
  - 'Середній_ранг' — середній ранг (менше = краще)
  - Колонки з рангами для методів, що закінчуються на '_ранг' (крім 'Середній_ранг')
Вихід:
  - feature_importance_heatmap_blue.(png|svg)
  - feature_importance_overall_blue.(png|svg)
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ---------------------------- Налаштування ---------------------------- #
CSV_PATH = 'feature_importance_summary.csv'  # шлях до даних
TOP_N = 15                                   # скільки ознак показувати
FONT_FAMILY = 'Times New Roman'
BASE_FIG_DPI = 300
TOTAL_FEATURES = 24                           # фіксована кількість ознак для барчарту та заголовку

# Мапінг коротких назв методів у повні англійські назви (якщо присутні)
METHOD_FULL_NAMES = {
    'AUC': 'Area Under Curve',
    'MI': 'Mutual Information',
    'dCor': 'Distance Correlation',
    'LogReg': 'Logistic Regression',
    'DecTree': 'Decision Tree',
}

# ---------------------------- Допоміжні функції ---------------------------- #

def setup_fonts(base_size: int = 18) -> None:
    """Встановити науковий шрифт і розмір."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [FONT_FAMILY]
    plt.rcParams['font.size'] = base_size


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {'Ознака', 'Середній_ранг'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    return df


def prepare_heatmap_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """Побудувати таблицю для теплокарти і повернути (plot_df, method_names, total_count)."""
    rank_cols = [c for c in df.columns if c.endswith('_ранг') and c != 'Середній_ранг']
    method_names = [c.replace('_ранг', '') for c in rank_cols]

    heatmap_df = pd.DataFrame({mn: df[f"{mn}_ранг"] for mn in method_names})
    heatmap_df['Feature'] = df['Ознака']
    heatmap_df['Average Rank'] = df['Середній_ранг']
    heatmap_df = heatmap_df.sort_values('Average Rank').reset_index(drop=True)
    # Як у R-версії: після сортування формуємо послідовний Overall Rank 1..N
    heatmap_df['Overall Rank'] = np.arange(1, len(heatmap_df) + 1)

    # НЕ обрізаємо до Top-N — показуємо всі фічі на теплокарті
    top_df = heatmap_df.copy()
    top_df = top_df.set_index('Feature')

    # Перейменування методів у повні англ. назви (якщо відомі)
    rename_map = {m: METHOD_FULL_NAMES.get(m, m) for m in method_names}
    top_df = top_df.rename(columns=rename_map)

    # Відсортуємо ознаки за середнім рангом (індекс — ознаки)
    total_count = heatmap_df.shape[0]
    return top_df, list(rename_map.values()), total_count


def prepare_overall_full(df: pd.DataFrame) -> pd.DataFrame:
    """Готує фрейм для загального рейтингу по ВСІХ ознаках (без обрізання Top-N).

    Повертає DataFrame з індексом Feature та колонкою 'Average Rank', відсортований зростанням.
    """
    full_df = df[['Ознака', 'Середній_ранг']].copy()
    full_df = full_df.rename(columns={'Ознака': 'Feature', 'Середній_ранг': 'Average Rank'})
    full_df = full_df.sort_values('Average Rank').set_index('Feature')
    return full_df


def plot_heatmap(plot_df: pd.DataFrame, method_cols: list[str], total_count: int) -> None:
    """Побудувати теплокарту рангів за методами (включно з Average Rank та Overall Rank) з логікою R-скрипта.
    Міняємо ЛИШЕ кольорову гаму на YlGnBu_r.
    """
    setup_fonts(26)

    # Кольорова гама: як просили, беремо з слайда 2 (YlGnBu_r)
    cmap = sns.color_palette('YlGnBu_r', as_cmap=True)

    cols_to_show = method_cols + ['Average Rank', 'Overall Rank']
    plot_data = plot_df[cols_to_show]

    # Анотації: Average Rank з 1 десятковим, інші — цілі
    annot_data = plot_data.copy()
    annot_texts = pd.DataFrame(index=annot_data.index, columns=annot_data.columns)
    for col in annot_data.columns:
        for idx in annot_data.index:
            val = annot_data.loc[idx, col]
            if col == 'Average Rank':
                annot_texts.loc[idx, col] = f"{val:.1f}"
            else:
                annot_texts.loc[idx, col] = f"{int(val)}"

    # Робимо перенесення рядків для довгих назв методів по осі X, щоб текст не наповзав
    plot_data.columns = [c.replace(' ', '\n') for c in plot_data.columns]

    # Повертаємо класичні горизонтальні пропорції, але більші для високої роздільної здатності
    plt.figure(figsize=(24, 14))
    ax = plt.gca()

    tick_positions = [5, 10, 15, 20]

    heatmap = sns.heatmap(
        plot_data,
        annot=annot_texts.values,
        cmap=cmap,
        fmt="",
        linewidths=.5,
        annot_kws={"size": 24},
        cbar_kws={'label': 'Rank (lower = more important)', 'ticks': tick_positions}
    )

    # Інвертуємо colorbar (менші ранги зверху)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.invert_yaxis()
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([str(t) for t in tick_positions])
    cbar.ax.tick_params(labelsize=22)
    cbar.set_label('Rank (lower = more important)', size=26)

    # Вісі і підписи
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks(rotation=0, ha='center', fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()

    plt.savefig('feature_importance_heatmap_blue.png', dpi=BASE_FIG_DPI, bbox_inches='tight')
    plt.savefig('feature_importance_heatmap_blue.svg', format='svg', bbox_inches='tight')
    plt.close()


def plot_overall_bar(
    plot_df: pd.DataFrame,
    top_n: int = TOP_N,
    filename_suffix: str = "",
    colorbar_min: float = 10.0,
    custom_title: str | None = None,
) -> None:
    """Горизонтальний bar-chart важливості на основі Average Rank.

    Довжина стовпчика: TOTAL_FEATURES - Average Rank (чим менший Average Rank, тим більше значення).
    Праворуч додається шкала кольорів (colorbar), узгоджена з довжиною стовпчиків.
    """
    setup_fonts(28)

    df = plot_df.copy()
    # Важливість як різниця між TOTAL_FEATURES і Average Rank (як у вимозі)
    # Додатково обмежуємо зверху TOTAL_FEATURES, щоб значення не виходили за межі осі
    df['importance_value'] = (TOTAL_FEATURES - df['Average Rank']).clip(lower=0, upper=TOTAL_FEATURES)
    df = df.sort_values('importance_value', ascending=False)
    if top_n is not None:
        df = df.head(top_n)

    # Кольори: найвище значення = темно-синій
    cmap = sns.color_palette('YlGnBu', as_cmap=True)
    # Шкала важливості на колорбарі: мінімум конфігурується (для all-features = 0)
    norm = Normalize(vmin=colorbar_min, vmax=TOTAL_FEATURES)
    colors = [cmap(norm(v)) for v in df['importance_value']]

    h = max(12, df.shape[0] * 0.8)
    plt.figure(figsize=(20, h))
    ax = plt.gca()

    ax.barh(df.index, df['importance_value'], color=colors, edgecolor='black', linewidth=0.6)
    ax.invert_yaxis()  # найважливіші зверху

    # Підписи значень: тільки для 'order_messages' — всередині праворуч; решта — ззовні праворуч
    for i, (y, v) in enumerate(zip(df.index, df['importance_value'])):
        if str(y) == 'order_messages':
            label_x = min(max(v - 0.3, 0.3), TOTAL_FEATURES - 0.6)
            ax.text(label_x, i, f"{v:.1f}", va='center', ha='right', color='white', clip_on=True, fontsize=24)
        else:
            label_x = v + 0.4
            ax.text(label_x, i, f"{v:.1f}", va='center', ha='left', clip_on=True, fontsize=24)

    # Сітка по X як у прикладі
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    # Межі осі X: 0 ... 24, з включенням 24 у тики
    ax.set_xlim(0, TOTAL_FEATURES)
    ax.set_xticks(list(range(0, TOTAL_FEATURES + 1, 2)))
    
    ax.tick_params(axis='both', which='major', labelsize=26)

    ax.set_xlabel('Importance (Quantity of features - Average Rank)', fontsize=28)
    ax.set_ylabel('Feature', fontsize=28)
    if custom_title:
        plt.title(custom_title, fontsize=30)
    else:
        n_for_title = df.shape[0]
        plt.title(f"Overall feature importance ranking (Top-{n_for_title} of {TOTAL_FEATURES} features)", fontsize=30)

    # Додаємо колорбар праворуч, синхронізований з довжиною стовпчиків
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Importance', size=28)
    cbar.ax.tick_params(labelsize=24)

    plt.tight_layout()

    base = 'feature_importance_overall_blue'
    if filename_suffix:
        base += filename_suffix
    plt.savefig(f'{base}.png', dpi=BASE_FIG_DPI, bbox_inches='tight')
    plt.savefig(f'{base}.svg', format='svg', bbox_inches='tight')
    plt.close()


# ---------------------------- Головний сценарій ---------------------------- #
if __name__ == '__main__':
    setup_fonts(20)
    data = load_data(CSV_PATH)

    # Підготовка даних для теплокарти
    heat_df, method_cols, total_count = prepare_heatmap_frame(data)

    # Побудова теплокарти
    plot_heatmap(heat_df, method_cols, total_count)

    # Побудова bar-chart важливості: Top-15 (беремо heat_df — він вже обрізаний до Top-N)
    plot_overall_bar(heat_df, top_n=TOP_N, filename_suffix="")
    # Побудова bar-chart важливості: усі ознаки (готуємо окремий повний фрейм)
    overall_full_df = prepare_overall_full(data)
    plot_overall_bar(
        overall_full_df,
        top_n=None,
        filename_suffix="_all",
        colorbar_min=0.0,
        custom_title="Overall feature importance ranking",
    )

    print('Saved: feature_importance_heatmap_blue.(png|svg)')
    print('Saved: feature_importance_overall_blue.(png|svg)')
    print('Saved: feature_importance_overall_blue_all.(png|svg)')
