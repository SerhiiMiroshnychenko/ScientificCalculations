# -*- coding: utf-8 -*-
import os
import math
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== Конфіг ====
INPUT_CSV = "feature_importance.csv"   # очікувана структура: feature, stat_model_importance, mfa_importance
TOP_KS = [5, 10]                       # для метрик перетину топ-k
TOP_N_OUTLIERS = 12                    # скільки найбільших розбіжностей підписувати/малювати
BUMP_TOP_N = 25                        # скільки ознак показати в bump-chart (за значущістю/середнім рангом)
FIG_DPI = 140
PALETTE = {
    "stat": "#1f77b4",   # синій
    "mfa": "#d62728",    # червоний
    "gray": "#7f7f7f",
}

def _clean_feature_name(s: str) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip()

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"feature", "stat_model_importance", "mfa_importance"}
    if not needed.issubset(set(c.lower() for c in df.columns)):
        # спроба нормалізувати назви колонок
        rename_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ["feature", "features", "name"]:
                rename_map[c] = "feature"
            elif "stat" in cl and "import" in cl:
                rename_map[c] = "stat_model_importance"
            elif ("mfa" in cl or "multi" in cl) and "import" in cl:
                rename_map[c] = "mfa_importance"
        df = df.rename(columns=rename_map)
    assert {"feature","stat_model_importance","mfa_importance"}.issubset(df.columns), \
        "Очікувані колонки: feature, stat_model_importance, mfa_importance"

    df["feature"] = df["feature"].map(_clean_feature_name)
    df = df.dropna(subset=["feature","stat_model_importance","mfa_importance"])
    # Привести до чисел
    df["stat_model_importance"] = pd.to_numeric(df["stat_model_importance"], errors="coerce")
    df["mfa_importance"] = pd.to_numeric(df["mfa_importance"], errors="coerce")
    df = df.dropna(subset=["stat_model_importance","mfa_importance"])
    # Прибрати дублікатні імена (беремо першу появу)
    df = df.drop_duplicates(subset=["feature"], keep="first").reset_index(drop=True)
    return df

def normalize_importances(df: pd.DataFrame) -> pd.DataFrame:
    # від’ємні значення обрізаємо до 0 (на випадок нестандартів)
    df["stat_clip"] = df["stat_model_importance"].clip(lower=0)
    df["mfa_clip"] = df["mfa_importance"].clip(lower=0)
    # нормування до суми 1
    stat_sum = df["stat_clip"].sum()
    mfa_sum  = df["mfa_clip"].sum()
    if stat_sum == 0 or mfa_sum == 0:
        raise ValueError("Сума важливостей дорівнює нулю для одного з методів після обрізання <0.")
    df["stat_norm"] = df["stat_clip"] / stat_sum
    df["mfa_norm"]  = df["mfa_clip"]  / mfa_sum
    return df

def rank_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ранги: 1 = найважливіша ознака (спадний порядок)
    df["rank_stat"] = df["stat_norm"].rank(method="average", ascending=False).astype(int)
    df["rank_mfa"]  = df["mfa_norm"].rank(method="average",  ascending=False).astype(int)
    df["rank_diff"] = (df["rank_stat"] - df["rank_mfa"]).abs()
    df["rank_signed_diff"] = df["rank_stat"] - df["rank_mfa"]
    # середній ранг для сортувань
    df["rank_mean"] = (df["rank_stat"] + df["rank_mfa"])/2.0
    return df

def compute_metrics(df: pd.DataFrame, top_ks=TOP_KS) -> dict:
    # Кореляції
    pearson = df[["stat_norm","mfa_norm"]].corr(method="pearson").iloc[0,1]
    spearman = df[["stat_norm","mfa_norm"]].corr(method="spearman").iloc[0,1]
    kendall = df[["stat_norm","mfa_norm"]].corr(method="kendall").iloc[0,1]

    # Перетини топ-k
    metrics = {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "kendall": float(kendall),
    }
    # Списки топ-k за кожним методом
    df_stat_sorted = df.sort_values("stat_norm", ascending=False)
    df_mfa_sorted  = df.sort_values("mfa_norm", ascending=False)
    for k in top_ks:
        top_stat = set(df_stat_sorted.head(k)["feature"])
        top_mfa  = set(df_mfa_sorted.head(k)["feature"])
        inter = len(top_stat & top_mfa)
        union = len(top_stat | top_mfa)
        jaccard = inter/union if union>0 else 0.0
        metrics[f"top_{k}_intersection"] = inter
        metrics[f"top_{k}_jaccard"] = jaccard

    # Середня абсолютна різниця рангів
    metrics["mean_abs_rank_diff"] = float(df["rank_diff"].mean())
    metrics["median_abs_rank_diff"] = float(df["rank_diff"].median())
    return metrics

def save_summary(df: pd.DataFrame, metrics: dict, out_path="comparison_summary.xlsx"):
    # Розкладемо метрики у DataFrame для Excel
    metrics_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    )
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.sort_values("rank_diff", ascending=False).to_excel(writer, sheet_name="per_feature", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

def wrap_label(s, width=26):
    return "\n".join(textwrap.wrap(s, width=width, replace_whitespace=False))

def plot_scatter(df: pd.DataFrame, out="viz_scatter.png"):
    plt.figure(figsize=(7.6, 6), dpi=FIG_DPI)
    x = df["stat_norm"].values
    y = df["mfa_norm"].values
    plt.scatter(x, y, s=36, color="#555", alpha=0.7, edgecolor="white", linewidth=0.5)

    # Лінія y = x
    xymax = max(float(x.max()), float(y.max()))
    plt.plot([0, xymax], [0, xymax], color="#aaaaaa", linestyle="--", linewidth=1)

    # Підписати найбільші відхилення (за різницею нормованих значень або рангів)
    df_annot = df.assign(diff_val=(df["stat_norm"] - df["mfa_norm"]).abs())
    df_annot = df_annot.sort_values(["rank_diff","diff_val"], ascending=False).head(TOP_N_OUTLIERS)
    for _, r in df_annot.iterrows():
        plt.annotate(
            wrap_label(r["feature"], 24),
            (r["stat_norm"], r["mfa_norm"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color="#333"
        )

    plt.xlabel("Важливість (статистична модель) — нормована")
    plt.ylabel("Важливість (багатофакторний аналіз) — нормована")
    plt.title("Порівняння важливостей: scatter із лінією y=x")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_dumbbell(df: pd.DataFrame, out="viz_dumbbell.png", top_n=None):
    # Сортуємо за середнім рангом (менший = важливіше)
    work = df.sort_values(["rank_mean","feature"], ascending=[True, True])
    if top_n:
        work = work.head(top_n)
    # Інвертуємо порядок для горизонтального графіка (верх — важливіше)
    work = work.iloc[::-1].copy()

    plt.figure(figsize=(10, max(5, 0.35*len(work))), dpi=FIG_DPI)
    y_pos = np.arange(len(work))
    # лінія між точками
    for i, (_, r) in enumerate(work.iterrows()):
        plt.plot(
            [r["stat_norm"], r["mfa_norm"]],
            [i, i],
            color=PALETTE["gray"],
            linewidth=1.5,
            alpha=0.8
        )
    # точки
    plt.scatter(work["stat_norm"], y_pos, color=PALETTE["stat"], label="Стат. модель", s=36)
    plt.scatter(work["mfa_norm"],  y_pos, color=PALETTE["mfa"],  label="Багатофакторний аналіз", s=36)

    plt.yticks(y_pos, [wrap_label(f, 40) for f in work["feature"]])
    plt.xlabel("Нормована важливість")
    plt.title("Поозначний порівняльний графік (dumbbell)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_bump(df: pd.DataFrame, out="viz_bump.png", top_n=BUMP_TOP_N):
    # Показуємо найважливіші (за середнім рангом)
    work = df.sort_values("rank_mean", ascending=True).head(top_n).copy()
    # Нормуємо ранг: 1 нагорі, тому інвертуємо вісь y
    plt.figure(figsize=(8, max(5, 0.35*len(work))), dpi=FIG_DPI)

    x_positions = [0, 1]  # 0=stat, 1=mfa
    for _, r in work.iterrows():
        xs = x_positions
        ys = [r["rank_stat"], r["rank_mfa"]]
        plt.plot(xs, ys, color="#bbbbbb", linewidth=1.5, zorder=1)
        plt.scatter(xs[0], ys[0], color=PALETTE["stat"], s=30, zorder=2)
        plt.scatter(xs[1], ys[1], color=PALETTE["mfa"],  s=30, zorder=2)
        # підписи зліва/справа
        plt.text(xs[0]-0.03, ys[0], r["feature"], ha="right", va="center", fontsize=8)
        plt.text(xs[1]+0.03, ys[1], r["feature"], ha="left",  va="center", fontsize=8)

    plt.gca().invert_yaxis()
    plt.xticks([0,1], ["Стат. модель", "Багатофакторний аналіз"])
    plt.yticks(range(1, int(work[["rank_stat","rank_mfa"]].max().max())+1))
    plt.title(f"Зміни рангів (bump chart) — топ-{len(work)} ознак")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_rank_diff_bars(df: pd.DataFrame, out="viz_rank_diff.png", top_n=TOP_N_OUTLIERS):
    work = df.sort_values("rank_diff", ascending=False).head(top_n).copy()
    plt.figure(figsize=(8.4, max(4, 0.35*len(work))), dpi=FIG_DPI)
    y = np.arange(len(work))
    plt.barh(y, work["rank_diff"], color="#8888cc")
    plt.yticks(y, [wrap_label(f, 40) for f in work["feature"]])
    plt.xlabel("|Δ рангів|")
    plt.title(f"Найбільші розбіжності в рангах (топ-{top_n})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def main():
    if not os.path.exists(INPUT_CSV):
        # Створимо шаблон, щоб було зручно заповнити
        template = pd.DataFrame({
            "feature": ["Feature A", "Feature B", "Feature C"],
            "stat_model_importance": [0.20, 0.15, 0.05],
            "mfa_importance": [0.18, 0.04, 0.11],
        })
        template.to_csv(INPUT_CSV, index=False)
        print(f"Створено шаблон {INPUT_CSV}. Заповніть його вашими даними та запустіть скрипт знову.")
        return

    df = load_data(INPUT_CSV)
    df = normalize_importances(df)
    df = rank_features(df)

    metrics = compute_metrics(df, TOP_KS)
    print("Метрики порівняння:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")

    save_summary(df, metrics, out_path="comparison_summary.xlsx")

    # Візуалізації
    plot_scatter(df, out="viz_scatter.png")
    plot_dumbbell(df, out="viz_dumbbell.png", top_n=None)         # усі ознаки
    plot_bump(df, out="viz_bump.png", top_n=BUMP_TOP_N)           # топ-N за важливістю
    plot_rank_diff_bars(df, out="viz_rank_diff.png", top_n=TOP_N_OUTLIERS)

    print("Збережено файли: viz_scatter.png, viz_dumbbell.png, viz_bump.png, viz_rank_diff.png, comparison_summary.xlsx")

if __name__ == "__main__":
    main()
