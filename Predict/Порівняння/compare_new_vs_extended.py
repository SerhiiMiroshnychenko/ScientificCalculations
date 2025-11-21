import pandas as pd
from pathlib import Path


NEW_PATH = r"new2.csv"
EXTENDED_PATH = r"extended_customer_data_2025-01-17.csv"


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    print(f"[INFO] Читаю файл: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Файл не знайдено: {p}")
    df = pd.read_csv(p)
    print(f"[INFO] {p.name}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    return df


def main():
    df_new = load_csv(NEW_PATH)
    df_ext = load_csv(EXTENDED_PATH)

    print("\n[INFO] Порівняння назв колонок")
    cols_new = set(df_new.columns)
    cols_ext = set(df_ext.columns)

    common_cols = cols_new & cols_ext
    only_in_new = cols_new - cols_ext
    only_in_ext = cols_ext - cols_new

    print(f"  Спільних колонок: {len(common_cols)}")
    print(f"  Тільки в new.csv: {len(only_in_new)} -> {sorted(only_in_new)}")
    print(f"  Тільки в extended_customer_data: {len(only_in_ext)} -> {sorted(only_in_ext)}")

    if "order_id" not in common_cols:
        print("\n[ERROR] У файлах немає спільної колонки 'order_id' – не можу порівнювати рядки.")
        return

    print("\n[INFO] Кількість унікальних order_id")
    print(f"  new.csv:      {df_new['order_id'].nunique()}")
    print(f"  extended.csv: {df_ext['order_id'].nunique()}")

    print("\n[INFO] Побудова inner join по order_id для порівняння значень...")
    merged = df_new.merge(df_ext, on="order_id", suffixes=("_new", "_ext"), how="inner")
    print(f"  Спільних order_id: {merged.shape[0]}")

    # Порівнюємо тільки спільні колонки (крім order_id)
    cols_to_compare = [c for c in common_cols if c != "order_id"]

    print("\n[INFO] Перевірка однаковості значень по спільних колонках (на спільних order_id)")
    diff_summary = []

    for col in cols_to_compare:
        col_new = f"{col}_new"
        col_ext = f"{col}_ext"

        if col_new not in merged.columns or col_ext not in merged.columns:
            continue

        equal_mask = merged[col_new].astype(str) == merged[col_ext].astype(str)
        equal_count = equal_mask.sum()
        total = len(equal_mask)
        equal_pct = 100.0 * equal_count / total if total else 0.0

        if equal_pct < 100.0:
            diff_summary.append((col, equal_pct))

        print(f"  Колонка {col}: {equal_count}/{total} ({equal_pct:.2f}%) рядків збігаються")

    print("\n[INFO] Підсумок відмінностей по спільних колонках (де не 100% збігів):")
    if not diff_summary:
        print("  Всі спільні колонки повністю збігаються за значеннями на спільних order_id.")
    else:
        for col, pct in sorted(diff_summary, key=lambda x: x[1]):
            print(f"  {col}: {pct:.2f}% збігів")

    print("\n[INFO] Порівняння завершено.")


if __name__ == "__main__":
    main()
