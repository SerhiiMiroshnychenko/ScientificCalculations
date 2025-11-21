import pandas as pd
from pathlib import Path

# ==========================================
# НАЛАШТУВАННЯ ШЛЯХІВ
# ==========================================
# EXTENDED_PATH — файл з data_collector (еталонний extended)
# DB1_PATH      — файл з pgAdmin (db1.csv), побудований за нашим SQL

EXTENDED_PATH = Path(r"extended_customer_data_2025-01-17.csv")
DB1_PATH = Path(r"db1.csv")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    print(f"[INFO] Читаю {label} з {path}...")
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Файл не знайдено: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] {label}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    print(f"[INFO] Перші колонки {label}: {list(df.columns[:15])}")
    return df


def compare_columns(df_ext: pd.DataFrame, df_db1: pd.DataFrame) -> None:
    cols_ext = set(df_ext.columns)
    cols_db1 = set(df_db1.columns)

    common = sorted(cols_ext & cols_db1)
    only_ext = sorted(cols_ext - cols_db1)
    only_db1 = sorted(cols_db1 - cols_ext)

    print("\n=== ПОРІВНЯННЯ КОЛОНОК ===")
    print(f"Спільних колонок: {len(common)}")
    print(f"Тільки в EXTENDED: {len(only_ext)}")
    if only_ext:
        print("  -", only_ext)
    print(f"Тільки в DB1: {len(only_db1)}")
    if only_db1:
        print("  -", only_db1)


def compare_order_ids(df_ext: pd.DataFrame, df_db1: pd.DataFrame) -> None:
    print("\n=== ПОРІВНЯННЯ ЗА order_id ===")
    if "order_id" not in df_ext.columns or "order_id" not in df_db1.columns:
        print("[WARN] Немає order_id в одному з файлів — пропускаю цей блок")
        return

    ids_ext = set(df_ext["order_id"].astype(str))
    ids_db1 = set(df_db1["order_id"].astype(str))

    common = ids_ext & ids_db1
    only_ext = ids_ext - ids_db1
    only_db1 = ids_db1 - ids_ext

    print(f"Загалом order_id в EXTENDED: {len(ids_ext)}")
    print(f"Загалом order_id в DB1:      {len(ids_db1)}")
    print(f"Спільних order_id:           {len(common)}")
    print(f"Тільки в EXTENDED:           {len(only_ext)}")
    print(f"Тільки в DB1:                {len(only_db1)}")

    if common:
        sample = list(common)[:10]
        print("Приклади спільних order_id (до 10):", sample)


def compare_numeric_column(col: str, df_ext: pd.DataFrame, df_db1: pd.DataFrame) -> None:
    print(f"\n--- ПОРІВНЯННЯ КОЛОНКИ '{col}' ---")
    if col not in df_ext.columns or col not in df_db1.columns:
        print("[INFO] Колонки немає хоча б в одному з файлів — пропускаю")
        return

    # Merge по order_id, щоб зіставити значення
    merged = df_ext[["order_id", col]].merge(
        df_db1[["order_id", col]], on="order_id", how="inner", suffixes=("_ext", "_db1")
    )

    if merged.empty:
        print("[INFO] Немає спільних order_id для порівняння цієї колонки")
        return

    # Спроба привести до числового типу
    for suf in ("_ext", "_db1"):
        merged[f"{col}{suf}"] = pd.to_numeric(merged[f"{col}{suf}"], errors="coerce")

    mask_valid = merged[f"{col}_ext"].notna() & merged[f"{col}_db1"].notna()
    comp = merged[mask_valid].copy()

    if comp.empty:
        print("[INFO] Після приведення до чисел немає валідних значень для порівняння")
        return

    comp["diff"] = comp[f"{col}_db1"] - comp[f"{col}_ext"]
    total = len(comp)
    equal = (comp["diff"] == 0).sum()
    match_ratio = equal / total if total else 0.0

    print(f"Кількість порівняних рядків: {total}")
    print(f"Співпало значень: {equal} ({match_ratio:.4f} від загальної кількості)")

    # Показати кілька прикладів різниць
    diffs = comp[comp["diff"] != 0].head(5)
    if not diffs.empty:
        print("Приклади розбіжностей (до 5):")
        print(diffs[["order_id", f"{col}_ext", f"{col}_db1", "diff"]])


def main() -> None:
    print("[INFO] === ПОЧАТОК ПОРІВНЯННЯ EXTENDED VS DB1 ===")
    print(f"[INFO] EXTENDED_PATH = {EXTENDED_PATH}")
    print(f"[INFO] DB1_PATH      = {DB1_PATH}")

    df_ext = load_csv(EXTENDED_PATH, "EXTENDED (data_collector_extended)")
    df_db1 = load_csv(DB1_PATH, "DB1 (db1.csv)")

    compare_columns(df_ext, df_db1)
    compare_order_ids(df_ext, df_db1)

    # Порівняємо ключові order-рівневі колонки, якщо вони спільні
    for col in [
        "create_date",
        "date_order",
        "previous_orders_count",
        "total_amount",
        "changes_count",
        "messages_count",
    ]:
        compare_numeric_column(col, df_ext, df_db1)

    print("[INFO] === ЗАВЕРШЕНО ПОРІВНЯННЯ EXTENDED VS DB1 ===")


if __name__ == "__main__":
    main()
