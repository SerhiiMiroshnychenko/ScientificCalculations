import pandas as pd
from pathlib import Path

NEW_B2B_PATH = Path(r"b2b_for_ml.csv")
OLD_B2B_PATH = Path(r"b2b.csv")

# Для db1: numeric(order_id_raw) + 1_000_000
ORDER_ID_OFFSET = 1_000_000


def load_b2b(path: Path, label: str) -> pd.DataFrame:
    print(f"[INFO] Читаю {label} з {path}...")
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Файл не знайдено: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] {label}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    return df


def normalize_order_id(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Додає колонку order_id_norm (ціле число без зсуву) для порівняння.

    Для NEW (b2b_for_ml): order_id = numeric + OFFSET → numeric = order_id - OFFSET.
    Для OLD (b2b): ми перевіримо, чи працює та сама формула.
    """
    df = df.copy()
    if "order_id" not in df.columns:
        print(f"[WARN] У {label} немає order_id")
        return df

    # Спробуємо інвертувати offset, вважаючи, що order_id = numeric + OFFSET
    df["order_id_norm"] = pd.to_numeric(df["order_id"], errors="coerce") - ORDER_ID_OFFSET
    return df


def compare_shapes_and_columns(df_new: pd.DataFrame, df_old: pd.DataFrame) -> None:
    print("\n=== SHAPE / COLUMNS ===")
    print(f"NEW shape: {df_new.shape}")
    print(f"OLD shape: {df_old.shape}")

    cols_new = set(df_new.columns)
    cols_old = set(df_old.columns)

    common = sorted(cols_new & cols_old)
    only_new = sorted(cols_new - cols_old)
    only_old = sorted(cols_old - cols_new)

    print(f"Спільних колонок: {len(common)}")
    print(f"Тільки в NEW: {len(only_new)}")
    if only_new:
        print("  -", only_new)
    print(f"Тільки в OLD: {len(only_old)}")
    if only_old:
        print("  -", only_old)


def compare_order_id_norm(df_new: pd.DataFrame, df_old: pd.DataFrame) -> pd.DataFrame:
    print("\n=== ПОРІВНЯННЯ order_id_norm ===")
    if "order_id_norm" not in df_new.columns or "order_id_norm" not in df_old.columns:
        print("[WARN] Немає order_id_norm в одному з файлів")
        return pd.DataFrame()

    ids_new = set(df_new["order_id_norm"].dropna().astype(int))
    ids_old = set(df_old["order_id_norm"].dropna().astype(int))

    common = ids_new & ids_old
    only_new = ids_new - ids_old
    only_old = ids_old - ids_new

    print(f"Загалом order_id_norm у NEW: {len(ids_new)}")
    print(f"Загалом order_id_norm у OLD: {len(ids_old)}")
    print(f"Спільних order_id_norm:      {len(common)}")
    print(f"Тільки в NEW:                {len(only_new)}")
    print(f"Тільки в OLD:                {len(only_old)}")

    if common:
        sample = list(sorted(common))[:10]
        print("Приклади спільних order_id_norm (до 10):", sample)

    # Побудуємо датафрейм лише по спільних нормалізованих order_id
    df_new_sub = df_new[df_new["order_id_norm"].isin(common)].copy()
    df_old_sub = df_old[df_old["order_id_norm"].isin(common)].copy()

    # Нормалізуємо типи
    df_new_sub["order_id_norm"] = df_new_sub["order_id_norm"].astype(int)
    df_old_sub["order_id_norm"] = df_old_sub["order_id_norm"].astype(int)

    merged = df_new_sub.merge(
        df_old_sub,
        on="order_id_norm",
        how="inner",
        suffixes=("_new", "_old"),
    )
    print(f"[INFO] Розмір merged по order_id_norm: {merged.shape}")

    return merged


def compare_columns_detailed(merged: pd.DataFrame) -> None:
    print("\n=== ДЕТАЛЬНЕ ПОРІВНЯННЯ КОЛОНОК ПО СПІЛЬНИХ order_id_norm ===")

    # Ядро order-рівня
    order_cols = [
        "is_successful",
        "order_amount",
        "order_messages",
        "order_changes",
    ]

    # partner-рівень
    partner_cols = [
        "partner_total_orders",
        "partner_total_messages",
        "partner_success_rate",
        "partner_order_age_days",
        "partner_avg_amount",
        "partner_success_avg_amount",
        "partner_fail_avg_amount",
        "partner_success_avg_messages",
        "partner_fail_avg_messages",
        "partner_success_avg_changes",
        "partner_fail_avg_changes",
    ]

    for col in order_cols + partner_cols:
        col_new = f"{col}_new"
        col_old = f"{col}_old"
        if col_new not in merged.columns or col_old not in merged.columns:
            print(f"\n[SKIP] Колонка {col} відсутня в одному з файлів")
            continue

        s_new = merged[col_new]
        s_old = merged[col_old]

        # Приводимо до чисел, де це можливо
        if pd.api.types.is_numeric_dtype(s_new) and pd.api.types.is_numeric_dtype(s_old):
            diff = (s_new - s_old).fillna(0)
            equal = (diff == 0).sum()
            total = len(diff)
            ratio = equal / total if total else 0.0

            print(f"\n--- Колонка '{col}' (numeric) ---")
            print(f"Співпало значень: {equal}/{total} ({ratio:.4f})")
            if ratio < 1.0:
                print("Приклади розбіжностей (до 5):")
                diffs = merged.loc[diff != 0, ["order_id_norm", col_new, col_old]].head(5)
                print(diffs)
        else:
            equal = (s_new == s_old).sum()
            total = len(s_new)
            ratio = equal / total if total else 0.0

            print(f"\n--- Колонка '{col}' (non-numeric) ---")
            print(f"Співпало значень: {equal}/{total} ({ratio:.4f})")
            if ratio < 1.0:
                print("Приклади розбіжностей (до 5):")
                diffs = merged.loc[s_new != s_old, ["order_id_norm", col_new, col_old]].head(5)
                print(diffs)


def main() -> None:
    print("[INFO] === ДЕТАЛЬНЕ ПОРІВНЯННЯ B2B-ФАЙЛІВ ===")
    print(f"[INFO] NEW_B2B_PATH = {NEW_B2B_PATH}")
    print(f"[INFO] OLD_B2B_PATH = {OLD_B2B_PATH}")

    df_new = load_b2b(NEW_B2B_PATH, "NEW (b2b_for_ml)")
    df_old = load_b2b(OLD_B2B_PATH, "OLD (b2b)")

    compare_shapes_and_columns(df_new, df_old)

    df_new_norm = normalize_order_id(df_new, "NEW")
    df_old_norm = normalize_order_id(df_old, "OLD")

    merged = compare_order_id_norm(df_new_norm, df_old_norm)
    if merged.empty:
        print("[INFO] merged порожній, немає що порівнювати далі")
        return

    compare_columns_detailed(merged)

    print("[INFO] === ЗАВЕРШЕНО ДЕТАЛЬНЕ ПОРІВНЯННЯ B2B-ФАЙЛІВ ===")


if __name__ == "__main__":
    main()
