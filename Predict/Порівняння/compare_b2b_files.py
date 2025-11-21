import pandas as pd
from pathlib import Path


# ==========================================
# НАЛАШТУВАННЯ ШЛЯХІВ ДО B2B-ФАЙЛІВ
# ==========================================
# Онови шляхи за потреби під свою структуру.
# new_b2b_path  — новий файл, збудований з pgAdmin (build_b2b_from_pgadmin_export.py)
# old_b2b_path  — існуючий робочий b2b, який використовують модулі Odoo.

NEW_B2B_PATH = Path(r"b2b_for_ml.csv")
OLD_B2B_PATH = Path(r"b2b.csv")


def load_b2b(path: Path, label: str) -> pd.DataFrame:
    """Завантажує b2b CSV з базовим логуванням."""
    print(f"[INFO] Читаю {label} з {path}...")
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Файл не знайдено: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] {label}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    print(f"[INFO] Перші колонки {label}: {list(df.columns[:10])}")
    return df


def compare_columns(df_new: pd.DataFrame, df_old: pd.DataFrame) -> None:
    """Порівняння наборів колонок між файлами."""
    cols_new = set(df_new.columns)
    cols_old = set(df_old.columns)

    common = sorted(cols_new & cols_old)
    only_new = sorted(cols_new - cols_old)
    only_old = sorted(cols_old - cols_new)

    print("\n=== ПОРІВНЯННЯ КОЛОНОК ===")
    print(f"Спільних колонок: {len(common)}")
    print(f"Тільки в NEW (b2b_for_ml): {len(only_new)}")
    if only_new:
        print("  -", only_new)
    print(f"Тільки в OLD (b2b): {len(only_old)}")
    if only_old:
        print("  -", only_old)


def compare_is_successful(df_new: pd.DataFrame, df_old: pd.DataFrame) -> None:
    """Порівняння розподілу is_successful, якщо колонка присутня."""
    print("\n=== ПОРІВНЯННЯ is_successful ===")

    if "is_successful" not in df_new.columns:
        print("[WARN] У NEW (b2b_for_ml) немає колонки is_successful")
    else:
        print("NEW (b2b_for_ml) — value_counts(is_successful):")
        print(df_new["is_successful"].value_counts(dropna=False))

    if "is_successful" not in df_old.columns:
        print("[WARN] У OLD (b2b) немає колонки is_successful")
    else:
        print("\nOLD (b2b) — value_counts(is_successful):")
        print(df_old["is_successful"].value_counts(dropna=False))


def compare_basic_stats(df_new: pd.DataFrame, df_old: pd.DataFrame) -> None:
    """Базові статистики по спільних числових колонках."""
    print("\n=== БАЗОВІ СТАТИСТИКИ ПО СПІЛЬНИХ ЧИСЛОВИХ КОЛОНКАХ ===")

    common_cols = sorted(set(df_new.columns) & set(df_old.columns))
    numeric_common = [c for c in common_cols if pd.api.types.is_numeric_dtype(df_new[c])]

    if not numeric_common:
        print("[INFO] Немає спільних числових колонок для порівняння.")
        return

    print(f"Спільні числові колонки: {numeric_common}")

    desc_new = df_new[numeric_common].describe().T
    desc_old = df_old[numeric_common].describe().T

    print("\nNEW (b2b_for_ml) — describe():")
    print(desc_new)

    print("\nOLD (b2b) — describe():")
    print(desc_old)


def compare_by_order_id(df_new: pd.DataFrame, df_old: pd.DataFrame) -> None:
    """Порівняння перетину за order_id та долі ідентичних рядків по спільних колонках."""
    print("\n=== ПОРІВНЯННЯ ЗА order_id ===")

    if "order_id" not in df_new.columns or "order_id" not in df_old.columns:
        print("[WARN] Не можу порівняти по order_id — немає колонки в одному з файлів.")
        return

    ids_new = set(df_new["order_id"].astype(str))
    ids_old = set(df_old["order_id"].astype(str))

    common_ids = ids_new & ids_old
    only_new_ids = ids_new - ids_old
    only_old_ids = ids_old - ids_new

    print(f"Загалом order_id у NEW: {len(ids_new)}")
    print(f"Загалом order_id у OLD: {len(ids_old)}")
    print(f"Спільних order_id: {len(common_ids)}")
    print(f"Тільки в NEW (b2b_for_ml): {len(only_new_ids)}")
    print(f"Тільки в OLD (b2b): {len(only_old_ids)}")

    # Оцінимо, наскільки однакові рядки по спільних колонках на перетині order_id
    if not common_ids:
        print("[INFO] Немає спільних order_id для детального порівняння.")
        return

    # Обмежимося підмножиною спільних order_id для швидкості, якщо їх дуже багато
    MAX_SAMPLE = 10000
    common_ids_sample = list(common_ids)[:MAX_SAMPLE]

    df_new_sub = df_new[df_new["order_id"].astype(str).isin(common_ids_sample)].copy()
    df_old_sub = df_old[df_old["order_id"].astype(str).isin(common_ids_sample)].copy()

    # Вирівнюємо типи order_id для надійного merge
    df_new_sub["order_id"] = df_new_sub["order_id"].astype(str)
    df_old_sub["order_id"] = df_old_sub["order_id"].astype(str)

    common_cols = sorted(set(df_new_sub.columns) & set(df_old_sub.columns))

    merged = df_new_sub.merge(df_old_sub, on="order_id", how="inner", suffixes=("_new", "_old"))
    print(f"Кількість рядків у merge по спільному order_id (sample): {merged.shape[0]}")

    if not common_cols:
        print("[INFO] Немає спільних колонок для порівняння значень.")
        return

    # Перевіримо, у якій частці рядків усі спільні колонки збігаються
    equal_mask = pd.Series(True, index=merged.index)
    for col in common_cols:
        col_new = f"{col}_new"
        col_old = f"{col}_old"
        if col_new not in merged.columns or col_old not in merged.columns:
            continue
        equal_mask &= (merged[col_new] == merged[col_old])

    identical_rows = equal_mask.sum()
    total_rows = len(equal_mask)

    print(f"Ідентичних рядків по всіх спільних колонках (у sample): {identical_rows} з {total_rows}")
    if total_rows > 0:
        print(f"Частка повністю однакових рядків: {identical_rows / total_rows:.3f}")


def main() -> None:
    print("[INFO] === ПОЧАТОК ПОРІВНЯННЯ B2B-ФАЙЛІВ ===")
    print(f"[INFO] NEW_B2B_PATH = {NEW_B2B_PATH}")
    print(f"[INFO] OLD_B2B_PATH = {OLD_B2B_PATH}")

    df_new = load_b2b(NEW_B2B_PATH, "NEW (b2b_for_ml)")
    df_old = load_b2b(OLD_B2B_PATH, "OLD (b2b)")

    compare_columns(df_new, df_old)
    compare_is_successful(df_new, df_old)
    compare_basic_stats(df_new, df_old)
    compare_by_order_id(df_new, df_old)

    print("[INFO] === ЗАВЕРШЕНО ПОРІВНЯННЯ B2B-ФАЙЛІВ ===")


if __name__ == "__main__":
    main()
