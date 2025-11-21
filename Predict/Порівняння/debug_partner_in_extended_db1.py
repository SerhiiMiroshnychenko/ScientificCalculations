import pandas as pd
from pathlib import Path

# Файли для порівняння
EXTENDED_PATH = Path(r"extended_customer_data_2025-01-17.csv")
DB1_PATH = Path(r"db1.csv")

# partner_id у b2b (зі зсувом). Для db1 offset = 1_000_000, тому сирий partner_id = PARTNER_ID_B2B - 1_000_000
PARTNER_ID_B2B = 1000003


def load_csv(path: Path, label: str) -> pd.DataFrame:
    print(f"[INFO] Читаю {label} з {path}...")
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Файл не знайдено: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] {label}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    return df


def debug_partner(ext_df: pd.DataFrame, db1_df: pd.DataFrame, partner_id_b2b: int) -> None:
    # Для db1 (db_index=1) внутрішній partner_id = partner_id_b2b - 1_000_000
    raw_partner_id = partner_id_b2b - 1_000_000
    print(f"[INFO] PARTNER_ID_B2B = {partner_id_b2b}, raw partner_id = {raw_partner_id}")

    # В extended partner_id називається customer_id
    if "customer_id" not in ext_df.columns:
        print("[ERROR] У data_collector_extended.csv немає customer_id")
        return

    ext_p = ext_df[ext_df["customer_id"] == raw_partner_id].copy()
    db1_p = db1_df[db1_df["partner_id"] == raw_partner_id].copy()

    print(f"[INFO] Рядків для цього партнера в EXTENDED: {ext_p.shape[0]}")
    print(f"[INFO] Рядків для цього партнера в DB1:      {db1_p.shape[0]}")

    if ext_p.empty and db1_p.empty:
        print("[INFO] Для цього партнера немає жодного замовлення ні в EXTENDED, ні в DB1")
        return

    cols_ext = [
        c for c in [
            "order_id",
            "create_date",
            "date_order",
            "messages_count",
            "changes_count",
            "total_amount",
            "state",
        ]
        if c in ext_p.columns
    ]
    cols_db1 = [
        c for c in [
            "order_id",
            "create_date",
            "date_order",
            "messages_count",
            "changes_count",
            "total_amount",
            "state",
        ]
        if c in db1_p.columns
    ]

    print("\n--- EXTENDED (data_collector_extended) для цього партнера ---")
    ext_p_sorted = ext_p.sort_values("create_date")[cols_ext]
    print(ext_p_sorted.head(30))

    print("\n--- DB1 (db1.csv) для цього партнера ---")
    db1_p_sorted = db1_p.sort_values("create_date")[cols_db1]
    print(db1_p_sorted.head(30))

    # Порівняння order_id множин
    ext_ids = set(ext_p_sorted["order_id"].astype(str)) if "order_id" in ext_p_sorted.columns else set()
    db1_ids = set(db1_p_sorted["order_id"].astype(str)) if "order_id" in db1_p_sorted.columns else set()

    common_ids = ext_ids & db1_ids
    only_ext = ext_ids - db1_ids
    only_db1 = db1_ids - ext_ids

    print("\n=== ПОРІВНЯННЯ order_id ДЛЯ ЦЬОГО ПАРТНЕРА ===")
    print(f"Загалом order_id в EXTENDED: {len(ext_ids)}")
    print(f"Загалом order_id в DB1:      {len(db1_ids)}")
    print(f"Спільних order_id:           {len(common_ids)}")
    print(f"Тільки в EXTENDED:           {len(only_ext)}")
    print(f"Тільки в DB1:                {len(only_db1)}")

    if common_ids:
        sample = list(sorted(common_ids))[:10]
        print("Приклади спільних order_id (до 10):", sample)

    if only_ext:
        sample = list(sorted(only_ext))[:10]
        print("Приклади order_id тільки в EXTENDED (до 10):", sample)

    if only_db1:
        sample = list(sorted(only_db1))[:10]
        print("Приклади order_id тільки в DB1 (до 10):", sample)


def main() -> None:
    print("[INFO] === ДЕБАГ ПАРТНЕРА В EXTENDED VS DB1 ===")
    print(f"[INFO] EXTENDED_PATH = {EXTENDED_PATH}")
    print(f"[INFO] DB1_PATH      = {DB1_PATH}")

    ext_df = load_csv(EXTENDED_PATH, "EXTENDED (data_collector_extended)")
    db1_df = load_csv(DB1_PATH, "DB1 (db1.csv)")

    debug_partner(ext_df, db1_df, PARTNER_ID_B2B)

    print("[INFO] === ЗАВЕРШЕНО ДЕБАГ ПАРТНЕРА В EXTENDED VS DB1 ===")


if __name__ == "__main__":
    main()
