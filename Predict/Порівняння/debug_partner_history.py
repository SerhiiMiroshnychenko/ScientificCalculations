import pandas as pd
from pathlib import Path

NEW_B2B_PATH = Path(r"b2b_for_ml.csv")
OLD_B2B_PATH = Path(r"b2b.csv")

# Вкажи тут partner_id, який хочеш дослідити (зі зсувом, як у b2b.csv)
PARTNER_ID = 1000003


def load_b2b(path: Path, label: str) -> pd.DataFrame:
    print(f"[INFO] Читаю {label} з {path}...")
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Файл не знайдено: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] {label}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    return df


def debug_partner_history(partner_id: int, df_new: pd.DataFrame, df_old: pd.DataFrame) -> None:
    base_cols = [
        "order_id",
        "create_date",
        "is_successful",
        "order_amount",
        "order_messages",
        "partner_total_orders",
        "partner_success_orders",
        "partner_success_rate",
        "partner_total_messages",
    ]

    # Беремо тільки ті колонки, які реально присутні в обох датафреймах
    common_cols = [c for c in base_cols if c in df_new.columns and c in df_old.columns]
    if not common_cols:
        print("[WARN] Немає спільних колонок для дебагу історії партнера")
        return

    new_p = df_new[df_new["partner_id"] == partner_id][common_cols].copy()
    old_p = df_old[df_old["partner_id"] == partner_id][common_cols].copy()

    if new_p.empty and old_p.empty:
        print(f"[INFO] Для partner_id={partner_id} немає рядків ні в NEW, ні в OLD")
        return

    print(f"\n=== ІСТОРІЯ ДЛЯ partner_id={partner_id} ===")

    # Сортуємо однаково
    new_p.sort_values("create_date", inplace=True)
    old_p.sort_values("create_date", inplace=True)

    print("\n--- NEW (b2b_for_ml) ---")
    print(new_p.head(20))

    print("\n--- OLD (b2b) ---")
    print(old_p.head(20))

    # Порівняння по спільних order_id
    common_ids = set(new_p["order_id"]).intersection(set(old_p["order_id"]))
    if not common_ids:
        print("\n[INFO] Немає спільних order_id для цього partner_id у NEW vs OLD")
        return

    common_ids = list(common_ids)
    common_ids.sort()
    sample_ids = common_ids[:10]

    print("\n--- ДЕТАЛЬНЕ ПОРІВНЯННЯ ПО СПІЛЬНИХ order_id (до 10) ---")
    print("order_id in sample:", sample_ids)

    new_s = new_p[new_p["order_id"].isin(sample_ids)].copy()
    old_s = old_p[old_p["order_id"].isin(sample_ids)].copy()

    merged = new_s.merge(
        old_s,
        on="order_id",
        how="inner",
        suffixes=("_new", "_old"),
    )

    cols_to_show = [
        "order_id",
        "create_date_new",
        "is_successful_new",
        "is_successful_old",
        "order_amount_new",
        "order_amount_old",
        "order_messages_new",
        "order_messages_old",
        "partner_total_orders_new",
        "partner_total_orders_old",
        "partner_success_rate_new",
        "partner_success_rate_old",
        "partner_total_messages_new",
        "partner_total_messages_old",
    ]

    available_cols = [c for c in cols_to_show if c in merged.columns]
    print(merged[available_cols].sort_values("create_date_new").head(10))


def main() -> None:
    print("[INFO] === ДЕБАГ ІСТОРІЇ PARTNER_ID В B2B-ФАЙЛАХ ===")
    print(f"[INFO] NEW_B2B_PATH = {NEW_B2B_PATH}")
    print(f"[INFO] OLD_B2B_PATH = {OLD_B2B_PATH}")
    print(f"[INFO] PARTNER_ID   = {PARTNER_ID}")

    df_new = load_b2b(NEW_B2B_PATH, "NEW (b2b_for_ml)")
    df_old = load_b2b(OLD_B2B_PATH, "OLD (b2b)")

    debug_partner_history(PARTNER_ID, df_new, df_old)

    print("[INFO] === ЗАВЕРШЕНО ДЕБАГ ІСТОРІЇ PARTNER_ID ===")


if __name__ == "__main__":
    main()
