import pandas as pd
from pathlib import Path

# Сирі дані
EXTENDED_PATH = Path(r"extended_customer_data_2025-01-17.csv")
DB1_PATH = Path(r"db1.csv")

# Оброблені дані
NEW_B2B_PATH = Path(r"b2b_for_ml.csv")
OLD_B2B_PATH = Path(r"b2b.csv")

ORDER_ID_OFFSET = 1_000_000  # для db1


def load_csv(path: Path, label: str) -> pd.DataFrame:
    print(f"[INFO] Читаю {label} з {path}...")
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Файл не знайдено: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] {label}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    return df


def describe_basic(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== БАЗОВА ІНФОРМАЦІЯ: {label} ===")
    print(f"shape = {df.shape}")
    print("Перші 10 колонок:", list(df.columns[:10]))


def normalize_order_id_raw_from_extended(df_ext: pd.DataFrame) -> pd.DataFrame:
    """Додає numeric order_id_raw до extended (SOxxxxx → xxxxx)."""
    df = df_ext.copy()
    if "order_id" not in df.columns:
        return df
    numeric_part = df["order_id"].astype(str).str.extract(r"(\d+)", expand=False)
    df["order_id_raw"] = pd.to_numeric(numeric_part, errors="coerce")
    return df


def normalize_order_id_raw_from_db1(df_db1: pd.DataFrame) -> pd.DataFrame:
    """Додає numeric order_id_raw до db1 (SOxxxxx → xxxxx)."""
    df = df_db1.copy()
    if "order_id" not in df.columns:
        return df
    numeric_part = df["order_id"].astype(str).str.extract(r"(\d+)", expand=False)
    df["order_id_raw"] = pd.to_numeric(numeric_part, errors="coerce")
    return df


def normalize_order_id_norm_from_b2b(df_b2b: pd.DataFrame, label: str) -> pd.DataFrame:
    """Додає order_id_norm = order_id - OFFSET до b2b-файлів."""
    df = df_b2b.copy()
    if "order_id" not in df.columns:
        print(f"[WARN] У {label} немає order_id")
        return df
    df["order_id_norm"] = pd.to_numeric(df["order_id"], errors="coerce") - ORDER_ID_OFFSET
    return df


def compare_extended_vs_db1_order_ids(df_ext: pd.DataFrame, df_db1: pd.DataFrame) -> None:
    print("\n=== EXTENDED vs DB1: ПОРІВНЯННЯ order_id_raw ===")
    if "order_id_raw" not in df_ext.columns or "order_id_raw" not in df_db1.columns:
        print("[WARN] Немає order_id_raw в одному з датафреймів")
        return

    ids_ext = set(df_ext["order_id_raw"].dropna().astype(int))
    ids_db1 = set(df_db1["order_id_raw"].dropna().astype(int))

    common = ids_ext & ids_db1
    only_ext = ids_ext - ids_db1
    only_db1 = ids_db1 - ids_ext

    print(f"Загалом order_id_raw в EXTENDED: {len(ids_ext)}")
    print(f"Загалом order_id_raw в DB1:      {len(ids_db1)}")
    print(f"Спільних:                        {len(common)}")
    print(f"Тільки в EXTENDED:               {len(only_ext)}")
    print(f"Тільки в DB1:                    {len(only_db1)}")

    if common:
        sample = list(sorted(common))[:10]
        print("Приклади спільних (до 10):", sample)


def link_all_four_layers(df_ext: pd.DataFrame, df_db1: pd.DataFrame,
                         df_new_b2b: pd.DataFrame, df_old_b2b: pd.DataFrame) -> pd.DataFrame:
    """Будує єдину таблицю по order_id_raw / order_id_norm із 4 рівнів."""

    # Візьмемо тільки кілька ключових колонок з кожного рівня
    ext_cols = [
        "order_id_raw",
        "order_id",
        "create_date",
        "total_amount",
        "messages_count",
        "changes_count",
        "state",
    ]
    db1_cols = [
        "order_id_raw",
        "order_id",
        "create_date",
        "total_amount",
        "messages_count",
        "changes_count",
        "state",
    ]

    b2b_cols_order = [
        "order_id_norm",
        "order_id",
        "is_successful",
        "create_date",
        "order_amount",
        "order_messages",
        "order_changes",
    ]

    b2b_cols_partner = [
        "order_id_norm",
        "partner_id",
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

    ext = df_ext[[c for c in ext_cols if c in df_ext.columns]].copy()
    db1 = df_db1[[c for c in db1_cols if c in df_db1.columns]].copy()

    new_order = df_new_b2b[[c for c in b2b_cols_order if c in df_new_b2b.columns]].copy()
    old_order = df_old_b2b[[c for c in b2b_cols_order if c in df_old_b2b.columns]].copy()

    new_partner = df_new_b2b[[c for c in b2b_cols_partner if c in df_new_b2b.columns]].copy()
    old_partner = df_old_b2b[[c for c in b2b_cols_partner if c in df_old_b2b.columns]].copy()

    # Для зв'язування: EXTENDED/DB1 по order_id_raw, B2B по order_id_norm
    # Вважаємо, що order_id_norm = order_id_raw (одна й та сама числова частина)

    # Зведемо EXTENDED + DB1 по order_id_raw
    merged_ext_db1 = ext.merge(
        db1,
        on="order_id_raw",
        how="outer",
        suffixes=("_ext", "_db1"),
    )

    print("[INFO] merged_ext_db1 shape:", merged_ext_db1.shape)

    # Тепер зведемо з NEW/OLD b2b по order_id_norm
    merged_ext_db1.rename(columns={"order_id_raw": "order_id_norm"}, inplace=True)

    merged_all = merged_ext_db1.merge(
        new_order,
        on="order_id_norm",
        how="left",
        suffixes=("", "_new"),
    )

    merged_all = merged_all.merge(
        old_order,
        on="order_id_norm",
        how="left",
        suffixes=("", "_old"),
    )

    # Партнерські метрики теж приєднаємо по order_id_norm.
    # (partner_id у merged_ext_db1 може бути відсутній, тому використовуємо тільки order_id_norm)
    merged_all = merged_all.merge(
        new_partner,
        on="order_id_norm",
        how="left",
        suffixes=("", "_partner_new"),
    )

    merged_all = merged_all.merge(
        old_partner,
        on="order_id_norm",
        how="left",
        suffixes=("", "_partner_old"),
    )

    print("[INFO] merged_all shape:", merged_all.shape)

    return merged_all


def main() -> None:
    print("[INFO] === ПОРІВНЯННЯ 4 РІВНІВ B2B-ДАНИХ ===")
    print(f"[INFO] EXTENDED_PATH = {EXTENDED_PATH}")
    print(f"[INFO] DB1_PATH      = {DB1_PATH}")
    print(f"[INFO] NEW_B2B_PATH  = {NEW_B2B_PATH}")
    print(f"[INFO] OLD_B2B_PATH  = {OLD_B2B_PATH}")

    df_ext = load_csv(EXTENDED_PATH, "EXTENDED")
    df_db1 = load_csv(DB1_PATH, "DB1")
    df_new = load_csv(NEW_B2B_PATH, "NEW (b2b_for_ml)")
    df_old = load_csv(OLD_B2B_PATH, "OLD (b2b)")

    describe_basic(df_ext, "EXTENDED")
    describe_basic(df_db1, "DB1")
    describe_basic(df_new, "NEW (b2b_for_ml)")
    describe_basic(df_old, "OLD (b2b)")

    # Додаємо order_id_raw / order_id_norm
    df_ext_n = normalize_order_id_raw_from_extended(df_ext)
    df_db1_n = normalize_order_id_raw_from_db1(df_db1)

    df_new_n = normalize_order_id_norm_from_b2b(df_new, "NEW")
    df_old_n = normalize_order_id_norm_from_b2b(df_old, "OLD")

    compare_extended_vs_db1_order_ids(df_ext_n, df_db1_n)

    merged_all = link_all_four_layers(df_ext_n, df_db1_n, df_new_n, df_old_n)

    # Показати список усіх колонок (для референсу)
    print("\n=== СПИСОК УСІХ КОЛОНОК merged_all ===")
    print(list(merged_all.columns))

    def compare_metric_across_layers(name: str,
                                     col_ext: str,
                                     col_db1: str,
                                     col_new: str,
                                     col_old: str) -> None:
        print(f"\n=== МЕТРИКА: {name} ===")
        cols = [col_ext, col_db1, col_new, col_old]
        for c in cols:
            if c not in merged_all.columns:
                print(f"[WARN] Колонки {c} немає в merged_all, пропускаю цю метрику")
                return

        a = merged_all[col_ext]
        b = merged_all[col_db1]
        c = merged_all[col_new]
        d = merged_all[col_old]

        # Вирівнюємо типи до float для коректного порівняння
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        c_num = pd.to_numeric(c, errors="coerce")
        d_num = pd.to_numeric(d, errors="coerce")

        mask_valid = ~(a_num.isna() & b_num.isna() & c_num.isna() & d_num.isna())
        a_num = a_num[mask_valid]
        b_num = b_num[mask_valid]
        c_num = c_num[mask_valid]
        d_num = d_num[mask_valid]

        total = len(a_num)
        if total == 0:
            print("[INFO] Немає валідних значень для цієї метрики")
            return

        eq_ext_db1 = (a_num == b_num)
        eq_ext_new = (a_num == c_num)
        eq_ext_old = (a_num == d_num)
        eq_new_old = (c_num == d_num)
        eq_all = eq_ext_db1 & eq_ext_new & eq_ext_old

        def count(mask):
            return int(mask.sum())

        print(f"Усього валідних рядків: {total}")
        print(f"Усi 4 шари рівні (EXT=DB1=NEW=OLD): {count(eq_all)}")
        print(f"EXT=DB1 (сирі збігаються): {count(eq_ext_db1)}")
        print(f"EXT=NEW (новий b2b узгоджений з сирими): {count(eq_ext_new)}")
        print(f"EXT=OLD (старий b2b узгоджений з сирими): {count(eq_ext_old)}")
        print(f"NEW=OLD (два b2b збігаються): {count(eq_new_old)}")

    # Order-рівень
    compare_metric_across_layers(
        name="order_amount",
        col_ext="total_amount_ext",
        col_db1="total_amount_db1",
        col_new="order_amount",
        col_old="order_amount_old",
    )

    compare_metric_across_layers(
        name="messages_count / order_messages",
        col_ext="messages_count_ext",
        col_db1="messages_count_db1",
        col_new="order_messages",
        col_old="order_messages_old",
    )

    compare_metric_across_layers(
        name="changes_count / order_changes",
        col_ext="changes_count_ext",
        col_db1="changes_count_db1",
        col_new="order_changes",
        col_old="order_changes_old",
    )

    # is_successful (EXT/DB1 тут немають цієї колонки, але можна порівняти NEW vs OLD окремо)
    print("\n=== МЕТРИКА: is_successful (NEW vs OLD) ===")
    if "is_successful" in merged_all.columns and "is_successful_old" in merged_all.columns:
        s_new = merged_all["is_successful"]
        s_old = merged_all["is_successful_old"]
        mask_valid = ~(s_new.isna() & s_old.isna())
        s_new = s_new[mask_valid]
        s_old = s_old[mask_valid]
        total = len(s_new)
        equal = int((s_new == s_old).sum())
        print(f"NEW=OLD is_successful: {equal}/{total} ({equal/total:.4f})")
    else:
        print("[WARN] is_successful / is_successful_old відсутні в merged_all")

    # Partner-рівень: порівнюємо NEW/OLD між собою та з EXT/DB1 опосередковано через total_amount/messages
    compare_metric_across_layers(
        name="partner_total_orders",
        col_ext="partner_total_orders",      # EXT/DB1 тут не мають сенсу, але залишаємо для симетрії
        col_db1="partner_total_orders_partner_old",
        col_new="partner_total_orders",
        col_old="partner_total_orders_partner_old",
    )

    compare_metric_across_layers(
        name="partner_total_messages",
        col_ext="partner_total_messages",
        col_db1="partner_total_messages_partner_old",
        col_new="partner_total_messages",
        col_old="partner_total_messages_partner_old",
    )

    compare_metric_across_layers(
        name="partner_success_rate",
        col_ext="partner_success_rate",
        col_db1="partner_success_rate_partner_old",
        col_new="partner_success_rate",
        col_old="partner_success_rate_partner_old",
    )

    compare_metric_across_layers(
        name="partner_avg_amount",
        col_ext="partner_avg_amount",
        col_db1="partner_avg_amount_partner_old",
        col_new="partner_avg_amount",
        col_old="partner_avg_amount_partner_old",
    )

    print("[INFO] === ЗАВЕРШЕНО ПОРІВНЯННЯ 4 РІВНІВ B2B-ДАНИХ ===")


if __name__ == "__main__":
    main()
