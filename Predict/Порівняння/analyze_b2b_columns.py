import pandas as pd
from pathlib import Path


B2B_PATH = r"b2b.csv"


def load_b2b(path: str) -> pd.DataFrame:
    p = Path(path)
    print(f"[INFO] Читаю b2b CSV: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Файл не знайдено: {p}")
    df = pd.read_csv(p)
    print(f"[INFO] {p.name}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    return df


def main():
    df = load_b2b(B2B_PATH)

    print("\n[INFO] Список колонок та типів:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    print("\n[INFO] Перші 5 рядків:")
    print(df.head())

    print("\n[INFO] Базова статистика для числових колонок:")
    print(df.describe())

    # Перелік «критично важливих» колонок, які вже використовуються в модулях
    required_cols = [
        "order_id",
        "is_successful",
        "create_date",
        "partner_id",
        "order_amount",
        "order_messages",
        "order_changes",
        "partner_success_rate",
        "partner_total_orders",
        "partner_order_age_days",
        "partner_avg_amount",
        "partner_success_avg_amount",
        "partner_fail_avg_amount",
        "partner_total_messages",
        "partner_success_avg_messages",
        "partner_fail_avg_messages",
        "partner_avg_changes",
        "partner_success_avg_changes",
        "partner_fail_avg_changes",
    ]

    print("\n[INFO] Перевірка наявності ключових колонок:")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("  [WARN] Відсутні очікувані колонки:", missing)
    else:
        print("  [INFO] Усі ключові колонки присутні в b2b.csv")

    print("\n[INFO] Грубий аналіз таргету та деяких фіч:")
    if "is_successful" in df.columns:
        print("  Розподіл is_successful:")
        print(df["is_successful"].value_counts(dropna=False))

    for col in ["order_amount", "order_messages", "order_changes",
                "partner_success_rate", "partner_total_orders"]:
        if col in df.columns:
            print(f"\n  Статистика для {col}:")
            print(df[col].describe())


if __name__ == "__main__":
    main()