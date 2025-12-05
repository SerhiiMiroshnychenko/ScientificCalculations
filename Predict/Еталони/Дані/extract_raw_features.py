import pandas as pd
from pathlib import Path

# Конфігурація шляхів
INPUT_FILE = Path(r"db8_for_ml.csv")
OUTPUT_FILE = Path(r"db8_raw_only.csv")


def main():
    print(f"[INFO] Loading {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print(f"[ERROR] File {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"[INFO] Original shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Визначаємо колонки, які є "сирими" даними з Odoo (або прямими агрегатами по замовленню)
    # is_successful - це цільова змінна (target), її залишаємо
    raw_columns = [
        'order_id',
        'create_date',
        'partner_id',
        'order_amount',  # amount_total
        'order_messages',  # messages_count
        'order_changes',  # changes_count
        'order_lines',  # lines_count
        'is_successful'  # target derived from state
    ]

    # Перевіряємо, чи всі колонки є в датафреймі
    existing_cols = [c for c in raw_columns if c in df.columns]

    if len(existing_cols) < len(raw_columns):
        missing = set(raw_columns) - set(existing_cols)
        print(f"[WARN] Missing columns: {missing}")

    # Створюємо новий датафрейм тільки з вибраними колонками
    df_raw = df[existing_cols].copy()

    print(f"\n[INFO] Extracted raw features shape: {df_raw.shape}")
    print(f"[INFO] Extracted columns: {list(df_raw.columns)}")

    # Зберігаємо
    df_raw.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
