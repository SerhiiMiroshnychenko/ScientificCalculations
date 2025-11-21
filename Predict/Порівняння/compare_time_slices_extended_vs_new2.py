import pandas as pd
from pathlib import Path


EXTENDED_PATH = r"extended_customer_data_2025-01-17.csv"
NEW_PATH      = r"new2.csv"


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    print(f"[INFO] Читаю файл: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Файл не знайдено: {p}")
    df = pd.read_csv(p)
    print(f"[INFO] {p.name}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    return df


def to_datetime_safe(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        print(f"[WARN] Колонки '{col}' немає у файлі")
        return pd.Series([pd.NaT] * len(df))
    s = pd.to_datetime(df[col], errors="coerce")
    return s


def print_time_slice(name: str, dates_create: pd.Series, dates_order: pd.Series):
    print(f"\n[INFO] Часовий зріз для {name}:")
    for label, s in [("create_date", dates_create), ("date_order", dates_order)]:
        if s.isna().all():
            print(f"  {label}: усі значення NaT / відсутні")
            continue
        print(f"  {label}:")
        print(f"    мін: {s.min()}")
        print(f"    макс: {s.max()}")
        print(f"    не-NaT значень: {s.notna().sum()}")


def main():
    df_ext = load_csv(EXTENDED_PATH)
    df_new = load_csv(NEW_PATH)

    # Переводимо create_date і date_order у datetime
    ext_create = to_datetime_safe(df_ext, "create_date")
    ext_order  = to_datetime_safe(df_ext, "date_order")

    new_create = to_datetime_safe(df_new, "create_date")
    new_order  = to_datetime_safe(df_new, "date_order")

    # Базова інформація по зрізах
    print_time_slice("extended_customer_data", ext_create, ext_order)
    print_time_slice("new2", new_create, new_order)

    # Порівняння діапазонів
    print("\n[INFO] Порівняння діапазонів дат (create_date):")
    print(f"  extended: мін={ext_create.min()}  макс={ext_create.max()}")
    print(f"  new2:     мін={new_create.min()}  макс={new_create.max()}")

    print("\n[INFO] Порівняння діапазонів дат (date_order):")
    print(f"  extended: мін={ext_order.min()}  макс={ext_order.max()}")
    print(f"  new2:     мін={new_order.min()}  макс={new_order.max()}")

    # Перевірка, чи є замовлення, які виходять за межі часових вікон одне одного
    print("\n[INFO] Аналіз виходу new2 за часові межі extended (create_date):")
    mask_new_before_ext_min = new_create < ext_create.min()
    mask_new_after_ext_max  = new_create > ext_create.max()
    print(f"  new2.create_date < min(extended): {mask_new_before_ext_min.sum()} рядків")
    print(f"  new2.create_date > max(extended): {mask_new_after_ext_max.sum()} рядків")

    print("\n[INFO] Аналіз виходу extended за часові межі new2 (create_date):")
    mask_ext_before_new_min = ext_create < new_create.min()
    mask_ext_after_new_max  = ext_create > new_create.max()
    print(f"  extended.create_date < min(new2): {mask_ext_before_new_min.sum()} рядків")
    print(f"  extended.create_date > max(new2): {mask_ext_after_new_max.sum()} рядків")

    print("\n[INFO] Аналіз завершено.")


if __name__ == "__main__":
    main()