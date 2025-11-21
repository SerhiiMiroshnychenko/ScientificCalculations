import pandas as pd
from pathlib import Path

# ==========================================
# НАЛАШТУВАННЯ ШЛЯХІВ
# ==========================================
# NEW_B2B_PATH — новий файл з build_b2b_from_pgadmin_export.py
# OLD_B2B_PATH — старий робочий b2b, який використовують модулі Odoo

NEW_B2B_PATH = Path(r"b2b_for_ml.csv")
OLD_B2B_PATH = Path(r"b2b.csv")


def load_b2b(path: Path, label: str) -> pd.DataFrame:
    print(f"[INFO] Читаю {label} з {path}...")
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Файл не знайдено: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] {label}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    return df


def analyze_id_column(df: pd.DataFrame, col: str, label: str) -> None:
    print(f"\n=== АНАЛІЗ КОЛОНКИ {col} ({label}) ===")
    if col not in df.columns:
        print(f"[WARN] У {label} немає колонки {col}")
        return

    s = df[col]
    print(f"Тип даних: {s.dtype}")
    print(f"Кількість унікальних значень: {s.nunique()}")

    # Показати перші 10 значень
    print("Перші 10 значень:")
    print(s.head(10).to_list())

    # Якщо числова — показати min/max
    if pd.api.types.is_numeric_dtype(s):
        print(f"min = {s.min()}, max = {s.max()}")
    else:
        # Спроба виділити числову частину (для order_id типу SO0001)
        try:
            numeric_part = (
                s.astype(str)
                 .str.extract(r"(\d+)", expand=False)
                 .dropna()
                 .astype(int)
            )
            if not numeric_part.empty:
                print("Числова частина (order_id), на основі перших 10 значень:")
                print(numeric_part.head(10).to_list())
                print(f"numeric min = {numeric_part.min()}, numeric max = {numeric_part.max()}")
        except Exception as e:
            print(f"[WARN] Не вдалося виділити числову частину з {col}: {e}")


def analyze_partner_id_shift(df_new: pd.DataFrame, df_old: pd.DataFrame) -> None:
    """Спроба оцінити можливий зсув (offset) між partner_id у NEW та OLD.

    Наприклад, якщо до partner_id у старому файлі додавали 1_000_000 для db1,
    то після сортування множин унікальних partner_id різниця old - new може бути сталою.
    """
    print("\n=== АНАЛІЗ МОЖЛИВОГО ЗСУВУ ДЛЯ partner_id ===")

    if "partner_id" not in df_new.columns or "partner_id" not in df_old.columns:
        print("[WARN] Немає partner_id в одному з файлів")
        return

    s_new = df_new["partner_id"]
    s_old = df_old["partner_id"]

    if not (pd.api.types.is_numeric_dtype(s_new) and pd.api.types.is_numeric_dtype(s_old)):
        print("[WARN] partner_id не є числовим у одному з файлів — пропускаю аналіз зсуву")
        return

    uniq_new = sorted(s_new.unique())
    uniq_old = sorted(s_old.unique())

    print(f"Кількість унікальних partner_id NEW: {len(uniq_new)}")
    print(f"Кількість унікальних partner_id OLD: {len(uniq_old)}")

    if len(uniq_new) != len(uniq_old):
        print("[INFO] Різна кількість унікальних partner_id —")
        print("      просте зіставлення 1:1 може бути некоректним.")

    # Візьмемо мінімальну довжину для зіставлення
    n = min(len(uniq_new), len(uniq_old))
    if n == 0:
        print("[INFO] Порожні множини partner_id, немає що аналізувати.")
        return

    # Обмежимося підмножиною для швидкості
    MAX_SAMPLE = 10000
    n_sample = min(n, MAX_SAMPLE)

    new_sample = pd.Series(uniq_new[:n_sample])
    old_sample = pd.Series(uniq_old[:n_sample])

    diff = old_sample - new_sample
    print("Перші 10 різниць old - new по відсортованих унікальних partner_id:")
    print(diff.head(10).to_list())

    # Якщо всі різниці однакові — є сталий зсув
    if (diff.nunique() == 1):
        offset = diff.iloc[0]
        print(f"[INFO] Виглядає, що існує сталий зсув partner_id: old = new + {offset}")
    else:
        print("[INFO] Різниці old - new не сталi, явного одного зсуву не виявлено.")


def main() -> None:
    print("[INFO] === ПОЧАТОК АНАЛІЗУ ID В B2B-ФАЙЛАХ ===")
    print(f"[INFO] NEW_B2B_PATH = {NEW_B2B_PATH}")
    print(f"[INFO] OLD_B2B_PATH = {OLD_B2B_PATH}")

    df_new = load_b2b(NEW_B2B_PATH, "NEW (b2b_for_ml)")
    df_old = load_b2b(OLD_B2B_PATH, "OLD (b2b)")

    # Аналіз partner_id
    analyze_id_column(df_new, "partner_id", "NEW (b2b_for_ml)")
    analyze_id_column(df_old, "partner_id", "OLD (b2b)")

    # Аналіз order_id
    analyze_id_column(df_new, "order_id", "NEW (b2b_for_ml)")
    analyze_id_column(df_old, "order_id", "OLD (b2b)")

    # Спроба оцінити зсув між partner_id у NEW та OLD
    analyze_partner_id_shift(df_new, df_old)

    print("[INFO] === ЗАВЕРШЕНО АНАЛІЗ ID В B2B-ФАЙЛАХ ===")


if __name__ == "__main__":
    main()
