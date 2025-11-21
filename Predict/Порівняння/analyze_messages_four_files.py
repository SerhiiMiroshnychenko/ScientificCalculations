import pandas as pd
from pathlib import Path

# Raw Files
RAW1_PATH = Path(r"extended_customer_data_2025-01-17.csv")
RAW2_PATH = Path(r"db1.csv")

# Processed Files
PROC1_PATH = Path(r"b2b.csv")
PROC2_PATH = Path(r"b2b_for_ml.csv")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    print(f"[INFO] Читаю {label} з {path}...")
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Файл не знайдено: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] {label}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    return df


def normalize_order_id_numeric(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Повертає копію df з доданою колонкою order_id_norm.

    * Для RAW-файлів (label починається з "RAW") завжди беремо числову частину з рядкового order_id (SOxxxxx → xxxxx).
    * Для PROC-файлів (label починається з "PROC") вважаємо, що order_id вже містить офсет 1_000_000 і віднімаємо його.
    """
    out = df.copy()

    if "order_id" not in out.columns:
        print(f"[WARN] У {label} немає order_id")
        return out

    label_upper = label.upper()

    if label_upper.startswith("RAW"):
        # Сирі дані: order_id вигляду SOxxxxx → дістаємо числову частину
        numeric_part = out["order_id"].astype(str).str.extract(r"(\d+)", expand=False)
        out["order_id_norm"] = pd.to_numeric(numeric_part, errors="coerce")
        print(f"[INFO] {label}: order_id_norm побудований з рядкового order_id (SOxxxxx → xxxxx)")
    elif label_upper.startswith("PROC"):
        # Оброблені дані: order_id вже зі зсувом 1_000_000
        out["order_id_norm"] = pd.to_numeric(out["order_id"], errors="coerce") - 1_000_000
        print(f"[INFO] {label}: order_id_norm = order_id - 1_000_000")
    else:
        # Фолбек: намагаємось витягнути числову частину, якщо є, інакше віднімаємо офсет
        numeric_part = out["order_id"].astype(str).str.extract(r"(\d+)", expand=False)
        tmp = pd.to_numeric(numeric_part, errors="coerce")
        if tmp.notna().any():
            out["order_id_norm"] = tmp
            print(f"[INFO] {label}: order_id_norm побудований з рядкового order_id (fallback)")
        else:
            out["order_id_norm"] = pd.to_numeric(out["order_id"], errors="coerce") - 1_000_000
            print(f"[INFO] {label}: order_id_norm = order_id - 1_000_000 (fallback)")

    return out


def main() -> None:
    print("[INFO] === АНАЛІЗ ПОВІДОМЛЕНЬ У 4 ФАЙЛАХ (RAW/PROC) ===")
    print(f"[INFO] RAW1_PATH  = {RAW1_PATH}")
    print(f"[INFO] RAW2_PATH  = {RAW2_PATH}")
    print(f"[INFO] PROC1_PATH = {PROC1_PATH}")
    print(f"[INFO] PROC2_PATH = {PROC2_PATH}")

    raw1 = load_csv(RAW1_PATH, "RAW1 (extended)")
    raw2 = load_csv(RAW2_PATH, "RAW2 (db1)")
    proc1 = load_csv(PROC1_PATH, "PROC1 (b2b)")
    proc2 = load_csv(PROC2_PATH, "PROC2 (b2b_for_ml)")

    # Підготуємо order_id_norm
    raw1_n = normalize_order_id_numeric(raw1, "RAW1 (extended)")
    raw2_n = normalize_order_id_numeric(raw2, "RAW2 (db1)")
    proc1_n = normalize_order_id_numeric(proc1, "PROC1 (b2b)")
    proc2_n = normalize_order_id_numeric(proc2, "PROC2 (b2b_for_ml)")

    # Діагностика по order_id_norm
    for df_norm, label in (
        (raw1_n, "RAW1"),
        (raw2_n, "RAW2"),
        (proc1_n, "PROC1"),
        (proc2_n, "PROC2"),
    ):
        if "order_id_norm" in df_norm.columns:
            total = df_norm.shape[0]
            non_null = df_norm["order_id_norm"].notna().sum()
            print(f"[INFO] {label}: order_id_norm non-null = {non_null}/{total}")

    # Витягуємо колонки з кількістю повідомлень
    # RAW: messages_count; PROC: order_messages
    cols = {}
    if "messages_count" in raw1_n.columns:
        part = raw1_n[["order_id_norm", "messages_count"]].dropna(subset=["order_id_norm"])
        cols["messages_raw1"] = part.rename(columns={"messages_count": "messages_raw1"})
    else:
        print("[WARN] У RAW1 немає messages_count")

    if "messages_count" in raw2_n.columns:
        part = raw2_n[["order_id_norm", "messages_count"]].dropna(subset=["order_id_norm"])
        cols["messages_raw2"] = part.rename(columns={"messages_count": "messages_raw2"})
    else:
        print("[WARN] У RAW2 немає messages_count")

    if "order_messages" in proc1_n.columns:
        part = proc1_n[["order_id_norm", "order_messages"]].dropna(subset=["order_id_norm"])
        cols["messages_proc1"] = part.rename(columns={"order_messages": "messages_proc1"})
    else:
        print("[WARN] У PROC1 немає order_messages")

    if "order_messages" in proc2_n.columns:
        part = proc2_n[["order_id_norm", "order_messages"]].dropna(subset=["order_id_norm"])
        cols["messages_proc2"] = part.rename(columns={"order_messages": "messages_proc2"})
    else:
        print("[WARN] У PROC2 немає order_messages")

    # Зводимо всі 4 шари по order_id_norm
    merged = None
    for key, df_part in cols.items():
        if merged is None:
            merged = df_part
        else:
            # Використовуємо inner-join по order_id_norm, щоб уникнути гігантського outer join по NaN
            merged = merged.merge(df_part, on="order_id_norm", how="inner")

    if merged is None:
        print("[ERROR] Не вдалося побудувати merged по messages_*")
        return

    print(f"[INFO] merged shape: {merged.shape}")

    # Базова статистика
    print("\n=== БАЗОВА СТАТИСТИКА ПО ПОВІДОМЛЕННЯХ ===")
    print(merged[[c for c in merged.columns if c.startswith("messages_")]].describe())

    # Порівняння попарно
    def compare_pair(col_a: str, col_b: str, label: str) -> None:
        if col_a not in merged.columns or col_b not in merged.columns:
            print(f"[WARN] Для {label} немає однієї з колонок: {col_a}, {col_b}")
            return
        a = merged[col_a]
        b = merged[col_b]
        mask_valid = ~(a.isna() & b.isna())
        a = a[mask_valid]
        b = b[mask_valid]
        total = len(a)
        if total == 0:
            print(f"\n=== ПОРІВНЯННЯ {label} ===")
            print("Валідних пар: 0 (немає що порівнювати)")
            return
        equal = (a == b)
        n_equal = int(equal.sum())
        print(f"\n=== ПОРІВНЯННЯ {label} ===")
        print(f"Валідних пар: {total}")
        print(f"Збігаються:  {n_equal} ({n_equal/total:.4f})")
        # Розподіл різниці (a-b)
        diff = (a - b).dropna()
        print("Розподіл різниці (a - b), кілька значень:")
        print(diff.value_counts().head(10))

    compare_pair("messages_raw1", "messages_raw2", "RAW1 vs RAW2")
    compare_pair("messages_raw1", "messages_proc1", "RAW1 vs PROC1 (b2b)")
    compare_pair("messages_raw1", "messages_proc2", "RAW1 vs PROC2 (b2b_for_ml)")
    compare_pair("messages_raw2", "messages_proc1", "RAW2 vs PROC1 (b2b)")
    compare_pair("messages_raw2", "messages_proc2", "RAW2 vs PROC2 (b2b_for_ml)")
    compare_pair("messages_proc1", "messages_proc2", "PROC1 (b2b) vs PROC2 (b2b_for_ml)")

    # Приклади розбіжностей для найтиповішої пари RAW2 vs PROC2
    if "messages_raw2" in merged.columns and "messages_proc2" in merged.columns:
        print("\n=== ПРИКЛАДИ РОЗБІЖНОСТЕЙ RAW2 (db1) vs PROC2 (b2b_for_ml) ===")
        sub = merged.dropna(subset=["messages_raw2", "messages_proc2"]).copy()
        sub["diff"] = sub["messages_proc2"] - sub["messages_raw2"]
        diffs = sub[sub["diff"] != 0]
        print(f"Усього рядків з diff!=0: {diffs.shape[0]}")
        if not diffs.empty:
            print(diffs.head(20).to_string(index=False))

    print("[INFO] === ЗАВЕРШЕНО АНАЛІЗ ПОВІДОМЛЕНЬ У 4 ФАЙЛАХ ===")


if __name__ == "__main__":
    main()
