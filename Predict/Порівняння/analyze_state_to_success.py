import pandas as pd
from pathlib import Path

DB1_PATH = Path(r"db1.csv")
NEW_B2B_PATH = Path(r"b2b_for_ml.csv")
OLD_B2B_PATH = Path(r"b2b.csv")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    print(f"[INFO] Читаю {label} з {path}...")
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Файл не знайдено: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] {label}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    return df


def analyze_state_mapping() -> None:
    df_db1 = load_csv(DB1_PATH, "DB1 (db1.csv)")
    df_new = load_csv(NEW_B2B_PATH, "NEW (b2b_for_ml)")
    df_old = load_csv(OLD_B2B_PATH, "OLD (b2b)")

    if "state" not in df_db1.columns:
        print("[ERROR] У db1.csv немає колонки state")
        return

    # order_id_norm = numeric(order_id) для db1 (SOxxxxx), для b2b вже числовий зі зсувом
    db1 = df_db1.copy()
    db1_id = db1["order_id"].astype(str).str.extract(r"(\d+)", expand=False)
    db1["order_id_norm"] = pd.to_numeric(db1_id, errors="coerce")

    new = df_new.copy()
    old = df_old.copy()

    # Нормалізуємо order_id у NEW/OLD (віднімаємо 1_000_000)
    for df, label in ((new, "NEW"), (old, "OLD")):
        if "order_id" not in df.columns:
            print(f"[WARN] У {label} немає order_id")
            return
        df["order_id_norm"] = pd.to_numeric(df["order_id"], errors="coerce") - 1_000_000

    # Зв'язуємо db1.state з is_successful_new та is_successful_old
    merged = db1[["order_id_norm", "state"]].merge(
        new[["order_id_norm", "is_successful"]],
        on="order_id_norm",
        how="left",
        suffixes=("", "_new"),
    ).merge(
        old[["order_id_norm", "is_successful"]].rename(columns={"is_successful": "is_successful_old"}),
        on="order_id_norm",
        how="left",
    )

    print("\n=== УНІКАЛЬНІ state В DB1 ===")
    print(merged["state"].value_counts(dropna=False))

    print("\n=== МАПІНГ state → is_successful_new (наш скрипт) ===")
    mapping_new = (
        merged.groupby("state")["is_successful"]
        .value_counts(dropna=False)
        .unstack(fill_value=0)
        .rename(columns={0: "is_successful=0", 1: "is_successful=1"})
    )
    print(mapping_new)

    print("\n=== МАПІНГ state → is_successful_old (старий b2b.csv) ===")
    if "is_successful_old" in merged.columns:
        mapping_old = (
            merged.groupby("state")["is_successful_old"]
            .value_counts(dropna=False)
            .unstack(fill_value=0)
            .rename(columns={0: "is_successful_old=0", 1: "is_successful_old=1"})
        )
        print(mapping_old)
    else:
        print("[WARN] У merged немає is_successful_old")


def main() -> None:
    print("[INFO] === АНАЛІЗ МАПІНГУ state → is_successful У B2B ===")
    print(f"[INFO] DB1_PATH      = {DB1_PATH}")
    print(f"[INFO] NEW_B2B_PATH  = {NEW_B2B_PATH}")
    print(f"[INFO] OLD_B2B_PATH  = {OLD_B2B_PATH}")

    analyze_state_mapping()

    print("[INFO] === ЗАВЕРШЕНО АНАЛІЗ МАПІНГУ state → is_successful ===")


if __name__ == "__main__":
    main()
