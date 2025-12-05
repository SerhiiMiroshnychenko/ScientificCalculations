import pandas as pd
import numpy as np
from pathlib import Path
import re


# ==========================================
# НАЛАШТУВАННЯ ШЛЯХІВ ВХІД/ВИХІД
# ==========================================
# INPUT_CSV  — мінімальний CSV з pgAdmin (dbN.csv) за нашим SQL
# OUTPUT_CSV — готовий b2b-файл для модулів Odoo

INPUT_CSV = Path(r"db8.csv")
OUTPUT_CSV = Path(r"db8_for_ml.csv")


def load_orders(path: Path) -> pd.DataFrame:
    """Завантажує мінімальний CSV з pgAdmin з базовим логуванням."""
    print(f"[INFO] Читаю CSV з pgAdmin: {path}...")
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Файл не знайдено: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Файл {path.name}: {df.shape[0]} рядків, {df.shape[1]} колонок")
    print(f"[INFO] Перші колонки: {list(df.columns[:15])}")
    return df


def get_db_offset(path: Path) -> int:
    """Визначає зсув ID за номером бази в імені файлу.

    db1 → 1 * 1_000_000
    db2 → 2 * 1_000_000
    ...

    Якщо цифр немає — повертаємо 0.
    """
    name = path.stem  # наприклад, "db1"
    m = re.search(r"(\d+)", name)
    if not m:
        print(f"[WARN] У назві файлу {name} не знайдено номер бази, зсув ID = 0")
        return 0
    db_index = int(m.group(1))
    offset = db_index * 1_000_000
    print(f"[INFO] Визначено номер бази db_index={db_index}, зсув ID = {offset}")
    return offset


def add_order_level_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Додає order-рівень: is_successful, create_date, order_amount, order_messages, order_changes."""
    required = [
        "order_id",
        "partner_id",
        "state",
        "total_amount",
        "messages_count",
        "changes_count",
        "order_lines",
        "create_date",
        "date_order",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"У вхідному CSV бракує обов'язкових колонок: {missing}")

    df = df.copy()

    # Ознака успішності замовлення
    successful_states = {"sale", "done"}
    df["is_successful"] = (
        df["state"].astype(str).str.lower().isin(successful_states).astype(int)
    )

    # Дата створення/підтвердження
    df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
    # date_order у мінімальному CSV вже є, але для надійності також приведемо до datetime
    df["date_order"] = pd.to_datetime(df["date_order"], errors="coerce")

    # Фінансові та активністні характеристики
    df["order_amount"] = df["total_amount"].astype(float)
    df["order_messages"] = pd.to_numeric(df["messages_count"], errors="coerce").fillna(0).astype(float)
    df["order_changes"] = pd.to_numeric(df["changes_count"], errors="coerce").fillna(0).astype(float)
    df["order_lines"] = pd.to_numeric(df["order_lines"], errors="coerce").fillna(0).astype(float)

    return df


def build_historical_partner_features(df: pd.DataFrame) -> pd.DataFrame:
    """Будує історичні partner_* агрегати станом на момент КОЖНОГО замовлення.

    ВАЖЛИВО: поточний ордер НЕ входить до історії.
    Усі метрики рахуються тільки по попередніх ордерах партнера.
    """

    df = df.copy()

    # Сортуємо всередині партнера за часом, щоб кумулятивні метрики були "на момент ордера".
    df.sort_values(["partner_id", "create_date", "order_id"], inplace=True)

    grp = df.groupby("partner_id", group_keys=False)

    # Кількість попередніх замовлень (поточний не враховується)
    df["partner_total_orders"] = grp.cumcount()

    # Кумулятивні суми по всіх замовленнях, зсунутих на 1 (щоб виключити поточний)
    cum_amount_all = grp["order_amount"].cumsum()
    cum_changes_all = grp["order_changes"].cumsum()
    cum_messages_all = grp["order_messages"].cumsum()

    df["partner_total_messages"] = (cum_messages_all - df["order_messages"]).astype(float)
    df["_amount_prev_all"] = (cum_amount_all - df["order_amount"]).astype(float)
    df["_changes_prev_all"] = (cum_changes_all - df["order_changes"]).astype(float)

    # Середні по всіх попередніх замовленнях
    df["partner_avg_amount"] = (
        df["_amount_prev_all"] / df["partner_total_orders"].replace(0, np.nan)
    ).fillna(0.0)

    df["partner_avg_changes"] = (
        df["_changes_prev_all"] / df["partner_total_orders"].replace(0, np.nan)
    ).fillna(0.0)

    # Маски успішних/неуспішних ордерів
    success_mask = df["is_successful"] == 1
    fail_mask = ~success_mask

    # Флаги успіху/невдачі для поточного ордера
    df["_success_flag"] = success_mask.astype(int)
    df["_fail_flag"] = fail_mask.astype(int)

    # Кумулятивні кількості успішних/неуспішних + зсув (виключаємо поточний)
    cum_success_orders = grp["_success_flag"].cumsum()
    cum_fail_orders = grp["_fail_flag"].cumsum()

    df["partner_success_orders"] = (cum_success_orders - df["_success_flag"]).astype(float)
    df["partner_fail_orders"] = (cum_fail_orders - df["_fail_flag"]).astype(float)

    # Кумулятивні суми по успішних/неуспішних (сума тільки попередніх)
    df["_success_amount"] = df["order_amount"].where(success_mask, 0.0)
    df["_fail_amount"] = df["order_amount"].where(fail_mask, 0.0)

    df["_success_messages"] = df["order_messages"].where(success_mask, 0.0)
    df["_fail_messages"] = df["order_messages"].where(fail_mask, 0.0)

    df["_success_changes"] = df["order_changes"].where(success_mask, 0.0)
    df["_fail_changes"] = df["order_changes"].where(fail_mask, 0.0)

    cum_success_amount = grp["_success_amount"].cumsum()
    cum_fail_amount = grp["_fail_amount"].cumsum()

    cum_success_messages = grp["_success_messages"].cumsum()
    cum_fail_messages = grp["_fail_messages"].cumsum()

    cum_success_changes = grp["_success_changes"].cumsum()
    cum_fail_changes = grp["_fail_changes"].cumsum()

    df["_amount_prev_success"] = (cum_success_amount - df["_success_amount"]).astype(float)
    df["_amount_prev_fail"] = (cum_fail_amount - df["_fail_amount"]).astype(float)

    df["_messages_prev_success"] = (cum_success_messages - df["_success_messages"]).astype(float)
    df["_messages_prev_fail"] = (cum_fail_messages - df["_fail_messages"]).astype(float)

    df["_changes_prev_success"] = (cum_success_changes - df["_success_changes"]).astype(float)
    df["_changes_prev_fail"] = (cum_fail_changes - df["_fail_changes"]).astype(float)

    # Середні по успішних/неуспішних (тільки попередні ордери)
    df["partner_success_avg_amount"] = (
        df["_amount_prev_success"] / df["partner_success_orders"].replace(0, np.nan)
    ).fillna(0.0)

    df["partner_fail_avg_amount"] = (
        df["_amount_prev_fail"] / df["partner_fail_orders"].replace(0, np.nan)
    ).fillna(0.0)

    df["partner_success_avg_messages"] = (
        df["_messages_prev_success"] / df["partner_success_orders"].replace(0, np.nan)
    ).fillna(0.0)

    df["partner_fail_avg_messages"] = (
        df["_messages_prev_fail"] / df["partner_fail_orders"].replace(0, np.nan)
    ).fillna(0.0)

    df["partner_success_avg_changes"] = (
        df["_changes_prev_success"] / df["partner_success_orders"].replace(0, np.nan)
    ).fillna(0.0)

    df["partner_fail_avg_changes"] = (
        df["_changes_prev_fail"] / df["partner_fail_orders"].replace(0, np.nan)
    ).fillna(0.0)

    # Успішність у % на момент ордера (враховуються тільки попередні ордери)
    df["partner_success_rate"] = (
        df["partner_success_orders"] /
        df["partner_total_orders"].replace(0, np.nan)
    ).fillna(0.0) * 100.0

    # "Вік" історії замовлень партнера на момент ордера
    first_order_date = grp["create_date"].transform("min")
    df["partner_order_age_days"] = (
        (df["create_date"] - first_order_date)
        .dt.days.fillna(0).astype(int)
    )

    # Прибираємо технічні колонки, які використовувалися тільки для проміжних розрахунків
    drop_cols = [
        "_amount_prev_all",
        "_changes_prev_all",
        "_success_flag",
        "_fail_flag",
        "_success_amount",
        "_fail_amount",
        "_success_messages",
        "_fail_messages",
        "_success_changes",
        "_fail_changes",
        "_amount_prev_success",
        "_amount_prev_fail",
        "_messages_prev_success",
        "_messages_prev_fail",
        "_changes_prev_success",
        "_changes_prev_fail",
    ]
    df.drop(columns=drop_cols, inplace=True)

    return df


def apply_id_offset(df: pd.DataFrame, id_offset: int) -> pd.DataFrame:
    """Застосовує зсув ID до partner_id та order_id згідно зі схемою dbN.

    - partner_id: просто +offset
    - order_id: беремо числову частину з рядка типу "SO001" і додаємо offset,
      зберігаючи як ціле число.
    """
    df = df.copy()

    if id_offset == 0:
        return df

    if "partner_id" in df.columns:
        df["partner_id"] = pd.to_numeric(df["partner_id"], errors="coerce").fillna(0).astype("int64") + id_offset

    if "order_id" in df.columns:
        order_str = df["order_id"].astype(str)
        numeric_part = order_str.str.extract(r"(\d+)", expand=False)
        mask = numeric_part.notna()
        if mask.any():
            numeric_part = numeric_part[mask].astype('int64') + id_offset
            df.loc[mask, "order_id"] = numeric_part.astype("int64")

    return df


def build_b2b(df_orders: pd.DataFrame, id_offset: int) -> pd.DataFrame:
    """Повний пайплайн побудови b2b з мінімального CSV."""

    df = add_order_level_columns(df_orders)
    df = build_historical_partner_features(df)
    df = apply_id_offset(df, id_offset)

    # Залишаємо рівно 19 колонок у потрібному порядку
    b2b = df[[
        "order_id",
        "is_successful",
        "create_date",
        "partner_id",
        "order_amount",
        "order_messages",
        "order_changes",
        "order_lines",
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
    ]].copy()

    return b2b


def main() -> None:
    print("[INFO] === ПОЧАТОК ПОБУДОВИ ІСТОРИЧНОГО B2B З CSV PGADMIN ===")
    print(f"[INFO] INPUT_CSV  = {INPUT_CSV}")
    print(f"[INFO] OUTPUT_CSV = {OUTPUT_CSV}")

    df_orders = load_orders(INPUT_CSV)
    id_offset = get_db_offset(INPUT_CSV)

    print("[INFO] Формую історичний b2b-датафрейм...")
    b2b_df = build_b2b(df_orders, id_offset=id_offset)

    print(f"[INFO] Розмір b2b: {b2b_df.shape[0]} рядків, {b2b_df.shape[1]} колонок")

    if "is_successful" in b2b_df.columns:
        print("[INFO] Розподіл is_successful:")
        print(b2b_df["is_successful"].value_counts(dropna=False))

    print("[INFO] Перші 5 рядків b2b:")
    print(b2b_df.head())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    b2b_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] B2B-датасет збережено до: {OUTPUT_CSV}")
    print("[INFO] === ЗАВЕРШЕНО ПОБУДОВУ B2B ===")


if __name__ == "__main__":
    main()
