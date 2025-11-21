# Мінімальний гайд: від SQL у pgAdmin до `b2b.csv` (тільки необхідні поля)

Цей документ описує **мінімально необхідний** набір даних та кроків, щоб отримати `b2b.csv`, який вже успішно використовується модулями:

- `order_partner_success_prediction`
- `smart_inventory_automation`

На відміну від повного гайду (`pgadmin_to_b2b_selfcontained_guide.md`), тут ми збираємо **лише ті поля, які реально потрапляють до `b2b.csv`**, без додаткових аналітичних ознак.

---

## 1. Які поля повинні бути в `b2b.csv`

Фактичний робочий `b2b.csv` (з `файл-для-smart_inventory_automation`) містить 19 колонок:

1. `order_id`
2. `is_successful`
3. `create_date`
4. `partner_id`
5. `order_amount`
6. `order_messages`
7. `order_changes`
8. `partner_success_rate`
9. `partner_total_orders`
10. `partner_order_age_days`
11. `partner_avg_amount`
12. `partner_success_avg_amount`
13. `partner_fail_avg_amount`
14. `partner_total_messages`
15. `partner_success_avg_messages`
16. `partner_fail_avg_messages`
17. `partner_avg_changes`
18. `partner_success_avg_changes`
19. `partner_fail_avg_changes`

Усі інші поля (`day_of_week`, `month`, `product_categories`, `payment_term`, `delivery_method`, `source` тощо) **не входять у робочий b2b‑формат** і не є обов’язковими для модулів.

---

## 2. Мінімальний SQL‑запит у pgAdmin для джерела даних

Мета SQL — зібрати тільки те, що потрібно для побудови цих 19 полів:

- на рівні замовлення: `order_id`, `create_date`, `date_order`, `partner_id`, `amount_total`, `messages_count`, `changes_count`, `state`;
- на рівні клієнта: `customer_id` (= `partner_id`), `partner_create_date`.

### 2.1. Мінімальний SQL‑запит

```sql
WITH base AS (
    SELECT
        so.id                                    AS order_db_id,
        so.name                                  AS order_id,
        so.create_date                           AS create_date,
        COALESCE(so.date_order, so.create_date)  AS date_order,
        rp.id                                    AS customer_id,
        rp.create_date                           AS partner_create_date,
        so.partner_id                            AS partner_id,
        so.amount_total                          AS total_amount,
        so.state                                 AS state
    FROM sale_order AS so
    JOIN res_partner AS rp       ON rp.id = so.partner_id
    -- За потреби додай фільтр по діапазону дат, наприклад:
    -- WHERE so.date_order::date BETWEEN DATE '2023-01-01' AND DATE '2023-12-31'
),
prev_orders AS (
    -- Кількість попередніх успішних замовлень по кожному конкретному замовленню
    -- Повністю повторює логіку Python-модуля data_collector:
    -- search_count([('partner_id', '=', order.partner_id.id),
    --               ('create_date', '<', order.create_date),
    --               ('state', 'in', ['sale', 'done'])])
    SELECT
        so1.id        AS order_db_id,
        COUNT(so2.id) AS previous_orders_count
    FROM sale_order AS so1
    LEFT JOIN sale_order AS so2
           ON  so2.partner_id = so1.partner_id
           AND so2.create_date < so1.create_date
           AND so2.state IN ('sale', 'done')
    GROUP BY so1.id
),
messages AS (
    -- Кількість повідомлень (аналог len(order.message_ids) з фільтрацією)
    SELECT
        mm.res_id,
        COUNT(*) AS messages_count
    FROM mail_message AS mm
    WHERE mm.model = 'sale.order'
      -- ВИКЛЮЧАЄМО технічні сповіщення (notification) та системні (user_notification)
      -- Залишаться тільки 'email' та 'comment' (коментарі користувачів)
      AND mm.message_type NOT IN ('notification', 'user_notification')
    GROUP BY mm.res_id
),
changes AS (
    -- Кількість повідомлень з трекінгом змін по кожному замовленню
    SELECT
        mm.res_id,
        COUNT(DISTINCT mm.id) AS changes_count
    FROM mail_message AS mm
    JOIN mail_tracking_value AS mtv
      ON mtv.mail_message_id = mm.id
    WHERE mm.model = 'sale.order'
    GROUP BY mm.res_id
)
SELECT
    b.order_id                                      AS order_id,
    b.create_date                                   AS create_date,
    b.date_order                                    AS date_order,
    b.customer_id                                   AS customer_id,
    (b.create_date::date - b.partner_create_date::date)
                                                    AS customer_relationship_days,
    COALESCE(po.previous_orders_count, 0)           AS previous_orders_count,
    b.total_amount                                  AS total_amount,
    COALESCE(ch.changes_count, 0)                   AS changes_count,
    COALESCE(msg.messages_count, 0)                 AS messages_count,
    b.state                                         AS state,
    b.partner_id                                    AS partner_id
FROM base AS b
LEFT JOIN prev_orders       AS po   ON po.order_db_id = b.order_db_id
LEFT JOIN messages          AS msg  ON msg.res_id = b.order_db_id
LEFT JOIN changes           AS ch   ON ch.res_id = b.order_db_id;
```

### 2.2. Що дає цей SQL

Результат містить тільки необхідні колонки для побудови `b2b`:

- `order_id`
- `create_date`
- `date_order`
- `customer_id` / `partner_id`
- `partner_create_date` (неявно через `customer_relationship_days`)
- `previous_orders_count`
- `total_amount`
- `messages_count`
- `changes_count`
- `state`

Усе інше (`delivery_method`, `payment_term`, `product_categories` тощо) навмисно опущено.

---

## 3. Як виконати запит у pgAdmin та зберегти мінімальний CSV

1. Відкрий pgAdmin і підключись до бази даних Odoo.
2. Відкрий **Query Tool** для потрібної БД.
3. Встав у вікно запитів **повний SQL із розділу 2.1**.
4. За потреби додай фільтр по даті в секції `WHERE` (в CTE `base`).
5. Натисни **Execute (F5)**.
6. Після завершення виконання:
   - у вкладці результатів натисни правою кнопкою миші → **Export Data…**;
   - у полі **Format** обери `CSV`;
   - вкажи шлях, наприклад:  
     `D:\DATABASES\Зібрані-дані\db1\extended_minimal_YYYY-MM-DD.csv`;
   - натисни **Export**.

Отриманий CSV будемо далі називати `extended_minimal.csv`.

---

## 4. Перетворення `extended_minimal.csv` у `b2b.csv`

### 4.1. Order‑рівень

У Python (логіка ідентична до повного гайду, але працює лише з мінімальним набором колонок):

1. `is_successful` з `state`:

   ```python
   successful_states = {"sale", "done"}
   df["is_successful"] = df["state"].astype(str).str.lower().isin(successful_states).astype(int)
   ```

2. `create_date` у форматі datetime (вже є у CSV, просто конвертація):

   ```python
   df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
   ```

3. Числові order‑поля:

   ```python
   df["order_amount"] = df["total_amount"].astype(float)
   df["order_messages"] = df.get("messages_count", 0)
   df["order_changes"] = df.get("changes_count", 0)
   ```

Отримаємо базовий набір для кожного замовлення:

- `order_id`, `is_successful`, `create_date`, `partner_id`, `order_amount`, `order_messages`, `order_changes`.

### 4.2. Агрегати по клієнту `partner_*`

Як і в повному гайді, але працюємо поверх `extended_minimal.csv`:

```python
orders = df[[
    "order_id",
    "partner_id",
    "is_successful",
    "order_amount",
    "order_messages",
    "order_changes",
    "create_date",
]].copy()

grp = orders.groupby("partner_id")

partner_total_orders   = grp["order_id"].count()
partner_total_messages = grp["order_messages"].sum()
partner_avg_amount     = grp["order_amount"].mean()
partner_avg_changes    = grp["order_changes"].mean()
first_order_date       = grp["create_date"].min()
last_order_date        = grp["create_date"].max()
partner_order_age_days = (last_order_date - first_order_date).dt.days

success = orders[orders["is_successful"] == 1].groupby("partner_id")
fail    = orders[orders["is_successful"] == 0].groupby("partner_id")

partner_success_orders        = success["order_id"].count()
partner_success_avg_amount    = success["order_amount"].mean()
partner_success_avg_messages  = success["order_messages"].mean()
partner_success_avg_changes   = success["order_changes"].mean()

partner_fail_orders           = fail["order_id"].count()
partner_fail_avg_amount       = fail["order_amount"].mean()
partner_fail_avg_messages     = fail["order_messages"].mean()
partner_fail_avg_changes      = fail["order_changes"].mean()

partner_stats = pd.DataFrame({
    "partner_total_orders": partner_total_orders,
    "partner_total_messages": partner_total_messages,
    "partner_avg_amount": partner_avg_amount,
    "partner_avg_changes": partner_avg_changes,
    "first_order_date": first_order_date,
    "last_order_date": last_order_date,
    "partner_success_orders": partner_success_orders,
    "partner_fail_orders": partner_fail_orders,
    "partner_success_avg_amount": partner_success_avg_amount,
    "partner_fail_avg_amount": partner_fail_avg_amount,
    "partner_success_avg_messages": partner_success_avg_messages,
    "partner_fail_avg_messages": partner_fail_avg_messages,
    "partner_success_avg_changes": partner_success_avg_changes,
    "partner_fail_avg_changes": partner_fail_avg_changes,
})

partner_stats["partner_success_orders"].fillna(0, inplace=True)
partner_stats["partner_total_orders"].fillna(0, inplace=True)

partner_stats["partner_success_rate"] = (
    partner_stats["partner_success_orders"] /
    partner_stats["partner_total_orders"].replace(0, pd.NA)
).fillna(0.0)

partner_stats["partner_order_age_days"] = (
    (partner_stats["last_order_date"] - partner_stats["first_order_date"])
    .dt.days.fillna(0).astype(int)
)

partner_stats.reset_index(inplace=True)  # partner_id стане окремою колонкою
```

### 4.3. Формування фінального `b2b`

```python
b2b = df.merge(partner_stats, on="partner_id", how="left")

b2b = b2b[[
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
]]

OUTPUT_B2B_PATH = r"D:\\DATABASES\\Зібрані-дані\\db1\\b2b_for_ml_YYYY-MM-DD.csv"

b2b.to_csv(OUTPUT_B2B_PATH, index=False)
```

---

## 5. Стислий чек‑лист для мінімального пайплайна

1. У pgAdmin виконати SQL із розділу 2.1 → експортувати у `extended_minimal_YYYY-MM-DD.csv`.
2. У Python:
   - завантажити `extended_minimal.csv`;
   - побудувати `is_successful`, `order_amount`, `order_messages`, `order_changes`;
   - по `partner_id` побудувати всі `partner_*` агрегати;
   - змерджити все в один датафрейм `b2b`;
   - зберегти як `b2b_for_ml_YYYY-MM-DD.csv`.
3. Використати `b2b_for_ml_*.csv` як вхід для модулів `order_partner_success_prediction` та `smart_inventory_automation`.
