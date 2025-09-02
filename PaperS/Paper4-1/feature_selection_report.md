# Академічний звіт з порівняння методів відбору ознак за AUC-PR

## Постановка задачі

У дослідженні проведено порівняння шести різних підходів до відбору ознак для задачі бінарної класифікації. Основною метрикою якості моделі обрано площу під PR-кривою (AUC-PR). Для кожного методу визначено оптимальний набір ознак, що забезпечує максимальне значення AUC-PR у діапазоні від 12 до 24 ознак. Аналіз проводився на одному й тому ж датасеті, що дозволяє об’єктивно порівняти ефективність різних стратегій відбору.

## Опис методів

У дослідженні розглянуто такі методи:
- Прямий XGBoost (відбір за важливістю ознак XGBoost)
- Зворотній XGBoost (зворотній відбір за важливістю XGBoost)
- Прямий Greedy (жадібний покроковий відбір)
- Зворотній Greedy (зворотній жадібний відбір)
- RFE (рекурсивне видалення ознак)
- Boruta (статистичний відбір на основі XGBoost)

## Порівняльна таблиця результатів

| Метод                  | Максимальний AUC-PR | Кількість ознак | Перелік ознак                                                                                                    |
|------------------------|---------------------|-----------------|------------------------------------------------------------------------------------------------------------------|
| Прямий XGBoost         | 0.97740             | 24              | order_messages, create_date_months, order_amount, partner_success_rate, order_changes, order_lines_count, source, partner_success_avg_amount, salesperson, partner_order_age_days, partner_total_orders, partner_success_avg_messages, partner_fail_avg_changes, partner_avg_amount, partner_total_messages, partner_avg_changes, partner_fail_avg_amount, partner_success_avg_changes, partner_fail_avg_messages, month, day_of_week, hour_of_day, discount_total, quarter |
| Зворотній XGBoost      | 0.97767             | 14              | partner_avg_amount, partner_fail_avg_changes, partner_success_avg_messages, partner_total_orders, partner_order_age_days, salesperson, partner_success_avg_amount, source, order_lines_count, order_changes, partner_success_rate, order_amount, create_date_months, order_messages |
| Прямий Greedy          | 0.97764             | 22              | order_messages, order_amount, create_date_months, order_changes, partner_success_avg_changes, order_lines_count, partner_success_avg_amount, partner_total_messages, partner_success_avg_messages, partner_order_age_days, salesperson, partner_avg_changes, source, day_of_week, partner_total_orders, partner_avg_amount, discount_total, partner_fail_avg_amount, partner_fail_avg_messages, quarter, partner_fail_avg_changes, month |
| Зворотній Greedy       | 0.97785             | 16              | partner_avg_changes, partner_total_messages, partner_avg_amount, partner_fail_avg_changes, partner_success_avg_messages, partner_total_orders, partner_order_age_days, salesperson, partner_success_avg_amount, source, order_lines_count, order_changes, partner_success_rate, order_amount, create_date_months, order_messages |
| RFE                    | 0.97752             | 16              | order_amount, order_messages, order_changes, partner_success_rate, partner_total_orders, partner_order_age_days, partner_success_avg_amount, partner_success_avg_messages, order_lines_count, create_date_months, source, salesperson, partner_fail_avg_changes, partner_total_messages, partner_avg_amount, partner_success_avg_changes |
| Boruta                 | 0.97712             | 19              | order_amount, order_messages, order_changes, partner_success_rate, partner_total_orders, partner_order_age_days, partner_avg_amount, partner_success_avg_amount, partner_fail_avg_amount, partner_total_messages, partner_success_avg_messages, partner_fail_avg_messages, partner_avg_changes, partner_success_avg_changes, partner_fail_avg_changes, order_lines_count, salesperson, source, create_date_months |

## Аналіз результатів

Найвищі значення AUC-PR отримано для зворотного Greedy, який досяг максимуму 0.97785 при використанні 16 ознак. Зворотній XGBoost показав другий результат — 0.97767 при 14 ознаках. Прямий XGBoost та прямий Greedy також показали високі результати, але для досягнення максимуму потребували більшої кількості ознак (24 та 22 відповідно). Метод RFE забезпечив максимальний AUC-PR 0.97752 при 16 ознаках, а Boruta — 0.97712 при 19 ознаках.

Варто відзначити, що оптимальні набори ознак для різних методів частково перетинаються, але не є ідентичними. Це свідчить про різну чутливість алгоритмів до структури даних та взаємозв’язків між ознаками. Методи зворотного відбору дозволяють досягти найкращого балансу між кількістю ознак та якістю моделі, що важливо для побудови інтерпретованих і компактних моделей.

## Висновки

Проведене дослідження демонструє, що для задачі бінарної класифікації на основі AUC-PR найбільш ефективними є методи зворотного відбору ознак, які дозволяють суттєво зменшити розмірність простору ознак без втрати якості. Водночас, класичні підходи (RFE, Boruta) також забезпечують конкурентоспроможні результати, але можуть потребувати більшої кількості ознак для досягнення аналогічної якості. 