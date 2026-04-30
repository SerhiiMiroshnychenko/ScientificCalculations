"""
Практична робота № 4
Варіант 20
Видаткова частина теплового балансу
"""

# Дано:
variant_no = 20

m_steel = 920.6
m_slag = 142.63
m_CO = 66.91
m_CO2 = 11.68
m_dust = 15
Q_in = 1721.48

t_tap = 1616
k_loss = 0.05

h_st_ref = 1250
Cp_st = 0.836
t_m = 1535

h_sl_ref = 209
Cp_sl = 0.9

t_gas = 1500
Cp_CO = 1.17
Cp_CO2 = 1.15

t_dust = t_gas
Cp_dust = 0.82

# Розрахунок:

## Крок 1: Ентальпія рідкої сталі + кількість тепла
h_steel = h_st_ref  + Cp_st * (t_tap - t_m) # кДж/кг
Q_steel = m_steel * h_steel / 1000 # МДж
print(f"{Q_steel = }")

## Крок 2: Ентальпія шлаку + кількість тепла
t_slag = t_tap + 50
h_slag = h_sl_ref + Cp_sl * t_slag # кДж/кг
Q_slag = m_slag * h_slag / 1000  # МДж
print(f"{Q_slag = }")

## Крок 3: Тепло відхідних газів
Q_gas = (m_CO * Cp_CO + m_CO2 * Cp_CO2) * t_gas / 1000 # МДж
print(f"{Q_gas = }")

## Крок 4: Ентальпія + тепло пилу
Q_dust = m_dust * Cp_dust * t_dust / 1000  # МДж
print(f"{Q_dust = }")

## Крок 5: Теплові втрати (футерівка + випромінювання)
Q_loss = k_loss * Q_in
print(f"{Q_loss = }")

## Крок 6: Сумарний видаток та тепловий надлишок
Q_out = Q_steel + Q_slag + Q_gas + Q_dust + Q_loss
print(f"{Q_out = }")

delta_Q = Q_in - Q_out
print(f"{delta_Q = }")

share_steel_pct = Q_steel / Q_out * 100
print(f"{share_steel_pct = }")

# =========================================================================
# 7) Вивід результатів
# =========================================================================
print("=" * 65)
print(f"  ПРАКТИЧНА РОБОТА №4: ВИДАТОК ТЕПЛА (ВАРІАНТ №{variant_no})")
print("=" * 65)
print(f"  1) Ентальпія сталі   h = {h_steel:.2f} кДж/кг  ->  {Q_steel:.2f} МДж")
print(f"  2) Ентальпія шлаку   h = {h_slag:.2f} кДж/кг  ->  {Q_slag:.2f} МДж")
print(f"  3) Тепло газів       (t_gas = {t_gas:.0f} °C)       {Q_gas:.2f} МДж")
print(f"  4) Тепло пилу        (t_dust = {t_dust:.0f} °C)      {Q_dust:.2f} МДж")
print(f"  5) Теплові втрати    (k = {k_loss:.3f} = {k_loss*100:.2f}%)     {Q_loss:.2f} МДж")
print("-" * 65)
print(f"  СУМАРНИЙ ВИДАТОК  Q_out  =  {Q_out:.2f} МДж")
print(f"  ЗАГАЛЬНИЙ ПРИХІД  Q_in   =  {Q_in:.2f} МДж")
print(f"  ТЕПЛОВИЙ НАДЛИШОК dQ     =  {delta_Q:+.2f} МДж")
print("=" * 65)

print(f"\n  Частка сталі у видатку: {share_steel_pct:.1f}%", end="  ")
if 70.0 <= share_steel_pct <= 80.0:
    print("[OK: 70–80%]")
else:
    print("[УВАГА: поза діапазоном 70–80% — перевірте формули]")

print()
if delta_Q > 0:
    print(f"  РЕЗУЛЬТАТ: Надлишок {delta_Q:.2f} МДж передається до ПР №5.")
    print("  Розрахунок маси охолоджувача (брухту/руди) — наступний крок.")
elif delta_Q == 0:
    print("  РЕЗУЛЬТАТ: Баланс точно замкнений. dQ = 0.")
else:
    print("  УВАГА: Тепловий дефіцит! Перевірте вхідні дані та формули.")
print("=" * 65)
