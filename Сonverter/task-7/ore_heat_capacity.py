# Імпортуємо необхідні бібліотеки
import numpy as np

# Вхідні дані
# Масові частки компонентів, %
x_fe2o3 = 34.1  # Fe2O3
x_h2o = 7.5     # H2O
x_sio2 = 58.4   # SiO2 (порожня порода)

# Теплоємності компонентів, кДж/(кг·°С)
c_fe2o3 = 0.61  # Fe2O3
c_h2o = 4.2     # H2O
c_sio2 = 1.17   # SiO2

print("1. Вхідні дані:")
print("   Масові частки компонентів, %:")
print(f"   Fe2O3: {x_fe2o3}")
print(f"   H2O: {x_h2o}")
print(f"   SiO2: {x_sio2}")
print("\n   Теплоємності компонентів, кДж/(кг·°С):")
print(f"   Fe2O3: {c_fe2o3}")
print(f"   H2O: {c_h2o}")
print(f"   SiO2: {c_sio2}")

# Переведення масових часток у десятковий вигляд
x_fe2o3_dec = x_fe2o3/100
x_h2o_dec = x_h2o/100
x_sio2_dec = x_sio2/100

print("\n2. Масові частки у десятковому вигляді:")
print(f"   Fe2O3: {x_fe2o3_dec:.4f}")
print(f"   H2O: {x_h2o_dec:.4f}")
print(f"   SiO2: {x_sio2_dec:.4f}")

# Перевірка суми масових часток
x_sum = x_fe2o3_dec + x_h2o_dec + x_sio2_dec
print(f"\n3. Сума масових часток: {x_sum:.4f}")

# Розрахунок питомої теплоємності суміші
c_mix = x_fe2o3_dec * c_fe2o3 + x_h2o_dec * c_h2o + x_sio2_dec * c_sio2

print("\n4. Внески компонентів у теплоємність суміші, кДж/(кг·°С):")
print(f"   Fe2O3: {x_fe2o3_dec * c_fe2o3:.5f}")
print(f"   H2O: {x_h2o_dec * c_h2o:.5f}")
print(f"   SiO2: {x_sio2_dec * c_sio2:.5f}")

print(f"\n5. Питома теплоємність суміші:")
print(f"   c = {c_mix:.5f} кДж/(кг·°С)")

# Переведення в одиниці SI
c_mix_SI = c_mix * 1000  # кДж -> Дж

print("\n6. Питома теплоємність в одиницях SI:")
print(f"   c = {c_mix_SI:.2f} Дж/(кг·К)")

print("\n7. Перевірка достовірності:")
print(f"   - Результат в межах теплоємностей компонентів ({min(c_fe2o3, c_h2o, c_sio2)} - {max(c_fe2o3, c_h2o, c_sio2)} кДж/(кг·°С)): {'✓' if min(c_fe2o3, c_h2o, c_sio2) <= c_mix <= max(c_fe2o3, c_h2o, c_sio2) else '✗'}")
print(f"   - Сума масових часток = 1: {'✓' if abs(x_sum - 1) < 1e-10 else '✗'}")
print(f"   - Розмірності збережені та переведені в SI: ✓")
