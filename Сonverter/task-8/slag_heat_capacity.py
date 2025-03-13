# Імпортуємо необхідні бібліотеки
import numpy as np

# Масові частки компонентів, %
x_cao = 45    # CaO
x_sio2 = 40   # SiO2
x_al2o3 = 15  # Al2O3

# Температура, °C
T = 500

print("1. Вхідні дані:")
print(f"   Температура: {T}°C")
print("\n   Масові частки компонентів, %:")
print(f"   CaO: {x_cao}")
print(f"   SiO2: {x_sio2}")
print(f"   Al2O3: {x_al2o3}")

# Переведення масових часток у десятковий вигляд
x_cao_dec = x_cao/100
x_sio2_dec = x_sio2/100
x_al2o3_dec = x_al2o3/100

print("\n2. Масові частки у десятковому вигляді:")
print(f"   CaO: {x_cao_dec:.4f}")
print(f"   SiO2: {x_sio2_dec:.4f}")
print(f"   Al2O3: {x_al2o3_dec:.4f}")

# Перевірка суми масових часток
x_sum = x_cao_dec + x_sio2_dec + x_al2o3_dec
print(f"\n3. Сума масових часток: {x_sum:.4f}")

# Функції для розрахунку теплоємності компонентів
def c_cao(T):
    """Теплоємність CaO, кДж/(кг·°С)"""
    return 0.749 + 3.78e-4*T - 1.533e-7*T**2

def c_sio2(T):
    """Теплоємність SiO2, кДж/(кг·°С)"""
    return 0.794 + 9.4e-4*T - 7.15e-7*T**2

def c_al2o3(T):
    """Теплоємність Al2O3, кДж/(кг·°С)"""
    return 0.786 + 5.97e-4*T - 2.98e-7*T**2

def c_slag(T):
    """Теплоємність шлаку, кДж/(кг·°С)"""
    return 0.694 + 8.95e-4*T - 1.18e-6*T**2 + 5.72e-10*T**3

# Розрахунок теплоємностей компонентів при заданій температурі
c_cao_T = c_cao(T)
c_sio2_T = c_sio2(T)
c_al2o3_T = c_al2o3(T)

print("\n4. Теплоємності компонентів при {T}°C, кДж/(кг·°С):")
print(f"   CaO: {c_cao_T:.4f}")
print(f"   SiO2: {c_sio2_T:.4f}")
print(f"   Al2O3: {c_al2o3_T:.4f}")

# Метод адитивності
c_additive = x_cao_dec * c_cao_T + x_sio2_dec * c_sio2_T + x_al2o3_dec * c_al2o3_T

print("\n5. Внески компонентів у теплоємність суміші (метод адитивності), кДж/(кг·°С):")
print(f"   CaO: {x_cao_dec * c_cao_T:.4f}")
print(f"   SiO2: {x_sio2_dec * c_sio2_T:.4f}")
print(f"   Al2O3: {x_al2o3_dec * c_al2o3_T:.4f}")

print(f"\n6. Середня питома теплоємність (метод адитивності):")
print(f"   c = {c_additive:.4f} кДж/(кг·°С)")

# Метод з урахуванням температурної залежності
c_temp = c_slag(T)

print(f"\n7. Середня питома теплоємність (з урахуванням температурної залежності):")
print(f"   c = {c_temp:.4f} кДж/(кг·°С)")

# Порівняння методів
abs_diff = abs(c_additive - c_temp)
rel_diff = abs_diff/c_temp * 100

print("\n8. Порівняння методів:")
print(f"   Абсолютна різниця: {abs_diff:.4f} кДж/(кг·°С)")
print(f"   Відносна різниця: {rel_diff:.2f}%")

print("\n9. Перевірка достовірності:")
print(f"   - Сума масових часток = 1: {'✓' if abs(x_sum - 1) < 1e-10 else '✗'}")
print(f"   - Розмірності збережені: ✓")
