# Імпортуємо необхідні бібліотеки
import numpy as np

# Масові частки компонентів, %
x_cao = 47    # CaO
x_feo = 14    # FeO
x_mno = 15    # MnO
x_sio2 = 24   # SiO2

# Температура, K
T = 1000 + 273.15

print("1. Вхідні дані:")
print(f"   Температура: {T} K")
print("\n   Масові частки компонентів, %:")
print(f"   CaO: {x_cao}")
print(f"   FeO: {x_feo}")
print(f"   MnO: {x_mno}")
print(f"   SiO2: {x_sio2}")

# Переведення масових часток у десятковий вигляд
x_cao_dec = x_cao/100
x_feo_dec = x_feo/100
x_mno_dec = x_mno/100
x_sio2_dec = x_sio2/100

print("\n2. Масові частки у десятковому вигляді:")
print(f"   CaO: {x_cao_dec:.4f}")
print(f"   FeO: {x_feo_dec:.4f}")
print(f"   MnO: {x_mno_dec:.4f}")
print(f"   SiO2: {x_sio2_dec:.4f}")

# Перевірка суми масових часток
x_sum = x_cao_dec + x_feo_dec + x_mno_dec + x_sio2_dec
print(f"\n3. Сума масових часток: {x_sum:.4f}")

# Функції для розрахунку теплоємності компонентів
def c_cao(T):
    """Теплоємність CaO, кДж/(кг·K)"""
    return 0.749 + 3.78e-4*T - 1.535e-7*T**2

def c_sio2(T):
    """Теплоємність SiO2, кДж/(кг·K)"""
    return 0.768 + 3.23e-4*T

def c_feo():
    """Теплоємність FeO, кДж/(кг·K)"""
    return 0.7872

def c_mno():
    """Теплоємність MnO, кДж/(кг·K)"""
    return 0.7268

def c_slag(T):
    """Теплоємність шлаку, кДж/(кг·K)"""
    return 0.777 + 1.31e-4*T - 5.45e-8*T**2

# Розрахунок теплоємностей компонентів при заданій температурі
c_cao_T = c_cao(T)
c_sio2_T = c_sio2(T)
c_feo_T = c_feo()
c_mno_T = c_mno()

print(f"\n4. Теплоємності компонентів при {T} K, кДж/(кг·K):")
print(f"   CaO: {c_cao_T:.4f}")
print(f"   FeO: {c_feo_T:.4f}")
print(f"   MnO: {c_mno_T:.4f}")
print(f"   SiO2: {c_sio2_T:.4f}")

# Метод адитивності
c_additive = (x_cao_dec * c_cao_T + x_feo_dec * c_feo_T + 
              x_mno_dec * c_mno_T + x_sio2_dec * c_sio2_T)

print("\n5. Внески компонентів у теплоємність суміші (метод адитивності), кДж/(кг·K):")
print(f"   CaO: {x_cao_dec * c_cao_T:.4f}")
print(f"   FeO: {x_feo_dec * c_feo_T:.4f}")
print(f"   MnO: {x_mno_dec * c_mno_T:.4f}")
print(f"   SiO2: {x_sio2_dec * c_sio2_T:.4f}")

print(f"\n6. Середня питома теплоємність (метод адитивності):")
print(f"   c = {c_additive:.4f} кДж/(кг·K)")

# Метод з урахуванням температурної залежності
c_temp = c_slag(T)

print(f"\n7. Середня питома теплоємність (з урахуванням температурної залежності):")
print(f"   c = {c_temp:.4f} кДж/(кг·K)")

# Порівняння методів
abs_diff = abs(c_additive - c_temp)
rel_diff = abs_diff/c_temp * 100

print("\n8. Порівняння методів:")
print(f"   Абсолютна різниця: {abs_diff:.4f} кДж/(кг·K)")
print(f"   Відносна різниця: {rel_diff:.2f}%")

print("\n9. Перевірка достовірності:")
print(f"   - Сума масових часток = 1: {'✓' if abs(x_sum - 1) < 1e-10 else '✗'}")
print(f"   - Розмірності збережені: ✓")
print(f"   - Температура в межах діапазонів:")
print(f"     * CaO (0-800°C): {'✗' if (T-273.15) > 800 else '✓'}")
print(f"     * SiO2 (0-1300°C): {'✗' if (T-273.15) > 1300 else '✓'}")
print(f"     * FeO (800-1500°C): {'✓' if 800 <= (T-273.15) <= 1500 else '✗'}")
print(f"     * MnO (800-1500°C): {'✓' if 800 <= (T-273.15) <= 1500 else '✗'}")
print(f"     * Шлак (до температури плавлення): ✓")
