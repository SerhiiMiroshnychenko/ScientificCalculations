# Імпортуємо необхідні бібліотеки
import numpy as np

# Константи та вхідні дані
T_STANDARD = 273.15  # К (0°C)

# Параметри задачі
mass = 0.5  # кг
t2 = 1280  # °C
# Нижня межа інтегрування - 0 К (для металургійних розрахунків)
T1 = 0  # К
# Верхня межа - кінцева температура в Кельвінах
T2 = t2 + T_STANDARD  # К

# Коефіцієнти для теплоємності (кДж/(кг·К))
a = 0.175      # константа
b = 6e-5       # коефіцієнт при T

print(f"1. Вхідні дані:")
print(f"   Маса шлаку: {mass} кг")
print(f"   Початкова температура: {T1} К")
print(f"   Кінцева температура: {t2}°C = {T2:.2f} К")
print(f"   Теплоємність: c = {a} + {b}·T кДж/(кг·К)")

# Аналітичний метод
# Q = m·∫(c(T)·dT) = m·∫(a + b·T)·dT = m·(a·T + b·T²/2)
Q_analytical = mass * (
        a * (T2 - T1) +          # Інтеграл від константи a
        b * (T2**2 - T1**2) / 2  # Інтеграл від лінійного члена b·T
)

# Числовий метод (метод трапецій)
steps = 1000  # кількість кроків інтегрування
T = np.linspace(T1, T2, steps)  # рівномірне розбиття інтервалу [T1, T2]
c = lambda T: a + b*T  # теплоємність як функція від T
Q_numerical = mass * np.trapezoid([c(t) for t in T], T)

print(f"\n2. Результати розрахунку теплоти:")
print(f"\tАналітичний метод: {Q_analytical:.3f} кДж")
print(f"\tЧисловий метод: {Q_numerical:.3f} кДж")
print(f"\tРізниця між методами: {abs(Q_analytical - Q_numerical):.6f} кДж")
relative_error = abs(Q_analytical - Q_numerical)/abs(Q_analytical)*100
print(f"\tВідносна похибка: {relative_error:.6f}%")

# Додатковий аналіз
print(f"\n3. Аналіз складових теплоти (аналітичний метод):")
Q1 = mass * a * (T2 - T1)
Q2 = mass * b * (T2**2 - T1**2) / 2
print(f"\tВід константної складової (a·T): {Q1:.3f} кДж ({Q1/Q_analytical*100:.1f}%)")
print(f"\tВід лінійної складової (b·T²/2): {Q2:.3f} кДж ({Q2/Q_analytical*100:.1f}%)")

# Перевірка розмірностей
print(f"\n4. Перевірка розмірностей:")
print(f"\tТеплоємність: [кДж/(кг·К)]")
print(f"\tТемпература: [К]")
print(f"\tМаса: [кг]")
print(f"\tРезультат: [кДж/(кг·К)] · [К] · [кг] = [кДж] ✓")

# Примітка щодо меж інтегрування
print(f"\n5. Примітка:")
print(f"\tРозрахунок виконано відповідно до методики металургійних розрахунків,")
print(f"\tде теплоємності та теплові ефекти рахуються від абсолютного нуля (0 К)")
print(f"\tдо заданої температури.")
