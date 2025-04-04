#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Виявлення та ідентифікація грубих промахів у рядах спостереження
відповідно до слайдів з презентації.
"""

import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані зі слайдів (варіант №4, m=0, n=4)
print("=" * 60)
print("РОЗВ'ЯЗАННЯ")
print("=" * 60)

# Таблиця значень (дані зі слайду)
# Створимо масиви для кожної серії
x1 = np.array([
    1.239, 1.206, 1.231, 1.226, 1.211, 1.234, 1.279, 1.236, 1.251, 1.325, 1.331, 1.209, 1.213
])  # серія x1 (m=0)

x2 = np.array([
    1.244, 1.244, 1.263, 1.262, 1.221, 1.232, 1.223, 1.234, 1.245, 1.226, 1.327, 1.248, 1.209
])  # серія x2 (n=4)

x3 = np.array([
    1.223, 1.252, 1.267, 1.262, 1.237, 1.239, 1.253, 1.232, 1.217, 1.202, 1.327, 1.266, 1.214
])  # серія x3 (m=4)

print("\nПочаткові дані:")
print(f"x1 = {x1}")
print(f"x2 = {x2}")
print(f"x3 = {x3}")

print(f"\nn = {len(x1)}")  # кількість спостережень

# 1. Знаходимо статистичні оцінки параметрів нормального закону розподілу
# Математичне очікування:
x1_mean = np.mean(x1)
x2_mean = np.mean(x2)
x3_mean = np.mean(x3)

print("\n1. Статистичні оцінки параметрів нормального закону розподілу:")
print(f"x1_ср = {x1_mean:.3f}")
print(f"x2_ср = {x2_mean:.3f}")
print(f"x3_ср = {x3_mean:.3f}")

# Дисперсія та СКВ
D_x1 = np.sum((x1 - x1_mean)**2) / (len(x1) - 1)
D_x2 = np.sum((x2 - x2_mean)**2) / (len(x2) - 1)
D_x3 = np.sum((x3 - x3_mean)**2) / (len(x3) - 1)

sigma_x1 = np.sqrt(D_x1)
sigma_x2 = np.sqrt(D_x2)
sigma_x3 = np.sqrt(D_x3)

print("\nС.к.в. результатів спостережень:")
print(f"D_x1 = {D_x1:.3f}, σ_x1 = {sigma_x1:.3f}")
print(f"D_x2 = {D_x2:.3f}, σ_x2 = {sigma_x2:.3f}")
print(f"D_x3 = {D_x3:.3f}, σ_x3 = {sigma_x3:.3f}")

# 2. Знаходимо співвідношення для критерію виявлення викидів
gamma1_x1 = (np.max(x1) - x1_mean) / sigma_x1
gamma2_x1 = (x1_mean - np.min(x1)) / sigma_x1

gamma1_x2 = (np.max(x2) - x2_mean) / sigma_x2
gamma2_x2 = (x2_mean - np.min(x2)) / sigma_x2

gamma1_x3 = (np.max(x3) - x3_mean) / sigma_x3
gamma2_x3 = (x3_mean - np.min(x3)) / sigma_x3

print("\n2. Знаходимо співвідношення:")
print(f"γ1_x1 = {gamma1_x1:.3f}, γ2_x1 = {gamma2_x1:.3f}")
print(f"γ1_x2 = {gamma1_x2:.3f}, γ2_x2 = {gamma2_x2:.3f}")
print(f"γ1_x3 = {gamma1_x3:.3f}, γ2_x3 = {gamma2_x3:.3f}")

# 3. Знаходимо γр за довідковою таблицею для P=0.95, n=13
gamma_p = 2.18
print(f"\n3. За довідковою таблицею γр = {gamma_p} для P=0.95, n=13")

# Перевіряємо умови
print("\nПорівнюємо обчислені значення з табличним:")
print(f"γ1_x1 = {gamma1_x1:.3f} {'<' if gamma1_x1 < gamma_p else '>'} γр = {gamma_p}")
print(f"γ2_x1 = {gamma2_x1:.3f} {'<' if gamma2_x1 < gamma_p else '>'} γр = {gamma_p}")
print(f"γ1_x2 = {gamma1_x2:.3f} {'>' if gamma1_x2 > gamma_p else '<'} γр = {gamma_p}")
print(f"γ2_x2 = {gamma2_x2:.3f} {'<' if gamma2_x2 < gamma_p else '>'} γр = {gamma_p}")
print(f"γ1_x3 = {gamma1_x3:.3f} {'>' if gamma1_x3 > gamma_p else '<'} γр = {gamma_p}")
print(f"γ2_x3 = {gamma2_x3:.3f} {'<' if gamma2_x3 < gamma_p else '>'} γр = {gamma_p}")

print("\nx1 приймається, x2 та x3 містять викиди")

# Видаляємо викиди (максимальні значення в x2 та x3)
max_x2 = np.max(x2)
max_x3 = np.max(x3)

print(f"\nmax(x2) = {max_x2}")
print(f"max(x3) = {max_x3}")

# Створюємо "очищені" масиви без викидів
x1_clean = x1.copy()  # x1 без змін
x2_clean = np.array([x for x in x2 if x != max_x2])
x3_clean = np.array([x for x in x3 if x != max_x3])

print("\nОчищені масиви:")
print(f"x2 = {x2_clean}")
print(f"x3 = {x3_clean}")
print(f"n = {len(x2_clean)}")

# Повторюємо процедуру для оновлених значень
x2_mean_clean = np.mean(x2_clean)
x3_mean_clean = np.mean(x3_clean)

print(f"\nx2_ср = {x2_mean_clean:.3f}")
print(f"x3_ср = {x3_mean_clean:.3f}")

# Обчислюємо оновлені дисперсії та СКВ
D_x2_clean = np.sum((x2_clean - x2_mean_clean)**2) / (len(x2_clean) - 1)
D_x3_clean = np.sum((x3_clean - x3_mean_clean)**2) / (len(x3_clean) - 1)

sigma_x2_clean = np.sqrt(D_x2_clean)
sigma_x3_clean = np.sqrt(D_x3_clean)

print(f"\nD_x2 = {D_x2_clean:.6f}, σ_x2 = {sigma_x2_clean:.3f}")
print(f"D_x3 = {D_x3_clean:.6f}, σ_x3 = {sigma_x3_clean:.3f}")

# Повторно перевіряємо на викиди
gamma1_x2_clean = (np.max(x2_clean) - x2_mean_clean) / sigma_x2_clean
gamma2_x2_clean = (x2_mean_clean - np.min(x2_clean)) / sigma_x2_clean

gamma1_x3_clean = (np.max(x3_clean) - x3_mean_clean) / sigma_x3_clean
gamma2_x3_clean = (x3_mean_clean - np.min(x3_clean)) / sigma_x3_clean

print(f"\nγ1_x2 = {gamma1_x2_clean:.3f}, γ2_x2 = {gamma2_x2_clean:.3f}")
print(f"γ1_x3 = {gamma1_x3_clean:.3f}, γ2_x3 = {gamma2_x3_clean:.3f}")

# Нове табличне значення для n=12, P=0.95
gamma_p_new = 2.2
print(f"\nНове табличне значення γр = {gamma_p_new} для n=12, P=0.95")

if max(gamma1_x2_clean, gamma2_x2_clean, gamma1_x3_clean, gamma2_x3_clean) < gamma_p_new:
    print("Всі проходять! Після другої ітерації всі серії не містять промахів.")

# 4. Визначення результату нерівноточного спостереження
print("\n4. Визначення результату нерівноточного спостереження.")

# Середні значення серій
print("\nСередні значення серій:")
print(f"x1_ср = {x1_mean:.3f}")
print(f"x2_ср = {x2_mean_clean:.3f}")
print(f"x3_ср = {x3_mean_clean:.3f}")

# СКВ результатів спостережень серій
print("\nСКВ результатів спостережень серій:")
print(f"σ_x1 = {sigma_x1:.3f}")
print(f"σ_x2 = {sigma_x2_clean:.3f}")
print(f"σ_x3 = {sigma_x3_clean:.3f}")

# Число результатів спостережень серій
n1 = len(x1)
n2 = len(x2_clean)
n3 = len(x3_clean)

print("\nЧисло результатів спостережень серій:")
print(f"n1 = {n1}")
print(f"n2 = {n2}")
print(f"n3 = {n3}")

# Вага результатів серій
p1 = 1 / (sigma_x1**2)
p2 = 1 / (sigma_x2_clean**2)
p3 = 1 / (sigma_x3_clean**2)

print("\nВага результатів серій:")
print(f"p1 = {p1:.3f}")
print(f"p2 = {p2:.3f}")
print(f"p3 = {p3:.3f}")

# Середнє зважене результатів в серіях
x_weighted = (x1_mean * p1 + x2_mean_clean * p2 + x3_mean_clean * p3) / (p1 + p2 + p3)

print("\nСереднє зважене результатів в серіях:")
print(f"x = {x_weighted:.3f}")

# СКВ результатів у серіях (стандартні похибки середніх)
sigma_x1_mean = sigma_x1 / np.sqrt(n1)
sigma_x2_mean = sigma_x2_clean / np.sqrt(n2)
sigma_x3_mean = sigma_x3_clean / np.sqrt(n3)

print("\nСКВ результатів у серіях:")
print(f"σ(x1_ср) = {sigma_x1_mean:.3f}")
print(f"σ(x2_ср) = {sigma_x2_mean:.3f}")
print(f"σ(x3_ср) = {sigma_x3_mean:.3f}")

# СКВ середнього зваженого
sigma_x_weighted = 1 / np.sqrt((1 / sigma_x1_mean)**2 + (1 / sigma_x2_mean)**2 + (1 / sigma_x3_mean)**2)

print("\nСКВ середнього зваженого:")
print(f"σ_x = {sigma_x_weighted:.3f}")

# Число ступенів свободи
m = 3  # кількість серій
min_n = min(n1, n2, n3)  # мінімальна кількість елементів у серії
f = m * (min_n - 1)  # число ступенів свободи

print("\nЧисло ступенів свободи:")
print(f"m = {m} (число серій)")
print(f"n = {min_n} (мінімальна кількість елементів у серії)")
print(f"f = {f}")

# Коефіцієнт Стьюдента для P=0.95, f=33
t_p_f = 2.04

print("\nЗа довідковими даними коефіцієнт Стьюдента:")
print(f"t_p,f = {t_p_f}, P = 0.95")

# Довірча межа випадкової величини
delta = t_p_f * sigma_x_weighted

print("\nДовірча межа випадкової величини:")
print(f"Δ = {delta:.3f}")

# Довірчий інтервал
lower_bound = x_weighted - delta
upper_bound = x_weighted + delta

print("\nДовірчий інтервал:")
print(f"x - Δ = {lower_bound:.3f}")
print(f"x + Δ = {upper_bound:.3f}")

# Точкова оцінка результату
print("\nТочкова оцінка результату:")
print(f"x = {x_weighted:.3f}    σ_x = {sigma_x_weighted:.3f}")

# Інтервальна оцінка нерівноточного результату
print("\nІнтервальна оцінка нерівноточного результату:")
print(f"x ∈ [{lower_bound:.3f}, {upper_bound:.3f}]    або    P = 0.95    f = {f}")

# Візуалізація даних
plt.figure(figsize=(12, 8))

# Графік початкових даних
plt.subplot(2, 1, 1)
plt.title('Початкові дані')
plt.plot(range(len(x1)), x1, 'bo-', label='Серія x1')
plt.plot(range(len(x2)), x2, 'ro-', label='Серія x2')
plt.plot(range(len(x3)), x3, 'go-', label='Серія x3')
plt.axhline(y=x1_mean, color='b', linestyle='--', label=f'Середнє x1 = {x1_mean:.3f}')
plt.axhline(y=x2_mean, color='r', linestyle='--', label=f'Середнє x2 = {x2_mean:.3f}')
plt.axhline(y=x3_mean, color='g', linestyle='--', label=f'Середнє x3 = {x3_mean:.3f}')
plt.grid(True)
plt.legend()

# Графік після вилучення викидів
plt.subplot(2, 1, 2)
plt.title('Дані після вилучення викидів')
plt.plot(range(len(x1)), x1, 'bo-', label='Серія x1 (оригінал)')
plt.plot(range(len(x2_clean)), x2_clean, 'ro-', label='Серія x2 (очищена)')
plt.plot(range(len(x3_clean)), x3_clean, 'go-', label='Серія x3 (очищена)')
plt.axhline(y=x1_mean, color='b', linestyle='--', label=f'Середнє x1 = {x1_mean:.3f}')
plt.axhline(y=x2_mean_clean, color='r', linestyle='--', label=f'Середнє x2 = {x2_mean_clean:.3f}')
plt.axhline(y=x3_mean_clean, color='g', linestyle='--', label=f'Середнє x3 = {x3_mean_clean:.3f}')
plt.axhline(y=x_weighted, color='k', linestyle='-', label=f'Середнє зважене = {x_weighted:.3f}')
plt.axhline(y=lower_bound, color='k', linestyle=':', label=f'Інтервал: [{lower_bound:.3f}, {upper_bound:.3f}]')
plt.axhline(y=upper_bound, color='k', linestyle=':', label='')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('D:\\WINDSURF\\SCRIPTs\\Стратегичні-Напрямки\\1\\результати_аналізу_промахів.png')
plt.close()

print("\nГрафік результатів збережено у файл 'результати_аналізу_промахів.png'")
print("=" * 60)
