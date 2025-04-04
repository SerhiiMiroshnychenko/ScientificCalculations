"""
Завдання: Моделювання двовимірного розподілу температури в квадратній пластині
методом скінченних різниць.

Граничні умови:
- Ліва сторона (x=0): T1 = 80°C
- Нижня сторона (y=max): T2 = 60°C
- Права сторона (x=max): T3 = 45°C
- Верхня сторона (y=0): T4 = 50°C

Параметри моделювання:
- Розмір сітки: 10x10 точок
- Кількість ітерацій: 25
- Початкова температура всередині пластини: 25°C
"""

import numpy as np
import matplotlib.pyplot as plt

# Розміри сітки
xdim = 10
ydim = 10

# Кількість ітерацій та крок
niter = 25
delta = 1

# Граничні умови (температура на границях пластини)
t1 = 80  # ліва сторона
t2 = 60  # нижня сторона
t3 = 45  # права сторона
t4 = 50  # верхня сторона

# Початкова температура всередині пластини
t_guess = 25

# Створення масиву температур та встановлення початкових умов
T = np.zeros((xdim,ydim))
T.fill(t_guess)
# Матриця температури до встановлення граничних умов

# Встановлення граничних умов
T[0:xdim,0] = t1    # ліва сторона
T[0,1:ydim] = t2    # нижня сторона
T[0:xdim,ydim-1] = t3  # права сторона
T[xdim-1,1:ydim] = t4  # верхня сторона

# Ітераційний процес розрахунку методом скінченних різниць
for it in range(0,niter):
    for i in range(1,xdim-1,delta):
        for j in range(1,ydim-1,delta):
            # Розрахунок нової температури в точці як середнє арифметичне
            # температур у сусідніх точках
            T[i,j] = 0.25*(T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])

# Візуалізація результатів
plt.figure(figsize=(10,10))
plt.contourf(T,80,cmap='jet')
plt.grid(color='black',linestyle='--')
plt.colorbar(label='Температура, °C')
plt.title('Розподіл температури в пластині')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.show()