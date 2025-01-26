"""
# Двовимірне рівняння теплопровідності

## Постановка задачі
Даний скрипт розв'язує двовимірне рівняння теплопровідності:
∂u/∂t = a * (∂²u/∂x² + ∂²u/∂y²)

де:
- T(x,y,t) - температура в точці (x,y) в момент часу t
- a - коефіцієнт температуропровідності
- t - час
- x, y - просторові координати
"""

# Імпортуємо необхідні бібліотеки
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm.auto import tqdm

# Визначаємо основні фізичні параметри системи
a = 0.4   # Коефіцієнт теплопровідності, Вт/ (м * К)
dx = 1.0  # Крок сітки по x
dy = 1.0  # Крок сітки по y

class SolverWithProgress:
    def __init__(self, total_time):
        self.pbar = tqdm(total=100, desc="Прогрес розрахунку")
        self.total_time = total_time
        self.last_progress = 0
        
    def f_2D_flattened(self, t, u):
        """
        Допоміжна функція для перетворення двовимірної задачі в одновимірну.
        """
        # Оновлюємо прогрес-бар
        current_progress = int(t / self.total_time * 100)
        if current_progress > self.last_progress:
            self.pbar.update(current_progress - self.last_progress)
            self.last_progress = current_progress
            
        # Перетворюємо одновимірний масив назад у двовимірний
        u = u.reshape(100, 100)

        # Створюємо масив для похідних
        unew = np.zeros([100, 100])

        # Розраховуємо похідні для всіх внутрішніх точок
        unew[1:-1,1:-1] = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) * a/dx**2 + \
                          (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) * a/dy**2

        # Повертаємо розгорнутий одновимірний масив
        return unew.flatten()
    
    def close(self):
        self.pbar.close()

# Визначаємо розміри розрахункової сітки
sizex = 100  # Кількість точок по x
sizey = 100  # Кількість точок по y

# Задаємо параметри часової еволюції
tStart = 0       # Початковий час
tEnd = 10000     # Кінцевий час
n_steps = 10001  # Кількість кроків для збереження результату

# Граничні умови (температура на границях пластини)
Tn = 20 # температура при нормальних умовах T(н)=20 °C
T0 = 400 # змінна для початкової температури, T(0)=400 °C.

# Створення масиву температур та встановлення початкових умов
T = np.zeros([sizex, sizey])
T.fill(T0)

T[0,:] = Tn    # Температура = 20 на лівій границі
T[:,0] = Tn    # Температура = 20 на нижній границі
T[-1,:] = Tn   # Температура = 20 на правій границі
T[:,-1] = Tn   # Температура = 20 на верхній границі

print("\nРозрахунок:")
# Розв'язуємо систему рівнянь
solver1 = SolverWithProgress(tEnd)
solution = integrate.solve_ivp(
    solver1.f_2D_flattened,
    [tStart, tEnd],
    T.flatten(),
    method='RK45',
    t_eval=np.linspace(tStart,tEnd,n_steps),
    vectorized=True
)
solver1.close()

# Створюємо сітку для візуалізації
x_list, y_list = np.meshgrid(np.arange(sizex), np.arange(sizey))

# Візуалізуємо результати для різних моментів часу
for tIndex in (0, n_steps//200, n_steps//50, n_steps//25,  -1):
    plt.figure(figsize=(10, 8))
    plt.xlabel('Координата x')
    plt.ylabel('Координата y')
    plt.title(f'Розподіл температури в момент часу (t = {solution.t[tIndex]:.1f})')
    plt.contourf(x_list, y_list, solution.y[:, tIndex].reshape(sizex, sizey))
    plt.colorbar(label='Температура')
    plt.show()
