```python
# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, r2_score

# Налаштування графіків для відображення кирилиці
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

# Встановлення випадкового зерна для відтворюваності результатів
np.random.seed(42)
```

## 2. Опис вихідних даних для побудови лінійної парної регресійної моделі

Для побудови лінійної регресійної моделі використовуємо дані, що відповідають варіанту 4 завдання. Справжнє рівняння моделі має вигляд:

$$y(x) = a_0 + a_1 \cdot x + rnd(b)$$

де:
- $a_0 = 3.4$ - вільний член рівняння
- $a_1 = 2.7$ - коефіцієнт при незалежній змінній
- $rnd(b) = 1.10$ - максимальне значення випадкового шуму

Спочатку згенеруємо набір даних, що складається з 20 точок у діапазоні значень $x$ від 0 до 10, з додаванням випадкового шуму в діапазоні $[-1.10, 1.10]$ до теоретичних значень $y_{теор} = 3.4 + 2.7 \cdot x$.

```python
def generate_data(a0, a1, noise_range, n=20, x_min=0, x_max=10, seed=42):
    """
    Генерує дані згідно з лінійним рівнянням y = a0 + a1*x + випадковий_шум
    
    Параметри:
    a0 - вільний член
    a1 - коефіцієнт при x
    noise_range - максимальне значення випадкового шуму
    n - кількість точок
    x_min, x_max - діапазон значень x
    seed - зерно для генератора випадкових чисел
    
    Повертає:
    x - масив значень незалежної змінної
    y - масив значень залежної змінної
    y_true - масив точних значень без шуму
    """
    np.random.seed(seed)
    x = np.linspace(x_min, x_max, n)
    noise = np.random.uniform(-noise_range, noise_range, n)
    y_true = a0 + a1 * x
    y = y_true + noise
    return x, y, y_true

# Параметри моделі згідно з варіантом 4
a0 = 3.4  # Вільний член
a1 = 2.7  # Коефіцієнт при x
noise_range = 1.10  # Діапазон випадкового шуму

# Генерація даних
n = 20  # Кількість точок
x, y, y_true = generate_data(a0, a1, noise_range, n)

# Виведення фрагменту даних
results_df = pd.DataFrame({
    'x': x,
    'y фактичні': y,
    'y теоретичні': y_true,
    'Шум': y - y_true
})

print("Фрагмент згенерованих даних:")
print(tabulate(results_df.head(10), headers='keys', tablefmt='psql', floatfmt='.4f'))
```

```python
# Візуалізація згенерованих даних
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Фактичні дані')
plt.plot(x, y_true, '--', color='green', label=f'Справжня функція: y = {a0} + {a1}*x')
plt.title('Згенеровані дані для аналізу')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```
