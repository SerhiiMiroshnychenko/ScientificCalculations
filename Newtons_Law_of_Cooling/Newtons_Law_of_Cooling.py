"""
Скрипт для визначення температури агломерату з часом
при охолодженні від 400 °C до нормальних умов (20 °C)
і візуалізація результату
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

""" 1. Вхідні дані """
# Обмеження по часу
t_start = 0  # Час початку охолодження, хв.
t_stop = 100  # Час закінчення процесу, хв.
t10 = 10  # Час для визначення температури, 10 хв.
t30 = 30  # Час для визначення температури, 30 хв.

# Кількість кроків по часу
step_number = 1000

# Граничні температури
T0 = 400  # змінна для початкової температури, T(0)=400 °C.
Tn = 20  # температура при нормальних умовах T(н)=20 °C.

# Теплофізичні константи
a = 0.058  # коефіцієнт тепловіддачі, Вт/м²/К.

""" 2. Визначення функції """


def sinter_ode_fun(T, t):
    """
    Локальна функція що імплементує Зако́н Нью́тона — Рі́хмана
    :param T: температура
    :param t: час
    :return: значення теплового потоку, Bt/м.кв
    """
    return - a * (T - Tn)


""" 3. Розв'язання """
# Змінна для інтервалу часу t=0 до 100 хвилин
t_range = np.linspace(t_start, t_stop, step_number)  # 1000 кроків від 0 до 100 хв
# Виклик odeint, щоб розв’язати диференціальне рівняння
T_sol = odeint(sinter_ode_fun, T0, t_range)

# Пошук температури через 10 хвилин
t10_index = np.abs(t_range - t10).argmin()
T10 = T_sol[t10_index][0]

# Пошук температури через 30 хвилин
t30_index = np.abs(t_range - t30).argmin()
T30 = T_sol[t30_index][0]

""" 4. Візуалізація """
# Графік охолодження від 400 °C до нормальних умов
plt.plot(t_range, T_sol, 'r', label='зміна температури від часу')

# Додавання точки та підпису температури через 10 хвилин
plt.scatter(t10, T10, color='red')
plt.text(t10 + 2, T10, f'{T10:.1f} °C - температура через {t10} хв', color='red', ha='left')

# Додавання точки та підпису температури через 30 хвилин
plt.scatter(t30, T30, color='red')
plt.text(t30 + 2, T30, f'{T30:.1f} °C - температура через {t30} хв', color='red', ha='left')

# Додавання пунктирних ліній від точки на осі x та y
plt.axhline(y=T10, xmin=0, xmax=t10 / t_stop, color='red', linestyle='dotted')  # від точки до осі y
plt.axvline(x=t10, ymin=0, ymax=(T10 - Tn) / (T0 - Tn), color='red', linestyle='dotted')  # від точки до осі x
plt.axhline(y=T30, xmin=0, xmax=t30 / t_stop, color='red', linestyle='dotted')  # від точки до осі y
plt.axvline(x=t30, ymin=0, ymax=0.01 + (T30 - Tn) / (T0 - Tn), color='red', linestyle='dotted')  # від точки до осі x

# Опис графіка
plt.title('Охолодження агломерату (Python)')
plt.xlabel('час (хвилини)')
plt.ylabel('температура (°C)')
plt.legend()
plt.grid(False)
# Встановлення межі осі x для початку графіка з 0
plt.xlim(left=t_start, right=t_stop)
plt.show()
