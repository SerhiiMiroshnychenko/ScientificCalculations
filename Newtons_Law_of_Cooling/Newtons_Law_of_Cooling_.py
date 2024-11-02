# Визначення температури об'єкта з часом при охолодженні від 400 °C до нормальних умов (20 °C)
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Вхідні дані:
t_start, t_stop, t10, t30 = 0, 100, 10, 30  # Час початку, закінчення та для визначення температури, хв.
T0, Tn = 400, 20  # Початкова температура та температура при нормальних умовах °C.
a, step_number = 0.058, 1000  # Коефіцієнт тепловіддачі, Вт/м²/К та кількість кроків.

def sinter_ode_fun(T, t):  # Локальна функція що імплементує Зако́н Нью́тона — Рі́хмана
    return - a * (T - Tn)

t_range = np.linspace(t_start, t_stop, step_number)  # Змінна для інтервалу часу t=0 до 100 хвилин
T_sol = odeint(sinter_ode_fun, T0, t_range)  # Виклик odeint, щоб розв’язати диференціальне рівняння
T10 = T_sol[np.abs(t_range - t10).argmin()][0]  # Пошук температури через 10 хвилин
T30 = T_sol[np.abs(t_range - t30).argmin()][0]  # Пошук температури через 30 хвилин
plt.plot(t_range, T_sol, 'r', label='зміна температури від часу')  # Графік охолодження
plt.scatter(t10, T10, color='red'); plt.scatter(t30, T30, color='red')
plt.text(t10 + 2, T10, f'{T10:.1f} °C - температура через {t10} хв', color='red', ha='left')
plt.text(t30 + 2, T30, f'{T10:.1f} °C - температура через {t30} хв', color='red', ha='left')
plt.axhline(y=T10, xmin=t_start, xmax=t10 / t_stop, color='red', linestyle='dotted')
plt.axvline(x=t10, ymin=t_start, ymax=(T10 - Tn) / (T0 - Tn), color='red', linestyle='dotted')
plt.axhline(y=T30, xmin=0, xmax=t30 / t_stop, color='red', linestyle='dotted')
plt.axvline(x=t30, ymin=0, ymax=0.01 + (T30 - Tn) / (T0 - Tn), color='red', linestyle='dotted')
plt.title("Охолодження об'єкта (Python)"); plt.legend()  # Опис та налаштування графіка
plt.xlabel('час (хвилини)'); plt.ylabel('температура (°C)'); plt.xlim(left=t_start, right=t_stop); plt.show()
