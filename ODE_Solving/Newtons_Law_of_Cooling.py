import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Define the ODE function
def pie_ode_fun(T, t, c=-0.028):
    return c * (T - 23)


# Set up time range
t_range = np.linspace(0, 200, 1000)  # 1000 points from 0 to 200 minutes

# Initial conditions
T0 = 175  # Initial temperature for cooling
T0_ = 5  # Initial temperature for heating

# Solve the ODE
T_sol = odeint(pie_ode_fun, T0, t_range)
T_sol_ = odeint(pie_ode_fun, T0_, t_range)

# Create the plot
plt.figure()
plt.plot(t_range, T_sol, 'r', label='Охолодження від 175°C')
plt.plot(t_range, T_sol_, 'b', label='Нагрів від 5°C')

# Label axes
plt.xlabel('time (minutes)')
plt.ylabel('temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
