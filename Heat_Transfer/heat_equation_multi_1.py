# %% [markdown]
# # Multidimensional differential equations
#
# - Börge Göbel

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# %% [markdown]
#

# %% [markdown]
# We solve the differential equations:
#
# \\(
# \frac{\partial}{\partial t} u(\vec{r},t) = a \Delta u(\vec{r},t)
# \\)

# %% [markdown]
# ## In one dimension:
#
# \\(
# \frac{\partial}{\partial t} u(x,t) = a \frac{\partial^2}{\partial x^2} u(x,t)
# \\)
#
# Here, \\( u(x,t) \\) is an array \\(\{ u_1, u_2, \dots, u_n \} \\) that has different values for different times. It describes the temperature. We can discretize the spatial derivative according to:
#
# \\(
# \frac{\partial^2}{\partial x^2} u_j = \frac{u_{j+1}-2u_{j}+u_{j-1}}{(\Delta x)^2}
# \\)
#
# For the edges we use double-forward or double-backward methods:
#
# \\(
# \frac{\partial^2}{\partial x^2} u_1 = \frac{u_{1}-2u_{2}+u_{3}}{(\Delta x)^2}\\
# \frac{\partial^2}{\partial x^2} u_n = \frac{u_{n}-2u_{n-1}+u_{n-2}}{(\Delta x)^2}
# \\)
#
# We can rewrite the heat equation as a set of coupled equation:
#
# \begin{align}
# \frac{\partial}{\partial t}u_1&=\frac{a}{(\Delta x)^2}\left(u_1-2u_2+u_3\right)\\
# \frac{\partial}{\partial t}u_2&=\frac{a}{(\Delta x)^2}\left(u_1-2u_2+u_3\right)\\
# \frac{\partial}{\partial t}u_3&=\frac{a}{(\Delta x)^2}\left(u_2-2u_3+u_4\right)\\
# \vdots\\
# \frac{\partial}{\partial t}u_j&=\frac{a}{(\Delta x)^2}\left(u_{j-1}-2u_j+u_{j+1}\right)\\
# \vdots\\
# \frac{\partial}{\partial t}u_{n-2}&=\frac{a}{(\Delta x)^2}\left(u_{n-3}-2u_{n-2}+u_{n-1}\right)\\
# \frac{\partial}{\partial t}u_{n-1}&=\frac{a}{(\Delta x)^2}\left(u_{n-2}-2u_{n-1}+u_{n}\right)\\
# \frac{\partial}{\partial t}u_n&=\frac{a}{(\Delta x)^2}\left(u_{n-2}-2u_{n-1}+u_{n}\right)
# \end{align}
#
# Alternatively, we can also keep the temperature at the edges constant and consider these to be (part of) the constant heat bath:
#
# \\( u_1 = \mathrm{const.}\\ u_n = \mathrm{const.} \\)
#

# %%
# Testing
u = np.array([1,4,9,16,25])
unew = np.zeros(5)
unew[1:-1] = u[2:] -2*u[1:-1] + u[:-2]

# %%
a = 1.0
dx = 1.0

def f_1D(t,u):
    unew = np.zeros(len(u))
    unew[1:-1] = u[2:] -2*u[1:-1] + u[:-2]
    return unew * a/dx**2

# %%
tStart = 0
tEnd = 5000

size = 100
u0 = np.zeros([size])
u0[0] = 1

solution = integrate.solve_ivp(f_1D, [tStart, tEnd], u0, method='RK45', t_eval=np.linspace(tStart,tEnd,10001))

# %%
index = size//2

plt.xlabel('Time t')
plt.ylabel('Temperature for cell #'+str(index))

plt.plot(solution.t, solution.y[index])
plt.show()

# %%
t_list, x_list = np.meshgrid(solution.t, np.arange(size))

plt.xlabel('Time t')
plt.ylabel('Coordinate')

plt.contourf(t_list, x_list, solution.y)
plt.colorbar()
plt.show()

# %% [markdown]
# ### Other starting parameters

# %%
tStart = 0
tEnd = 2000

size = 100
u0 = np.zeros([size])
u0[0] = 1
u0[-1] = 1

solution = integrate.solve_ivp(f_1D, [tStart, tEnd], u0, method='RK45', t_eval=np.linspace(tStart,tEnd,10001))

# %%
index = size//2

plt.xlabel('Time t')
plt.ylabel('Temperature for cell #'+str(index))

plt.plot(solution.t, solution.y[index])
plt.show()

# %%
t_list, x_list = np.meshgrid(solution.t, np.arange(size))

plt.xlabel('Time t')
plt.ylabel('Coordinate')

plt.contourf(t_list, x_list, solution.y)
plt.colorbar()
plt.show()
