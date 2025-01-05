# %% [markdown]
# # Multidimensional differential equations
#
# - Börge Göbel

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# %% [markdown]
# ## In 2 dimensions

# %% [markdown]
# \\(
# \frac{\partial}{\partial t} u(\vec{r},t) = a \left(\frac{\partial^2}{\partial x^2} u(\vec{r},t) + \frac{\partial^2}{\partial y^2} u(\vec{r},t)\right)
# \\)
#
# Here, \\( u(\vec{r},t) \\) is an array \\(\{ u_{1,1}, u_{1,2}, \dots, u_{n,n} \} \\) that has different values for different times. We can discretize the spatial derivative according to:
#
# \\(
# \frac{\partial^2}{\partial x^2} u_{i,j} + \frac{\partial^2}{\partial y^2} u_{i,j} = \frac{u_{i+1,j}-2u_{i,j}+u_{i-1,j}}{(\Delta x)^2}+\frac{u_{i,j+1}-2u_{i,j}+u_{i,j-1}}{(\Delta y)^2}
# \\)

# %%
a = 1.0
dx = 1.0
dy = 1.0

def f_2D(t,u):
    unew = np.zeros( [len(u),len(u)] )
    unew[1:-1,1:-1] = (u[2:,1:-1] -2*u[1:-1,1:-1] + u[:-2,1:-1]) * a/dx**2 + (u[1:-1,2:] -2*u[1:-1,1:-1] + u[1:-1,:-2]) * a/dy**2
    return unew

sizex = 100
sizey = 100

def f_2D_flattened(t,u):
    u = u.reshape(sizex, sizey)
    unew = np.zeros( [sizex,sizey] )
    unew[1:-1,1:-1] = (u[2:,1:-1] -2*u[1:-1,1:-1] + u[:-2,1:-1]) * a/dx**2 + (u[1:-1,2:] -2*u[1:-1,1:-1] + u[1:-1,:-2]) * a/dy**2
    return unew.flatten()

# %%
tStart = 0
tEnd = 10000

u0 = np.zeros([sizex,sizey])
u0[0,:] = 1
u0[:,0] = 1

solution = integrate.solve_ivp(f_2D_flattened, [tStart, tEnd], u0.flatten(), method='RK45', t_eval=np.linspace(tStart,tEnd,10001))

# %%
x_list, y_list = np.meshgrid(np.arange(sizex), np.arange(sizey))


tIndex = 0
plt.xlabel('Coordinate x')
plt.ylabel('Coordinate y')
plt.contourf(x_list, y_list, solution.y[:,tIndex].reshape(sizex, sizey))
plt.colorbar()
plt.title('t = '+str(solution.t[tIndex]))
plt.show()

tIndex = tEnd//200
plt.xlabel('Coordinate x')
plt.ylabel('Coordinate y')
plt.contourf(x_list, y_list, solution.y[:,tIndex].reshape(sizex, sizey))
plt.colorbar()
plt.title('t = '+str(solution.t[tIndex]))
plt.show()

tIndex = tEnd
plt.xlabel('Coordinate x')
plt.ylabel('Coordinate y')
plt.contourf(x_list, y_list, solution.y[:,tIndex].reshape(sizex, sizey))
plt.colorbar()
plt.title('t = '+str(solution.t[tIndex]))
plt.show()

# %% [markdown]
# ### Different starting conditions

# %%
tStart = 0
tEnd = 10000

u0 = np.zeros([sizex,sizey])
u0[0,:] = 1
u0[:,0] = 1

u0[-1,:] = 1
u0[:,-1] = 1

solution = integrate.solve_ivp(f_2D_flattened, [tStart, tEnd], u0.flatten(), method='RK45', t_eval=np.linspace(tStart,tEnd,10001))

# %%
x_list, y_list = np.meshgrid(np.arange(sizex), np.arange(sizey))


tIndex = 0
plt.xlabel('Coordinate x')
plt.ylabel('Coordinate y')
plt.contourf(x_list, y_list, solution.y[:,tIndex].reshape(sizex, sizey))
plt.colorbar()
plt.title('t = '+str(solution.t[tIndex]))
plt.show()

tIndex = tEnd//200
plt.xlabel('Coordinate x')
plt.ylabel('Coordinate y')
plt.contourf(x_list, y_list, solution.y[:,tIndex].reshape(sizex, sizey))
plt.colorbar()
plt.title('t = '+str(solution.t[tIndex]))
plt.show()

tIndex = tEnd
plt.xlabel('Coordinate x')
plt.ylabel('Coordinate y')
plt.contourf(x_list, y_list, solution.y[:,tIndex].reshape(sizex, sizey))
plt.colorbar()
plt.title('t = '+str(solution.t[tIndex]))
plt.show()


