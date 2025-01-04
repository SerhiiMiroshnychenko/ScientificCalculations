import numpy as np
import matplotlib.pyplot as plt

xdim = 10
ydim = 10

T1 = 80
T2 = 60
T3 = 45
T4 = 120

T_guess = 35
T = np.zeros((xdim,ydim))
T.fill(T_guess)
T[0:xdim,0] = T1
T[0,1:ydim] = T2
T[0:xdim,ydim-1] = T3
T[xdim-1,1:ydim] = T4
niter = 80
for n in range(0,niter):
    for i in range(1,xdim-1,1):
        for j in range(1,ydim-1,1):
            T[i][j] = (T[i-1][j] + T[i+1][j] + T[i][j-1] + T[i][j+1])/4
#print(T)

plt.figure(figsize=(7.5,7.5))
plt.contourf(T,80,cmap='jet')
plt.grid(color='black',linestyle='--')
plt.colorbar()
plt.show()