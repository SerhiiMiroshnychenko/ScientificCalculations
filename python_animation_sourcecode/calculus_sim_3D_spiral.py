'''
LICENSE AGREEMENT

In relation to this Python file:

1. Copyright of this Python file is owned by the author: Mark Misin

2. This Python code can be freely used and distributed

3. The copyright label in this Python file such as

copyright=ax_main.text(x,y,'© Mark Misin Engineering',size=z)

that indicate that the Copyright is owned by Mark Misin MUST NOT be removed.

WARRANTY DISCLAIMER!

This Python file comes with absolutely NO WARRANTY! In no event can the author of this Python file be held responsible for whatever happens in relation to this Python file. For example, if there is a bug in the code and because of that a project, invention, or whatever it is used for fails - the author is NOT RESPONSIBLE!
'''

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Set simulation duration time
t0=0.0
t_end=10.0
dt=0.02
# Create array for time
t=np.arange(t0,t_end+dt,dt)

# Create array for position_x
ampl_x=3
freq_x=1
position_x=ampl_x*t*np.cos(2*np.pi*freq_x*t)

# Create array for position_y
ampl_y=6
freq_y=1
position_y=ampl_y*t*np.sin(2*np.pi*freq_y*t)

#####################################################################

# Create array for position_z
position_z=t

####################################################################


# Set up the animation function
frame_amount=int(t_end/dt)
size_constant=1

# The update function
# The update function needs to be above the figures
def update_plot(num):

    # Trajectory
    plane_trajectory.set_xdata(position_x[0:num])
    plane_trajectory.set_ydata(position_y[0:num])
    plane_trajectory.set_3d_properties(position_z[0:num])

    pos_x.set_data(t[0:num],position_x[0:num])
    pos_y.set_data(t[0:num],position_y[0:num])
    pos_z.set_data(t[0:num],position_z[0:num])

    return plane_trajectory,pos_x,pos_y,pos_z


# Set up your figure properties
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(3,4)

# Airplane motion
ax0=fig.add_subplot(gs[:,0:3],projection='3d',facecolor=(0.9,0.9,0.9))
plane_trajectory,=ax0.plot([],[],[],'r',linewidth=4,label='Flight trajectory')
ax0.set_xlim(min(position_y),max(position_y))
ax0.set_ylim(min(position_y),max(position_y))
ax0.set_zlim(0,position_z[-1])
plt.xlabel('position_x [m]',fontsize=12)
plt.ylabel('position_y [m]',fontsize=12)
ax0.set_zlabel('position_z [m]',fontsize=12)
plt.grid(True)

# Copyright
copyright=ax0.text(-t_end,max(position_x),position_z[-1]*1.1,'© Mark Misin Engineering',size=15)

ax1=fig.add_subplot(gs[0,3],facecolor=(0.9,0.9,0.9))
pos_x,=ax1.plot([],[],'b',linewidth=2,label='Pos_x = '+str(ampl_x)+'cos(2π'+str(freq_x)+'t)')
plt.xlim(t0,t_end)
plt.ylim(min(position_x),max(position_x))
plt.ylabel('position_x [m]',fontsize=12)
plt.grid(True)
plt.legend(loc='upper left',fontsize='small')

ax2=fig.add_subplot(gs[1,3],facecolor=(0.9,0.9,0.9))
pos_y,=ax2.plot([],[],'b',linewidth=2,label='Pos_y = '+str(ampl_y)+'sin(2π'+str(freq_y)+'t)')
plt.xlim(t0,t_end)
plt.ylim(min(position_y),max(position_y))
plt.ylabel('position_y [m]',fontsize=12)
plt.grid(True)
plt.legend(loc='upper left',fontsize='small')

ax3=fig.add_subplot(gs[2,3],facecolor=(0.9,0.9,0.9))
pos_z,=ax3.plot([],[],'b',linewidth=2,label='Pos_z = t')
plt.xlim(t0,t_end)
plt.ylim(0,position_z[-1])
plt.xlabel('time [s]',fontsize=12)
plt.ylabel('position_z [m]',fontsize=12)
plt.grid(True)
plt.legend(loc='lower right',fontsize='small')


plane_ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=True,blit=True)

plt.show()
