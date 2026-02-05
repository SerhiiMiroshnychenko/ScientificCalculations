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


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np


#########################################################################
# Set simulation inputs
t0=0.0
t_end=1
dt=0.001
# Create array for time
t=np.arange(t0,t_end+dt,dt)
########################################################################
frame_amount=int(t_end/dt+1) # Don't forget to chang frame rate
f1=1
f2=3
theta1=2*np.pi*f1*t
theta2=2*np.pi*f2*t
theta_tot=theta1+theta2
length_j1=3+t
x=np.cos(theta1)*(length_j1+2*np.cos(theta2))-np.sin(theta1)*2*np.sin(theta2)
y=np.sin(theta1)*(length_j1+2*np.cos(theta2))+np.cos(theta1)*2*np.sin(theta2)
# plt.plot(x,y)
x_1=(3+t)*np.cos(theta1)
y_1=(3+t)*np.sin(theta1)
x_2=2*np.cos(theta_tot)
y_2=2*np.sin(theta_tot)
def update_plot(num):
    # len(num)=122 -> num=0-121 -> t[0:121]=t[0]->t[120]
    joint_1.set_data([0,x_1[num]],[0,y_1[num]])
    joint_2.set_data([x_1[num],x_1[num]+x_2[num]],[y_1[num],y_1[num]+y_2[num]])
    length_j1_funct.set_data(t[0:num],length_j1[0:num])
    theta1_funct.set_data(t[0:num],theta1[0:num])
    theta2_funct.set_data(t[0:num],theta2[0:num])
    trajectory.set_data(x[0:num],y[0:num])

    return joint_1,joint_2,trajectory,length_j1_funct,theta1_funct,theta2_funct


# Set up your figure properties
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(3,3)
plt.subplots_adjust(left=0.03,bottom=0.035,right=0.99,top=0.97,wspace=0.15,hspace=0.2)

# Action
ax1=fig.add_subplot(gs[:,0:2],facecolor=(0.9,0.9,0.9))
base_line,=ax1.plot([0,0],[0,0.4],'k',linewidth=20)
joint_1,=ax1.plot([],[],'k',linewidth=4)
joint_2,=ax1.plot([],[],'b',linewidth=4)
trajectory,=ax1.plot([],[],'r',linewidth=2)
ax1.spines['left'].set_position('center')
ax1.spines['bottom'].set_position(('center'))
ax1.xaxis.set_label_coords(0.5, -0.01)
ax1.yaxis.set_label_coords(-0.002, 0.5)
size_x=10
size_y=10
plt.xlim(-size_x,size_x)
plt.ylim(-size_y,size_y)
plt.xticks(np.arange(-size_x,size_x+1,1))
plt.yticks(np.arange(-size_y,size_y+1,1))
plt.xlabel('meters [m]',fontsize=12)
plt.ylabel('meters [m]',fontsize=12)
plt.grid(True)
copyright=ax1.text(-size_x,size_y*1.01,'© Mark Misin Engineering',size=10)

ax2=fig.add_subplot(gs[0,2],facecolor=(0.9,0.9,0.9))
length_j1_funct,=ax2.plot([],[],'b',linewidth=2)
plt.xlim(t0,t_end)
plt.ylim(length_j1[0]-1,length_j1[-1]+1)
plt.xlabel('time [s]',fontsize=12)
plt.ylabel('meters [m]',fontsize=12)
plt.grid(True)

ax3=fig.add_subplot(gs[1,2],facecolor=(0.9,0.9,0.9))
theta1_funct,=ax3.plot([],[],'b',linewidth=2)
plt.xlim(t0,t_end)
plt.ylim(0,4*np.pi)
plt.yticks(np.arange(0,6*np.pi+0.1,np.pi),['0','π','2π','3π','4π','5π','6π'])
plt.xlabel('time [s]',fontsize=12)
plt.ylabel('angle [rad]',fontsize=12)
plt.grid(True)

ax4=fig.add_subplot(gs[2,2],facecolor=(0.9,0.9,0.9))
theta2_funct,=ax4.plot([],[],'b',linewidth=2)
plt.xlim(t0,t_end)
plt.ylim(0,4*np.pi)
plt.yticks(np.arange(0,6*np.pi+0.1,np.pi),['0','π','2π','3π','4π','5π','6π'])
plt.xlabel('time [s]',fontsize=12)
plt.ylabel('angle [rad]',fontsize=12)
plt.grid(True)

ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=10,repeat=True,blit=True)

plt.show()
