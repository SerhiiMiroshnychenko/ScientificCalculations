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
import numpy as np
import time


power=-1
dt=0.05
t_end0=0
d_t_end=10.0

t01=0.05
t_end1=t_end0+d_t_end
t1=np.arange(t01,t_end1+dt,dt)

t02=t_end1
t_end2=t_end1+d_t_end
t2=np.arange(t02,t_end2+dt,dt)

t03=t_end2
t_end3=t_end2+d_t_end
t3=np.arange(t03,t_end3+dt,dt)

y_1=t1**power
y_2=t2**power
y_3=t3**power

frame_amount=len(y_1)+len(y_2)+len(y_3)


def update_plot(num):

    if num<200:
        y1.set_data(t1[0:num],y_1[0:num])
    elif num<401:
        y2.set_data(t2[0:num-200],y_2[0:num-200])
    else:
        y3.set_data(t3[0:num-400],y_3[0:num-400])

    return y1,y2,y3


# Set up your figure properties
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(3,1)
bbox_props_time=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='g',lw=1.0)


# Figure 1
ax1=fig.add_subplot(gs[0,0],facecolor=(0.9,0.9,0.9))
y1,=ax1.plot([],[],'-b',linewidth=3,label='y=t^('+str(power)+')')
plt.xlim(0,t_end1)
plt.ylim(0,y_1[0])
plt.grid(True)
plt.legend(loc='upper right',fontsize='medium')
txt1b=ax1.text(t1[-1]*0.1,y_1[0]*0.85,'y_begin = '+str(round(y_1[0],3)),size='15',color='g',bbox=bbox_props_time)
txt1e=ax1.text(t1[-1]*0.8,y_1[0]*0.2,'y_fin = '+str(round(y_1[-1],3)),size='15',color='g',bbox=bbox_props_time)
# Copyright
copyright=ax1.text(0,y_1[0]*3.2/3,'© Mark Misin Engineering',size=12)

# Figure 2
ax2=fig.add_subplot(gs[1,0],facecolor=(0.9,0.9,0.9))
y2,=ax2.plot([],[],'-b',linewidth=3,label='y=t^('+str(power)+')')
plt.xlim(t_end1+dt,t_end2)
plt.ylim(0,y_2[0])
plt.grid(True)
plt.legend(loc='upper right',fontsize='medium')
txt2b=ax2.text(t2[-1]*0.55,y_2[0]*0.6,'y_begin = '+str(round(y_2[0],3)),size='15',color='g',bbox=bbox_props_time)
txt2e=ax2.text(t2[-1]*0.9,y_2[0]*0.2,'y_fin = '+str(round(y_2[-1],3)),size='15',color='g',bbox=bbox_props_time)

# Figure 3
ax3=fig.add_subplot(gs[2,0],facecolor=(0.9,0.9,0.9))
y3,=ax3.plot([],[],'-b',linewidth=3,label='y=t^('+str(power)+')')
plt.xlim(t_end2+dt,t_end3)
plt.ylim(0,y_3[0])
plt.grid(True)
plt.legend(loc='upper right',fontsize='medium')
txt3b=ax3.text(t3[-1]*0.7,y_3[0]*0.6,'y_begin = '+str(round(y_3[0],3)),size='15',color='g',bbox=bbox_props_time)
txt3e=ax3.text(t3[-1]*0.9,y_3[0]*0.2,'y_fin = '+str(round(y_3[-1],3)),size='15',color='g',bbox=bbox_props_time)


plane_ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=True,blit=True)

plt.show()
