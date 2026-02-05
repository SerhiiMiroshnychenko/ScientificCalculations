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



radius=5
bottom=0
final_volume=100
dVol=10
width_ratio=1
dt=0.04
t0=0
t_end=5
frame_amount=int(t_end/dt)
t=np.arange(t0,t_end+dt,dt)

volume_Tank1=np.zeros(len(t))
volume_Tank2=np.zeros(len(t))
volume_Tank3=np.zeros(len(t))

for i in range(0,len(t)):
    if t[i]<1+11**0.5:
        volume_Tank1[i]=50+2*t[i]
        volume_Tank2[i]=40+t[i]**2
        x=(22+2*11**0.5)/(1+11**0.5)**3
        volume_Tank3[i]=30+x*t[i]**3
        temp=i
    else:
        volume_Tank1[i]=50+2*t[temp]
        volume_Tank2[i]=40+t[temp]**2
        x=(22+2*11**0.5)/(1+11**0.5)**3
        volume_Tank3[i]=30+x*t[temp]**3


def update_plot(num):
    # if num>=len(volume_Tank1):
    #     num=len(volume_Tank1)-1

    tank_1.set_data([-radius*width_ratio,radius*width_ratio],
        [volume_Tank1[num],volume_Tank1[num]])
    tank_12.set_data([0,0],[-62,volume_Tank1[num]-62])
    tnk_1.set_data(t[0:num],volume_Tank1[0:num])

    tank_2.set_data([-radius*width_ratio,radius*width_ratio],
        [volume_Tank2[num],volume_Tank2[num]])
    tank_22.set_data([0,0],[-62,volume_Tank2[num]-62])
    tnk_2.set_data(t[0:num],volume_Tank2[0:num])

    tank_3.set_data([-radius*width_ratio,radius*width_ratio],
        [volume_Tank3[num],volume_Tank3[num]])
    tank_32.set_data([0,0],[-62,volume_Tank3[num]-62])
    tnk_3.set_data(t[0:num],volume_Tank3[0:num])


    return tank_1,tank_12,tank_2,tank_22,tank_3,tank_32,\
    tnk_1,tnk_2,tnk_3,\


# Set up your figure properties20
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(2,3)

# Create object for Tank1
ax0=fig.add_subplot(gs[0,0],facecolor=(0.9,0.9,0.9))
tank_1,=ax0.plot([],[],linewidth=2)
tank_12,=ax0.plot([],[],'royalblue',linewidth=260)
plt.xlim(-radius*width_ratio,radius*width_ratio)
plt.ylim(bottom,final_volume)
plt.xticks(np.arange(-radius,radius+1,radius))
plt.yticks(np.arange(bottom,final_volume+dVol,dVol))
plt.ylabel('tank volume [m^3]')
plt.title('Tank 1')
copyright=ax0.text(-radius*width_ratio,(final_volume+10)*3.2/3,'© Mark Misin Engineering',size=12)

# Create object for Tank2
ax1=fig.add_subplot(gs[0,1],facecolor=(0.9,0.9,0.9))
tank_2,=ax1.plot([],[],linewidth=2)
tank_22,=ax1.plot([],[],'royalblue',linewidth=260)
plt.xlim(-radius*width_ratio,radius*width_ratio)
plt.ylim(bottom,final_volume)
plt.xticks(np.arange(-radius,radius+1,radius))
plt.yticks(np.arange(bottom,final_volume+dVol,dVol))
plt.title('Tank 2')


# Create object for Tank3
ax2=fig.add_subplot(gs[0,2],facecolor=(0.9,0.9,0.9))
tank_3,=ax2.plot([],[],linewidth=2)
tank_32,=ax2.plot([],[],'royalblue',linewidth=260)
plt.xlim(-radius*width_ratio,radius*width_ratio)
plt.ylim(bottom,final_volume)
plt.xticks(np.arange(-radius,radius+1,radius))
plt.yticks(np.arange(bottom,final_volume+dVol,dVol))
plt.title('Tank 3')

# Create volume function
ax3=fig.add_subplot(gs[1,:], facecolor=(0.9,0.9,0.9))
tnk_1,=ax3.plot([],[],'blue',linewidth=3,label='Tank 1')
tnk_2,=ax3.plot([],[],'green',linewidth=3,label='Tank 2')
tnk_3,=ax3.plot([],[],'red',linewidth=3,label='Tank 3')
plt.xlim(0,t_end)
plt.ylim(0,final_volume)
plt.xticks([0, 1, 2, 3, 4, 1+11**0.5, 5])
plt.ylabel('tank volume [m^3]')
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

plane_ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=30,repeat=True,blit=True)
plt.show()
