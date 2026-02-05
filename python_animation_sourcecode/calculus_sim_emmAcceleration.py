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

# Same sphere on all planets
radius=5 # meters
bottom=-10
final_altitude=100
dAlt=10
width_ratio=1.2

t0=0
t_end=12
dt=0.02
frame_amount=int(t_end/dt)
t=np.arange(t0,t_end+dt,dt)


altitude_Earth=final_altitude-(9.8/2)*t**2
altitude_Mars=final_altitude-(3.7/2)*t**2
altitude_Moon=final_altitude-(1.6/2)*t**2

velocity_Earth=-9.8*t
velocity_Mars=-3.7*t
velocity_Moon=-1.6*t

acceleration_Earth=-9.8
acceleration_Mars=-3.7
acceleration_Moon=-1.6

def create_circle(r):
    degrees=np.arange(0,361)
    radians=degrees*np.pi/180
    sphere_x=r*np.cos(radians)
    sphere_y=r*np.sin(radians)
    return sphere_x,sphere_y

def update_plot(num):
    if altitude_Earth[num]-radius>=0:
        spehere_Earth.set_data([sphere_xE],[sphere_yE+altitude_Earth[num]])
        alt_E.set_data(t[0:num],altitude_Earth[0:num])
        vel_E.set_data(t[0:num],velocity_Earth[0:num])
        acc_E.set_data(t[0:num],[acceleration_Earth])

    if altitude_Mars[num]-radius>=0:
        spehere_Mars.set_data([sphere_xMa],[sphere_yMa+altitude_Mars[num]])
        alt_Ma.set_data(t[0:num],altitude_Mars[0:num])
        vel_Ma.set_data(t[0:num],velocity_Mars[0:num])
        acc_Ma.set_data(t[0:num],[acceleration_Mars])

    if altitude_Moon[num]-radius>=0:
        spehere_Moon.set_data([sphere_xMo],[sphere_yMo+altitude_Moon[num]])
        alt_Mo.set_data(t[0:num],altitude_Moon[0:num])
        vel_Mo.set_data(t[0:num],velocity_Moon[0:num])
        acc_Mo.set_data(t[0:num],[acceleration_Moon])


    return spehere_Earth,spehere_Mars,spehere_Moon,alt_E,alt_Ma,alt_Mo,\
    vel_E,vel_Ma,vel_Mo,acc_E,acc_Ma,acc_Mo


# Set up your figure properties20
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(3,4)

# Create object for Earth
ax0=fig.add_subplot(gs[:,0],facecolor=(0.9,0.9,0.9))
sphere_xE,sphere_yE=create_circle(radius)
spehere_Earth,=ax0.plot([],[],'k',linewidth=3)
land_Earth=ax0.plot([-radius*width_ratio,radius*width_ratio],[-5,-5],linewidth=38)
plt.xlim(-radius*width_ratio,radius*width_ratio)
plt.ylim(bottom,final_altitude+dAlt)
plt.xticks(np.arange(-radius,radius+1,radius))
plt.yticks(np.arange(bottom,final_altitude+2*dAlt,dAlt))
plt.ylabel('altitude [m]')
plt.title('Earth')
copyright=ax0.text(-radius*width_ratio,(final_altitude+10)*3.2/3,'© Mark Misin Engineering',size=12)

# Create object for Mars
ax1=fig.add_subplot(gs[:,1],facecolor=(0.9,0.9,0.9))
sphere_xMa,sphere_yMa=create_circle(radius)
spehere_Mars,=ax1.plot([],[],'k',linewidth=3)
land_Mars=ax1.plot([-radius*width_ratio,radius*width_ratio],[-5,-5],'orangered',linewidth=38)
plt.xlim(-radius*width_ratio,radius*width_ratio)
plt.ylim(bottom,final_altitude+dAlt)
plt.xticks(np.arange(-radius,radius+1,radius))
plt.yticks(np.arange(bottom,final_altitude+2*dAlt,dAlt))
plt.title('Mars')


# Create object for Moon
ax2=fig.add_subplot(gs[:,2],facecolor=(0.9,0.9,0.9))
sphere_xMo,sphere_yMo=create_circle(radius)
spehere_Moon,=ax2.plot([],[],'k',linewidth=3)
land_Moon=ax2.plot([-radius*width_ratio,radius*width_ratio],[-5,-5],'gray',linewidth=38)
plt.xlim(-radius*width_ratio,radius*width_ratio)
plt.ylim(bottom,final_altitude+dAlt)
plt.xticks(np.arange(-radius,radius+1,radius))
plt.yticks(np.arange(bottom,final_altitude+2*dAlt,dAlt))
plt.title('Moon')

# Create position function
ax3=fig.add_subplot(gs[0,3], facecolor=(0.9,0.9,0.9))
alt_E,=ax3.plot([],[],'',linewidth=3,label='Alt_Earth = '+str(final_altitude)+' - '+str(round(9.8/2,1))+'t^'+str(2)+' [m]')
alt_Ma,=ax3.plot([],[],'orangered',linewidth=3,label='Alt_Mars = '+str(final_altitude)+' - '+str(round(3.7/2,1))+'t^'+str(2)+' [m]')
alt_Mo,=ax3.plot([],[],'gray',linewidth=3,label='Alt_Moon = '+str(final_altitude)+' - '+str(round(1.6/2,1))+'t^'+str(2)+' [m]')
plt.xlim(0,t_end)
plt.ylim(0,final_altitude)
plt.legend(loc=(0.6,0.7),fontsize='x-small')

# Create velocity function
ax4=fig.add_subplot(gs[1,3], facecolor=(0.9,0.9,0.9))
vel_E,=ax4.plot([],[],'',linewidth=3,label='Vel_Earth = '+str(-9.8)+'t [m/s]')
vel_Ma,=ax4.plot([],[],'orangered',linewidth=3,label='Vel_Mars = '+str(-3.7)+'t [m/s]')
vel_Mo,=ax4.plot([],[],'gray',linewidth=3,label='Vel_Moon = '+str(-1.6)+'t [m/s]')
plt.xlim(0,t_end)
plt.ylim(velocity_Earth[-1],0)
plt.legend(loc='lower left',fontsize='x-small')

# Create acceleration function
ax5=fig.add_subplot(gs[2,3], facecolor=(0.9,0.9,0.9))
acc_E,=ax5.plot([],[],'',linewidth=3,label='Acc_Earth = '+str(-9.8)+' [(m/s)/s ≡ m/s^2]')
acc_Ma,=ax5.plot([],[],'orangered',linewidth=3,label='Acc_Mars = '+str(-3.7)+' [(m/s)/s ≡ m/s^2]')
acc_Mo,=ax5.plot([],[],'gray',linewidth=3,label='Acc_Moon = '+str(-1.6)+' [(m/s)/s ≡ m/s^2]')
plt.xlim(0,t_end)
plt.ylim(acceleration_Earth-1.2,0)
plt.legend(loc=(0.02,0.25),fontsize='x-small')



plane_ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=True,blit=True)

plt.show()
