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


# Determine power of the FUNCTION
power=1
power_2=0.5
power_3=1

# Set simulation duration time
t0=0.0
t_end=2.0
dt=0.005
# Create array for time
t=np.arange(t0,t_end+dt,dt)

# Create funtion for the airplane
start_dist=0
finish_dist=int(800*t_end**power)
step_dist=int(finish_dist/(len(t)-1))
ground=0
altitude=3
altitude_2=2
altitude_3=1

init_speed=0

# Create distance array using function
a=800
b=0
distance=a*t**power+b
distance_2=a*t**power_2+b
distance_3=a*t**power_3+b
# Create array for distance
height=np.ones(int(finish_dist/step_dist))*altitude
height_2=np.ones(int(finish_dist/step_dist))*altitude_2
height_3=np.ones(int(finish_dist/step_dist))*altitude_3
# Create function for speed
if power<1:
    t[0]=0.0025
else:
    t[0]=0



# Set up the animation function
frame_amount=int(t_end/dt)
time_during_all_frames=2 # X hours during all frames
frames_per_1unit_time=frame_amount/time_during_all_frames
biasX=220 # To shift texts horizontally on the distance plot
biasY=0.2 # To shift texts vertically on the distance plot
biasY_extra=0.125 #To adjust biasY


# The update function
# The update function needs to be above the figures
def update_plot(num):

    plane_trajectory.set_data(distance[0:num],height[0:num])
    plane_trajectory_2.set_data(distance_2[0:num],height_2[0:num])
    plane_trajectory_3.set_data(distance_3[0:num],height_3[0:num])

    # Draw a plane
    plane_1.set_data([distance[num]-40,distance[num]+20],
        [height[num],height[num]])
    plane_2.set_data([distance[num]-20,distance[num]],
        [height[num]+0.3,height[num]])
    plane_3.set_data([distance[num]-20,distance[num]],
        [height[num]-0.3,height[num]])
    plane_4.set_data([distance[num]-40,distance[num]-30],
        [height[num]+0.15,height[num]])
    plane_5.set_data([distance[num]-40,distance[num]-30],
        [height[num]-0.15,height[num]])

    # Draw a plane 2
    plane_1_2.set_data([distance_2[num]-40,distance_2[num]+20],
        [height_2[num],height_2[num]])
    plane_2_2.set_data([distance_2[num]-20,distance_2[num]],
        [height_2[num]+0.3,height_2[num]])
    plane_3_2.set_data([distance_2[num]-20,distance_2[num]],
        [height_2[num]-0.3,height_2[num]])
    plane_4_2.set_data([distance_2[num]-40,distance_2[num]-30],
        [height_2[num]+0.15,height_2[num]])
    plane_5_2.set_data([distance_2[num]-40,distance_2[num]-30],
        [height_2[num]-0.15,height_2[num]])

    # Draw a plane 3
    plane_1_3.set_data([distance_3[num]-40,distance_3[num]+20],
        [height_3[num],height_3[num]])
    plane_2_3.set_data([distance_3[num]-20,distance_3[num]],
        [height_3[num]+0.3,height_3[num]])
    plane_3_3.set_data([distance_3[num]-20,distance_3[num]],
        [height_3[num]-0.3,height_3[num]])
    plane_4_3.set_data([distance_3[num]-40,distance_3[num]-30],
        [height_3[num]+0.15,height_3[num]])
    plane_5_3.set_data([distance_3[num]-40,distance_3[num]-30],
        [height_3[num]-0.15,height_3[num]])


    dist.set_data(t[0:num],distance[0:num])
    dist_2.set_data(t[0:num],distance_2[0:num])
    dist_3.set_data(t[0:num],distance_3[0:num])




    return plane_trajectory, plane_trajectory_2,plane_trajectory_3,\
    dist, dist_2, dist_3, \
    plane_1, plane_2, plane_3, plane_4,plane_5,\
    plane_1_2, plane_2_2, plane_3_2, plane_4_2,plane_5_2,\
    plane_1_3, plane_2_3, plane_3_3, plane_4_3,plane_5_3,\



# Set up your figure properties
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(2,1)

# Airplane motion
ax0=fig.add_subplot(gs[0,:],facecolor=(0.9,0.9,0.9))
plane_trajectory,=ax0.plot([],[],'-r',linewidth=4)
plane_trajectory_2,=ax0.plot([],[],'-m',linewidth=4)
plane_trajectory_3,=ax0.plot([],[],'-b',linewidth=4)

# Draw an airplane
plane_1,=ax0.plot([],[],'k',linewidth=10)
plane_2,=ax0.plot([],[],'k',linewidth=5)
plane_3,=ax0.plot([],[],'k',linewidth=5)
plane_4,=ax0.plot([],[],'k',linewidth=3)
plane_5,=ax0.plot([],[],'k',linewidth=3)

# Draw an airplane 2
plane_1_2,=ax0.plot([],[],'k',linewidth=10)
plane_2_2,=ax0.plot([],[],'k',linewidth=5)
plane_3_2,=ax0.plot([],[],'k',linewidth=5)
plane_4_2,=ax0.plot([],[],'k',linewidth=3)
plane_5_2,=ax0.plot([],[],'k',linewidth=3)

# Draw an airplane 3
plane_1_3,=ax0.plot([],[],'k',linewidth=10)
plane_2_3,=ax0.plot([],[],'k',linewidth=5)
plane_3_3,=ax0.plot([],[],'k',linewidth=5)
plane_4_3,=ax0.plot([],[],'k',linewidth=3)
plane_5_3,=ax0.plot([],[],'k',linewidth=3)


plt.xlim(start_dist,finish_dist+10)
plt.ylim(ground,altitude+1)
plt.xticks(np.arange(start_dist,finish_dist+1,finish_dist/4),size=15)
plt.yticks(np.arange(ground,altitude+2),size=15)
plt.xlabel('distance [km]',fontsize=15)
plt.ylabel('height [km]',fontsize=15)
# plt.title('Airplane',fontsize=20)
plt.grid(True)

# Copyright
copyright=ax0.text(0,(altitude+1)*3.2/3,'© Mark Misin Engineering',size=15)



# Distance function
ax2=fig.add_subplot(gs[1,:], facecolor=(0.9,0.9,0.9))
dist,=ax2.plot([],[],'-r',linewidth=3,label='D = '+str(a)+'*t^'+str(power))
dist_2,=ax2.plot([],[],'-m',linewidth=3,label='D = '+str(a)+'*t^'+str(power_2))
dist_3,=ax2.plot([],[],'--b',linewidth=3,label='D = '+str(a)+'*t^'+str(power_3))
plt.xlim(t0,t_end+0.05)
plt.ylim(start_dist,finish_dist+50)
plt.xticks(np.arange(t0,t_end+0.5,0.5),size=15)
plt.yticks(np.arange(start_dist,finish_dist+1,finish_dist/4),size=15)
plt.xlabel('time [hrs]',fontsize=15)
plt.ylabel('distance [km]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper left',fontsize='x-large')


plane_ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=True,blit=True)

plt.show()
