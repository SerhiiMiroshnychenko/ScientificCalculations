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
import add_plane as ap


# Determine power of the FUNCTION
power=0.5 # Use power of a max altitude
power_2=2
power_3=1


# Set simulation duration time
t0=0.0
t_end=2.0
dt=0.005

# Create array for time
t=np.arange(t0,t_end+dt,dt)

# Create funtion for the airplane
start_dist=0
finish_dist=int(800*t_end**power) # Always use the power of max altitude as reference
step_dist=int(finish_dist/(len(t)-1))
ground=0
altitude=3 # Use max altitude, other planes must be below it.
altitude_2=2
altitude_3=1


# Initial speed
init_speed=0

# Function coefficients for max altitude
a=800
b=0

# Set up the animation function
frame_amount=int(t_end/dt)
time_during_all_frames=2 # X hours during all frames
frames_per_1unit_time=frame_amount/time_during_all_frames
biasX=220 # To shift texts horizontally on the distance plot
biasY=0.2 # To shift texts vertically on the distance plot
biasY_extra=0.125 #To adjust biasY
bbox_props_time=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='g',lw=1.0)
bbox_props_dist=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='r',lw=1.0)

# Set up distance, height, and speed arrays
experiment=ap.AddPlane(finish_dist,step_dist,t)
distance,height,speed=experiment.create_arrays(power,altitude,a,b) # Put here the max altitude data
distance_2,height_2,speed_2=experiment.create_arrays(power_2,altitude_2,a,b)
distance_3,height_3,speed_3=experiment.create_arrays(power_3,altitude_3,a,b)

n=20
dotted_index=np.zeros(frame_amount)
dot=np.zeros(frame_amount)
dotted_index2=np.zeros(frame_amount)
dot2=np.zeros(frame_amount)
dotted_index3=np.zeros(frame_amount)
dot3=np.zeros(frame_amount)
for i in range(0,frame_amount):
    if i/n==n/n:
        dotted_index[i]=n
        dot[i]=distance[n]
        dotted_index2[i]=n
        dot2[i]=distance_2[n]
        dotted_index3[i]=n
        dot3[i]=distance_3[n]
        n=n+20
    else:
        dotted_index[i]=n-20
        dot[i]=distance[n-20]
        dotted_index2[i]=n-20
        dot2[i]=distance_2[n-20]
        dotted_index3[i]=n-20
        dot3[i]=distance_3[n-20]
# The update function
# The update function needs to be above the figures
def update_plot(num):

    # Draw planes, trajectories, functions
    experiment.create_plane(spd,dist,plane_trajectory,plane_1,plane_2,
        plane_3,plane_4,plane_5,num,distance,height,speed,dotted_index,dot,frame_amount)
    experiment.create_plane(spd_2,dist_2,plane_trajectory_2,plane_1_2,
        plane_2_2,plane_3_2,plane_4_2,plane_5_2,num,distance_2,height_2,speed_2,dotted_index2,dot2,frame_amount)
    experiment.create_plane(spd_3,dist_3,plane_trajectory_3,plane_1_3,
        plane_2_3,plane_3_3,plane_4_3,plane_5_3,num,distance_3,height_3,speed_3,dotted_index3,dot3,frame_amount)


    return plane_trajectory, plane_trajectory_2,plane_trajectory_3,\
    dist, dist_2, dist_3, spd, spd_2, spd_3,\
    plane_1, plane_2, plane_3, plane_4, plane_5,\
    plane_1_2, plane_2_2, plane_3_2, plane_4_2,plane_5_2,\
    plane_1_3, plane_2_3, plane_3_3, plane_4_3,plane_5_3,


# Set up your figure properties
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(2,2)

# Airplane motion
ax0=fig.add_subplot(gs[0,:],facecolor=(0.9,0.9,0.9))

extraction=experiment.initialize_stuff(ax0,'b:o')
extraction_2=experiment.initialize_stuff(ax0,'r:o')
extraction_3=experiment.initialize_stuff(ax0,'k:o')


plane_trajectory=extraction[0]
plane_trajectory_2=extraction_2[0]
plane_trajectory_3=extraction_3[0]


# Draw an airplane
plane_1,plane_2,plane_3,plane_4,plane_5=extraction[1:6]
plane_1_2,plane_2_2,plane_3_2,plane_4_2,plane_5_2=extraction_2[1:6]
plane_1_3,plane_2_3,plane_3_3,plane_4_3,plane_5_3=extraction_3[1:6]


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
ax2=fig.add_subplot(gs[1,0], facecolor=(0.9,0.9,0.9))

dist,=ax2.plot([],[],'-b',linewidth=3,label='D = '+str(a)+'*t^'+str(power))
dist_2,=ax2.plot([],[],'-r',linewidth=3,label='D = '+str(a)+'*t^'+str(power_2))
dist_3,=ax2.plot([],[],'-k',linewidth=3,label='D = '+str(a)+'*t^'+str(power_3))


plt.xlim(t0,t_end+0.05)
plt.ylim(start_dist,finish_dist+50)
plt.xticks(np.arange(t0,t_end+0.5,0.5),size=15)
plt.yticks(np.arange(start_dist,finish_dist+1,finish_dist/4),size=15)
plt.xlabel('time [hrs]',fontsize=15)
plt.ylabel('distance [km]',fontsize=15)
plt.title('Distance as a function of time',fontsize=15)
plt.grid(True)
plt.legend(loc='upper left',fontsize='x-large')

# Speed function
ax4=fig.add_subplot(gs[1,1], facecolor=(0.9,0.9,0.9))
spd,=ax4.plot([],[],'-b',linewidth=3,label='d(D)/d(t) = '+str(int(power*a))+'*t^('+str(round(power-1,2))+')')
spd_2,=ax4.plot([],[],'-r',linewidth=3,label='d(D)/d(t) = '+str(int(power_2*a))+'*t^('+str(round(power_2-1,2))+')')
spd_3,=ax4.plot([],[],'-k',linewidth=3,label='d(D)/d(t) = '+str(int(power_3*a))+'*t^('+str(round(power_3-1,2))+')')


plt.xlim(t0,t_end+0.05)
plt.xticks(np.arange(t0,t_end+0.5,0.5),size=15)
if power <1:
    plt.ylim(init_speed,speed[0]+50)
    plt.yticks(np.arange(init_speed,speed[0]+1,speed[0]/4),size=15)
else:
    plt.ylim(init_speed,speed[-1]+50)
    plt.yticks(np.arange(speed[0],speed[-1]+1,speed[-1]/4),size=15)
plt.xlabel('time [hrs]',fontsize=15)
plt.ylabel('speed [km/hrs]',fontsize=15)
plt.title('Speed as a function of time',fontsize=15)
plt.grid(True)
if power==1:
    plt.legend(loc='lower left',fontsize='x-large')
else:
    plt.legend(loc='upper left',fontsize='x-large')


plane_ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=True,blit=True)

plt.show()
