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
import add_plane_cubic_competition as ap


# Determine power of the FUNCTION
power=3 # Use power of a max altitude
power_2=3
power_3=3
power_4=0.5
power_5=3


# Set simulation duration time
t0=0.0
t_end=2.0
dt=0.005

# Create array for time
t=np.arange(t0,t_end+dt,dt)#*0 # For stopping the cars

# Function coefficients for max altitude
a=1.5
b=0

# Create funtion for the airplane
start_dist=0
finish_dist=int(a*t_end**power) # Always use the power of max altitude as reference
step_dist=finish_dist/(len(t)-1)
ground=0
altitude=5 # Use max altitude, other planes must be below it.
altitude_2=4
altitude_3=3
altitude_4=2
altitude_5=1



# Initial speed
init_speed=0


# Set up the animation function
frame_amount=int(t_end/dt)
bbox_props_start=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='g',lw=1.0)
bbox_props_finish=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='r',lw=1.0)

# Set up distance, height, and speed arrays
a2=a-1
a3=a-1
a4=a*4
a5=a-3
b2=b
b3=b+3
b4=b
b5=b

experiment=ap.AddPlane(finish_dist,step_dist,t)
distance,height,speed=experiment.create_arrays(power,altitude,a,b) # Put here the max altitude data
distance_2,height_2,speed_2=experiment.create_arrays(power_2,altitude_2,a2,b2)
distance_3,height_3,speed_3=experiment.create_arrays(power_3,altitude_3,a3,b3)
distance_4,height_4,speed_4=experiment.create_arrays(power_4,altitude_4,a4,b4)
distance_5,height_5,speed_5=experiment.create_arrays(power_5,altitude_5,a5,b5)


# The update function
# The update function needs to be above the figures
def update_plot(num):

    # Draw planes, trajectories, functions
    experiment.create_plane(dist,plane_trajectory,plane_1,plane_2,
        plane_3,plane_4,plane_5,num,distance,height,speed)
    experiment.create_plane(dist_2,plane_trajectory_2,plane_1_2,
        plane_2_2,plane_3_2,plane_4_2,plane_5_2,num,distance_2,height_2,speed_2)
    experiment.create_plane(dist_3,plane_trajectory_3,plane_1_3,
        plane_2_3,plane_3_3,plane_4_3,plane_5_3,num,distance_3,height_3,speed_3)
    experiment.create_plane(dist_4,plane_trajectory_4,plane_1_4,
        plane_2_4,plane_3_4,plane_4_4,plane_5_4,num,distance_4,height_4,speed_4)
    experiment.create_plane(dist_5,plane_trajectory_5,plane_1_5,
        plane_2_5,plane_3_5,plane_4_5,plane_5_5,num,distance_5,height_5,speed_5)


    return plane_trajectory, plane_trajectory_2,plane_trajectory_3,plane_trajectory_4,plane_trajectory_5,\
    dist, dist_2, dist_3,dist_4,dist_5,\
    plane_1, plane_2, plane_3, plane_4, plane_5,\
    plane_1_2, plane_2_2, plane_3_2, plane_4_2,plane_5_2,\
    plane_1_3, plane_2_3, plane_3_3, plane_4_3,plane_5_3,\
    plane_1_4, plane_2_4, plane_3_4, plane_4_4,plane_5_4,\
    plane_1_5, plane_2_5, plane_3_5, plane_4_5,plane_5_5


# Set up your figure properties
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(5,4)

# Airplane motion
ax0=fig.add_subplot(gs[:,0:3],facecolor=(0.9,0.9,0.9))

extraction=experiment.initialize_stuff(ax0,'-b')
extraction_2=experiment.initialize_stuff(ax0,'-r')
extraction_3=experiment.initialize_stuff(ax0,'-k')
extraction_4=experiment.initialize_stuff(ax0,'-g')
extraction_5=experiment.initialize_stuff(ax0,'-m')


plane_trajectory=extraction[0]
plane_trajectory_2=extraction_2[0]
plane_trajectory_3=extraction_3[0]
plane_trajectory_4=extraction_4[0]
plane_trajectory_5=extraction_5[0]


# Draw an airplane
plane_1,plane_2,plane_3,plane_4,plane_5=extraction[1:6]
plane_1_2,plane_2_2,plane_3_2,plane_4_2,plane_5_2=extraction_2[1:6]
plane_1_3,plane_2_3,plane_3_3,plane_4_3,plane_5_3=extraction_3[1:6]
plane_1_4, plane_2_4, plane_3_4, plane_4_4,plane_5_4=extraction_4[1:6]
plane_1_5, plane_2_5, plane_3_5, plane_4_5,plane_5_5=extraction_5[1:6]


plt.xlim(start_dist-3,finish_dist)
plt.ylim(ground,altitude+1)
plt.xticks(np.arange(start_dist-3,finish_dist+1,finish_dist/4),size=15)
plt.yticks(np.arange(ground,altitude+2),size=15)
# plt.xlabel('distance [km]',fontsize=15)
plt.ylabel('plane number',fontsize=15)
# plt.title('Airplane',fontsize=20)
plt.grid(True)

# Start and finish sign
start=ax0.text(-1.5,-0.6,'START @ 0 km',size='20',color='g',bbox=bbox_props_start)
finish=ax0.text(4.5,-0.6,'FINISH @ 6 km',size='20',color='r',bbox=bbox_props_finish)
finish=ax0.text(10.5,-0.6,'FINISH @ 12 km',size='20',color='r',bbox=bbox_props_finish)

# Copyright
copyright=ax0.text(-3,(altitude+1)*3.1/3,'© Mark Misin Engineering',size=15)



# Distance functions
extraction_distance=experiment.initialize_distance(fig,gs,0,3,'-b',t0,t_end,start_dist,finish_dist,power,a,b)
ax1=extraction_distance[0]
dist=extraction_distance[1]

extraction_distance=experiment.initialize_distance(fig,gs,1,3,'-r',t0,t_end,start_dist,finish_dist,power_2,a2,b2)
ax2=extraction_distance[0]
dist_2=extraction_distance[1]

extraction_distance=experiment.initialize_distance(fig,gs,2,3,'-k',t0,t_end,start_dist,finish_dist,power_3,a3,b3)
ax3=extraction_distance[0]
dist_3=extraction_distance[1]

extraction_distance=experiment.initialize_distance(fig,gs,3,3,'-g',t0,t_end,start_dist,finish_dist,power_4,a4,b4)
ax4=extraction_distance[0]
dist_4=extraction_distance[1]

extraction_distance=experiment.initialize_distance(fig,gs,4,3,'-m',t0,t_end,start_dist,finish_dist,power_5,a5,b5)
ax5=extraction_distance[0]
dist_5=extraction_distance[1]

plane_ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=True,blit=True)

plt.show()
