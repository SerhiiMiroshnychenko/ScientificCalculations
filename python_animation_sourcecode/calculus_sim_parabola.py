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
power=3

# Set simulation duration time
t0=0.0
t_end=2.0
dt=0.005
# Create array for time
t=np.arange(t0,t_end+dt,dt)

# Create funtion for the airplane
start_dist=0
finish_dist=int(800*t_end**power)/8
step_dist=int(finish_dist/(len(t)-1))
ground=0
altitude=3
init_speed=0

# Create distance array using function
a=800
b=0
distance=a*t**power+b
# Create array for distance
height=np.ones(int(finish_dist/step_dist))*altitude
# Create function for speed
if power<1:
    t[0]=0.0025
else:
    t[0]=0
speed=power*a*t**(power-1)


# Set up the animation function
frame_amount=int(t_end/dt)
time_during_all_frames=2 # X hours during all frames
frames_per_1unit_time=frame_amount/time_during_all_frames
biasX=220 # To shift texts horizontally on the distance plot
biasY=0.2 # To shift texts vertically on the distance plot
biasY_extra=0.125 #To adjust biasY
bbox_props_time=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='g',lw=1.0)
bbox_props_dist=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='r',lw=1.0)
bbox_props_title=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='b',lw=1.0)

n=20
dotted_index=np.zeros(frame_amount)
dot=np.zeros(frame_amount)
for i in range(0,frame_amount):
    if i/n==n/n:
        dotted_index[i]=n
        dot[i]=distance[n]
        n=n+20
    else:
        dotted_index[i]=n-20
        dot[i]=distance[n-20]

# The update function
# The update function needs to be above the figures
def update_plot(num):

    if dotted_index[num-1]!=dotted_index[num] and num!=0:
        plane_trajectory.set_data(dot[0:num],height[0:num])
    if num==frame_amount-1:
        plane_trajectory.set_data(dot[0:num]+20,height[0:num])
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


    # Connect plane with axes
    dist_indic_vert.set_data([distance[num],distance[num]],
        [ground,altitude])
    dist_indic_horiz.set_data([start_dist,t[num]],
        [distance[num],distance[num]])
    time_indic_vert.set_data([t[num],t[num]],
        [ground,distance[num]])
    time_indic_vert_2.set_data([t[num],t[num]],
        [init_speed,speed[num]])
    dist.set_data(t[0:num],distance[0:num])
    spd.set_data(t[0:num],speed[0:num])

    stopwatch0.set_text(str(round(t[num+1],1))+' hrs')
    dist_counter0.set_text(str(int(round(distance[num+1])))+' km')
    if distance[num]>800:
        dist_title.set_text('Applied Calculus for Engineers')

    # Speed calculation display
    division_dist.set_text('d(D)')
    division_time.set_text('d(t)')
    division_speed.set_text('= '+str(int(round(speed[num+1])))+' km/hr')


    return plane_trajectory, dist, spd, dist_indic_vert, dist_indic_horiz, \
    time_indic_vert, time_indic_vert_2,division_dist,division_time, \
    division_speed , plane_1, plane_2, plane_3, plane_4,plane_5,\
    stopwatch0, dist_counter0,dist_title


# Set up your figure properties
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(2,2)

# Airplane motion
ax0=fig.add_subplot(gs[0,:],facecolor=(0.9,0.9,0.9))
plane_trajectory,=ax0.plot([],[],'r:o',linewidth=2)

# Draw an airplane
plane_1,=ax0.plot([],[],'k',linewidth=10)
plane_2,=ax0.plot([],[],'k',linewidth=5)
plane_3,=ax0.plot([],[],'k',linewidth=5)
plane_4,=ax0.plot([],[],'k',linewidth=3)
plane_5,=ax0.plot([],[],'k',linewidth=3)

dist_indic_vert,=ax0.plot([],[],'k:o',linewidth=2)
plt.xlim(start_dist,finish_dist+10)
plt.ylim(ground,altitude+1)
plt.xticks(np.arange(start_dist,finish_dist+1,finish_dist/4),size=15)
plt.yticks(np.arange(ground,altitude+2),size=15)
plt.xlabel('distance [km]',fontsize=15)
plt.ylabel('height [km]',fontsize=15)
# plt.title('Airplane',fontsize=20)
plt.grid(True)

# Time display on distance graph
stopwatch0=ax0.text(finish_dist*0.875,(altitude+1)*0.65/3,'',size='20',color='g',bbox=bbox_props_time)
# Distance display on distance graph
dist_counter0=ax0.text(finish_dist*0.875,(altitude+1)*0.2/3,'',size='20',color='r',bbox=bbox_props_dist)
dist_title=ax0.text(185,1.33,'',size='30',color='b',bbox=bbox_props_title)

# Copyright
copyright=ax0.text(0,(altitude+1)*3.2/3,'© Mark Misin Engineering',size=15)



# Distance function
ax2=fig.add_subplot(gs[1,0], facecolor=(0.9,0.9,0.9))
dist,=ax2.plot([],[],'-b',linewidth=3,label='D = '+str(a)+'*t^'+str(power))
dist_indic_horiz,=ax2.plot([],[],'g:o',linewidth=2, label='time spent')
time_indic_vert,=ax2.plot([],[],'r:o',linewidth=2, label='distance covered')
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
time_indic_vert_2,=ax4.plot([],[],'b:o',linewidth=2,label='speed')
plt.xlim(t0,t_end+0.05)
plt.xticks(np.arange(t0,t_end+0.5,0.5),size=15)
if power <1:
    plt.ylim(init_speed,speed[0]+50)
    plt.yticks(np.arange(init_speed,speed[0]+1,speed[0]/4),size=15)
    division_dist=ax4.text(0.1,1015/1600*(speed[0]),'',fontsize=20,color='r')
    division_line,=ax4.plot([0.1,0.25],
        [995/1600*(speed[0]),995/1600*(speed[0])],'k',linewidth=1)
    division_time=ax4.text(0.1,865/1600*(speed[0]),'',fontsize=20,color='g')
    division_speed=ax4.text(0.3,950/1600*(speed[0]),'',fontsize=20,color='b')
else:
    plt.ylim(init_speed,speed[-1]+50)
    plt.yticks(np.arange(speed[0],speed[-1]+1,speed[-1]/4),size=15)
    division_dist=ax4.text(0.1,1015/1600*(speed[-1]),'',fontsize=20,color='r')
    division_line,=ax4.plot([0.1,0.25],
        [995/1600*(speed[-1]),995/1600*(speed[-1])],'k',linewidth=1)
    division_time=ax4.text(0.1,865/1600*(speed[-1]),'',fontsize=20,color='g')
    division_speed=ax4.text(0.35,950/1600*(speed[-1]),'',fontsize=20,color='b')
plt.xlabel('time [hrs]',fontsize=15)
plt.ylabel('speed [km/hr]',fontsize=15)
plt.title('Speed as a function of time',fontsize=15)
plt.grid(True)
if power==1:
    plt.legend(loc='lower left',fontsize='x-large')
else:
    plt.legend(loc='upper left',fontsize='x-large')


plane_ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=False,blit=True)

plt.show()
