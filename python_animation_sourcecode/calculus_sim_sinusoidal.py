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
t_end=8.0
dt=0.02
f1=0.125 # Frequency
f2=0.125
A1=7 # Amplitude
A2=-7
delay_car1=2
delay_car2=6
frame_amount=int(t_end/dt+2) # Don't forget to chang frame rate
bbox_props_time=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='k',lw=1.0)
# For the graph parameters
tick_size=10
label_size=10
########################################################################

# Create array for time
t=np.arange(t0,t_end+dt,dt)

delay_car1_el=int(delay_car1/dt)
tu1=np.concatenate([t,[t[-1]+dt]])
tu1[0:delay_car1_el]=0
tu1[delay_car1_el:-1]=t[0:len(tu1)-1-delay_car1_el]

delay_car2_el=int(delay_car2/dt)
tu2=np.concatenate([t,[t[-1]+dt]])
tu2[0:delay_car2_el]=0
tu2[delay_car2_el:-1]=t[0:len(tu2)-1-delay_car2_el]

# Deviation (D)
D_h1=A1*np.sin(2*np.pi*f1*t)#+A1/7*np.sin(2*np.pi*f1*7*t) # Chaotic
D_h2=A2*np.cos(2*np.pi*f2*t)
D_h1u=np.concatenate([D_h1,[0.0]]) # Update, one extra element for blocks
D_h2u=np.concatenate([D_h2,[0.0]])


car1=13-2*tu1**2
car2=13-2*tu2**1

for i in range(0,len(car1)-1):
    if car1[i]-1>4 and car1[i+1]-1<=4:
        temp1_bb=i
    if car1[i]+1>3 and car1[i+1]+1<=3:
        temp1_be=i
    if car1[i]-1>2 and car1[i+1]-1<=2:
        temp1_rb=i
    if car1[i]+1>1 and car1[i+1]+1<=1:
        temp1_re=i
    if car2[i]-1>4 and car2[i+1]-1<=4:
        temp2_bb=i
    if car2[i]+1>3 and car2[i+1]+1<=3:
        temp2_be=i
    if car2[i]-1>2 and car2[i+1]-1<=2:
        temp2_rb=i
    if car2[i]+1>1 and car2[i+1]+1<=1:
        temp2_re=i

def update_plot(num):
    # len(num)=122 -> num=0-121 -> t[0:121]=t[0]->t[120]

    # Update sinusoidal functions
    y_1.set_data(t[0:num],D_h1[0:num])
    y_2.set_data(t[0:num],D_h2[0:num])
    y_1f.set_data(t[0:num],D_h1[0:num]+1)
    y_2f.set_data(t[0:num],D_h2[0:num]+1)
    y_1b.set_data(t[0:num],D_h1[0:num]-1)
    y_2b.set_data(t[0:num],D_h2[0:num]-1)

    # Update the blocks
    block_1.set_data([D_h1u[num]-0.45,D_h1u[num]+0.45],[3.5,3.5])
    block_2.set_data([D_h2u[num]-0.45,D_h2u[num]+0.45],[1.5,1.5])

    # Update cars
    block_c1.set_data([3.5,3.5],[car1[num]-0.5,car1[num]+0.5])
    block_c2.set_data([-3.5,-3.5],[car2[num]-0.5,car2[num]+0.5])

    # Update the block functions
    block_1f.set_data([-0.05,-0.05],[D_h1u[num]-0.9,D_h1u[num]+0.9])
    block_2f.set_data([-0.05,-0.05],[D_h2u[num]-0.9,D_h2u[num]+0.9])

    change_color(no_no_area1,vline1b,vline1r,num,3,'blue','red',1)
    change_color(no_no_area2,vline2b,vline2r,num,6,'blue','red',1)

    change_color(no_no_area5,vline3b,vline3r,num,4,'blue','red',-1)
    change_color(no_no_area6,vline4b,vline4r,num,1,'blue','red',-1)

    change_color(no_no_area3,vline5b,vline5r,num,-3,'blue','red',1)
    change_color(no_no_area4,vline6b,vline6r,num,-6,'blue','red',1)

    change_color(no_no_area7,vline7b,vline7r,num,-4,'blue','red',-1)
    change_color(no_no_area8,vline8b,vline8r,num,-1,'blue','red',-1)

    # Car 1
    if num<delay_car1_el:
        car_time1_delay.set_data(t[0:num],[3.5])
    elif num>=delay_car1_el and num<temp1_bb:
        car_time1.set_data(t[delay_car1_el:num],[3.5])
    elif num>=temp1_bb and num<temp1_be:
        car_time1b.set_data(t[temp1_bb:num],[3.5])

    if num>=temp1_rb and num<temp1_re:
        car_time1r.set_data(t[temp1_rb:num],[3.5])


    # Car 2
    if num<delay_car2_el:
        car_time2_delay.set_data(t[0:num],[-3.5])
    elif num>=delay_car2_el and num<temp2_bb:
        car_time2.set_data(t[delay_car2_el:num],[-3.5])
    elif num>=temp2_bb and num<temp2_be:
        car_time2b.set_data(t[temp2_bb:num],[-3.5])

    if num>=temp2_rb and num<temp2_re:
        car_time2r.set_data(t[temp2_rb:num],[-3.5])


    # Return function values

    # If only one return value, have comma in the end to make it iterable
    # SOLUTION MODE: Uncomment other things
    # return y_1,y_2,y_1f,y_2f,y_1b,y_2b,\
    # block_1,block_1f,\
    # block_2,block_2f,\
    # block_c1,block_c2,\
    # no_no_area1,no_no_area2,no_no_area3,no_no_area4,\
    # no_no_area5,no_no_area6,no_no_area7,no_no_area8,\
    # vline1b,vline2b,vline3b,vline4b,vline5b,vline6b,vline7b,vline8b,\
    # vline1r,vline2r,vline3r,vline4r,vline5r,vline6r,vline7r,vline8r,\
    # car_time1_delay,car_time1,car_time1b,car_time1r,\
    # car_time2_delay,car_time2,car_time2b,car_time2r


    # EXPLANATION MODE:

    return y_1,block_1,block_1f,block_c1,block_c2,#y_1f,y_1b,\
    no_no_area1,no_no_area2,no_no_area3,no_no_area4,no_no_area5,no_no_area6,no_no_area7,no_no_area8,\
    vline1b,vline2b,vline3b,vline4b,vline5b,vline6b,vline7b,vline8b,\
    car_time1_delay,car_time1,car_time1b,\
    car_time2_delay,car_time2,car_time2b,\


    # INSTRUCTION MODE:

    # return y_1,block_1,block_1f,block_c1,block_c2,y_2,block_2,block_2f\




def change_color(no_no_area,vline1,vline2,num,a,color1,color2,b):
    if a>=0:
        if b==1:
            if D_h1u[num-1]+1<a and D_h1u[num]+1>=a and num>0:
                no_no_area.set_data([0,t[-1]],[a,a])
                no_no_area.set_color(color1)
                vline1.set_data([t[num],t[num]],[0,8])
                #input()
            elif D_h2u[num-1]+1<a and D_h2u[num]+1>=a and num>0:
                no_no_area.set_data([0,t[-1]],[a,a])
                no_no_area.set_color(color2)
                vline2.set_data([t[num],t[num]],[0,8])
            #     #input()
        else:
            if D_h1u[num-1]-1>a and D_h1u[num]-1<=a and num>0:
                no_no_area.set_data([0,t[-1]],[a,a])
                no_no_area.set_color(color1)
                vline1.set_data([t[num],t[num]],[0,8])
                #input()
            elif D_h2u[num-1]-1>a and D_h2u[num]-1<=a and num>0:
                no_no_area.set_data([0,t[-1]],[a,a])
                no_no_area.set_color(color2)
                vline2.set_data([t[num],t[num]],[0,8])
            #     #input()
    else:
        if b==1:
            if D_h1u[num-1]-1>a and D_h1u[num]-1<=a and num>0:
                no_no_area.set_data([0,t[-1]],[a,a])
                no_no_area.set_color(color1)
                vline1.set_data([t[num],t[num]],[0,-8])
                #input()
            elif D_h2u[num-1]-1>a and D_h2u[num]-1<=a and num>0:
                no_no_area.set_data([0,t[-1]],[a,a])
                no_no_area.set_color(color2)
                vline2.set_data([t[num],t[num]],[0,-8])
            #     #input()
        else:
            if D_h1u[num-1]+1<a and D_h1u[num]+1>=a and num>0:
                no_no_area.set_data([0,t[-1]],[a,a])
                no_no_area.set_color(color1)
                vline1.set_data([t[num],t[num]],[0,-8])
                #input()
            elif D_h2u[num-1]+1<a and D_h2u[num]+1>=a and num>0:
                no_no_area.set_data([0,t[-1]],[a,a])
                no_no_area.set_color(color2)
                if t[num-1]==t[-1]:
                    vline2.set_data([t[num-1],t[num-1]],[0,-8])
                else:
                    vline2.set_data([t[num],t[num]],[0,-8])
            #     #input()


# Set up your figure properties
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(1,2)
plt.subplots_adjust(left=0.03,bottom=0.035,right=0.99,top=0.97,wspace=0.035,hspace=0.2)

# Function
ax1=fig.add_subplot(gs[:,0],facecolor=(0.9,0.9,0.9))
y_1,=ax1.plot([],[],'-b',linewidth=3,label='D_h1 = '+str(A1)+'*sin(2π*'+str(f1)+'*t)')
y_2,=ax1.plot([],[],'-r',linewidth=3,label='D_h2 = '+str(A2)+'*cos(2π*'+str(f2)+'*t)')
y_2,=ax1.plot([],[],'-r',linewidth=3)
y_1f,=ax1.plot([],[],':b',linewidth=2)
y_2f,=ax1.plot([],[],':r',linewidth=2)
y_1b,=ax1.plot([],[],':b',linewidth=2)
y_2b,=ax1.plot([],[],':r',linewidth=2)
block_1f,=ax1.plot([],[],'-b',linewidth=5)
block_2f,=ax1.plot([],[],'-r',linewidth=5)
danger_zone_1,=ax1.plot([0,t[-1]],[3,3],'-k',linewidth=6)
danger_zone_2,=ax1.plot([0,t[-1]],[4,4],'-k',linewidth=6)
danger_zone_3,=ax1.plot([0,t[-1]],[-3,-3],'-k',linewidth=6)
danger_zone_4,=ax1.plot([0,t[-1]],[-4,-4],'-k',linewidth=6)
no_no_area1,=ax1.plot([],[],'--k',linewidth=2)
no_no_area2,=ax1.plot([],[],'--k',linewidth=2)
no_no_area3,=ax1.plot([],[],'--k',linewidth=2)
no_no_area4,=ax1.plot([],[],'--k',linewidth=2)
no_no_area5,=ax1.plot([],[],'-k',linewidth=2)
no_no_area6,=ax1.plot([],[],'-k',linewidth=2)
no_no_area7,=ax1.plot([],[],'-k',linewidth=2)
no_no_area8,=ax1.plot([],[],'-k',linewidth=2)
vline1b,=ax1.plot([],[],'-b')
vline2b,=ax1.plot([],[],'--b')
vline3b,=ax1.plot([],[],'--b')
vline4b,=ax1.plot([],[],'-b')
vline5b,=ax1.plot([],[],'-b')
vline6b,=ax1.plot([],[],'--b')
vline7b,=ax1.plot([],[],'--b')
vline8b,=ax1.plot([],[],'-b')
vline1r,=ax1.plot([],[],'-r')
vline2r,=ax1.plot([],[],'--r')
vline3r,=ax1.plot([],[],'--r')
vline4r,=ax1.plot([],[],'-r')
vline5r,=ax1.plot([],[],'-r')
vline6r,=ax1.plot([],[],'--r')
vline7r,=ax1.plot([],[],'--r')
vline8r,=ax1.plot([],[],'-r')

linew=4
car_time1_delay,=ax1.plot([],[],'-m',linewidth=linew)
car_time2_delay,=ax1.plot([],[],'-m',linewidth=linew)
car_time1,=ax1.plot([],[],'-k',linewidth=linew)
car_time2,=ax1.plot([],[],'-k',linewidth=linew)
car_time1b,=ax1.plot([],[],'-b',linewidth=linew)
car_time1r,=ax1.plot([],[],':r',linewidth=linew)
car_time2b,=ax1.plot([],[],'-b',linewidth=linew)
car_time2r,=ax1.plot([],[],':r',linewidth=linew)

ax1.text(1.2,2.5,'Car1_start: '+str(delay_car1)+' s',size=10,color='k',bbox=bbox_props_time)
ax1.text(1.2,2.0,'Car1_blueS: '+str(round(temp1_bb*dt,2))+' s',size=10,color='b',bbox=bbox_props_time)
ax1.text(1.2,1.5,'Car1_blueF: '+str(round(temp1_be*dt,2))+' s',size=10,color='b',bbox=bbox_props_time)
ax1.text(1.2,1.0,'Car1_redS: '+str(round(temp1_rb*dt,2))+' s',size=10,color='r',bbox=bbox_props_time)
ax1.text(1.2,0.5,'Car1_redF: '+str(round(temp1_re*dt,2))+' s',size=10,color='r',bbox=bbox_props_time)

ax1.text(5.2,2.5,'Car2_start: '+str(delay_car2)+' s',size=10,color='k',bbox=bbox_props_time)
ax1.text(5.2,2.0,'Car2_blueS: '+str(round(temp2_bb*dt,2))+' s',size=10,color='b',bbox=bbox_props_time)
ax1.text(5.2,1.5,'Car2_blueF: '+str(round(temp2_be*dt,2))+' s',size=10,color='b',bbox=bbox_props_time)
ax1.text(5.2,1.0,'Car2_redS: '+str(round(temp2_rb*dt,2))+' s',size=10,color='r',bbox=bbox_props_time)
ax1.text(5.2,0.5,'Car2_redF: '+str(round(temp2_re*dt,2))+' s',size=10,color='r',bbox=bbox_props_time)

plt.xlim(t0-0.25,t_end+5*dt)
# plt.ylim(min(A1,A2)-1,max(A1,A2)+1)
plt.ylim(-8,8)
ax1.spines['bottom'].set_position('center')
ax1.spines['left'].set_position(('data',0))
ax1.xaxis.set_label_coords(0.5, -0.01)
plt.xticks(np.arange(t0+1,t_end+dt,1),size=tick_size)
# plt.yticks(np.concatenate([np.arange(min(A1,A2)-1,0,1),np.arange(1,max(A1,A2)+2,1)]),size=tick_size)
plt.yticks(np.concatenate([np.arange(-7-1,0,1),np.arange(1,7+2,1)]),size=tick_size)
plt.xlabel('seconds [s]',fontsize=label_size)
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1.01,1.01),fontsize='medium')
# copyright=ax1.text(0,(max(A1,A2)+1)*1.01,'© Mark Misin Engineering',size=10)
copyright=ax1.text(0,(7+1)*1.01,'© Mark Misin Engineering',size=10)

# Action
ax2=fig.add_subplot(gs[:,1],facecolor=(0.9,0.9,0.9))
block_1,=ax2.plot([],[],'-b',linewidth=36)
block_2,=ax2.plot([],[],'-r',linewidth=36)
block_c1,=ax2.plot([],[],'-g',linewidth=33)
block_c2,=ax2.plot([],[],'-g',linewidth=33)


# Create danger zone 1
danger_zone1_1,=ax2.plot([3,4],[1,1],'-k',linewidth=3)
danger_zone1_2,=ax2.plot([3,4],[2,2],'-k',linewidth=3)
danger_zone1_3,=ax2.plot([3,3],[1,2],'-k',linewidth=3)
danger_zone1_4,=ax2.plot([4,4],[1,2],'-k',linewidth=3)

# Create danger zone 2
danger_zone2_1,=ax2.plot([3,4],[3,3],'-k',linewidth=3)
danger_zone2_2,=ax2.plot([3,4],[4,4],'-k',linewidth=3)
danger_zone2_3,=ax2.plot([3,3],[3,4],'-k',linewidth=3)
danger_zone2_4,=ax2.plot([4,4],[3,4],'-k',linewidth=3)

# Create danger zone 3
danger_zone3_1,=ax2.plot([-3,-4],[1,1],'-k',linewidth=3)
danger_zone3_2,=ax2.plot([-3,-4],[2,2],'-k',linewidth=3)
danger_zone3_3,=ax2.plot([-3,-3],[1,2],'-k',linewidth=3)
danger_zone3_4,=ax2.plot([-4,-4],[1,2],'-k',linewidth=3)

# Create danger zone 4
danger_zone4_1,=ax2.plot([-3,-4],[3,3],'-k',linewidth=3)
danger_zone4_2,=ax2.plot([-3,-4],[4,4],'-k',linewidth=3)
danger_zone4_3,=ax2.plot([-3,-3],[3,4],'-k',linewidth=3)
danger_zone4_4,=ax2.plot([-4,-4],[3,4],'-k',linewidth=3)

ax2.text(0.3,13.5,'car1=13-2t^2',size=10,color='k',bbox=bbox_props_time)
ax2.text(-7.8,13.5,'car2=13-2t',size=10,color='k',bbox=bbox_props_time)


plt.xlim(min(A1,A2)-1,max(A1,A2)+1)
plt.ylim(-2,14)
ax2.spines['left'].set_position('center')
ax2.spines['bottom'].set_position(('data',0))
ax2.xaxis.set_label_coords(0.5, -0.01)
ax2.yaxis.set_label_coords(-0.002, 0.5)
# plt.xticks(np.concatenate([np.arange(min(A1,A2)-1,0,1),np.arange(1,max(A1,A2)+2,1)]),size=tick_size)
plt.xticks(np.concatenate([np.arange(-7-1,0,1),np.arange(1,7+2,1)]),size=tick_size)
plt.yticks(np.concatenate([np.arange(-2,0,1),np.arange(1,15,1)]),size=tick_size)
plt.xlabel('meters [m]',fontsize=label_size)
plt.ylabel('meters [m]',fontsize=label_size)
plt.grid(True)

ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=True,blit=True)

plt.show()
