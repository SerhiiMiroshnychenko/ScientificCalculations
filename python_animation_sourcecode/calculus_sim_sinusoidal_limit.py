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
t_end=4.0
dt=0.01
f1=0.01 # Frequency
f2=0.01
A1=7 # Amplitude
A2=-7
delay_car1=1
delay_car2=0
frame_amount=int(t_end/dt+2) # Don't forget to chang frame rate
bbox_props_time=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='k',lw=1.0)
# For the graph parameters
tick_size=10
label_size=10
########################################################################

# Create array for time
last_value=0.01
t=np.arange(t_end,t0,-dt)
t=np.concatenate([t,[last_value]])

# # Sampling time explanation
# dtb=0.2
# dtr=0.001
# frame_amount=int(t_end/dtr+2)
# net_frame_amount=frame_amount-int(t_end/dtb+2)
# tb=np.arange(t_end,t0,-dtb)
# tb=np.concatenate([tb,[0.001],np.ones(net_frame_amount)*last_value])
# tr=np.arange(t_end,t0,-dtr)
# tr=np.concatenate([tr,[0.001]])
# D_h1=A1*np.sin(1/(2*np.pi*f1*tb))#+A1/7*np.cos(2*np.pi*f1*7*t) # Chaotic
# D_h2=-A2*np.sin(1/(2*np.pi*f2*tr))


# # Deviation (D)
D_h1=A1*np.sin(1/(2*np.pi*f1*t))#+A1/7*np.cos(2*np.pi*f1*7*t) # Chaotic
D_h2=-A2*np.sin(1/(2*np.pi*f2*t))
D_h1u=np.concatenate([D_h1,[0.0]]) # Update, one extra element for blocks
D_h2u=np.concatenate([D_h2,[0.0]])



def update_plot(num):
    # len(num)=122 -> num=0-121 -> t[0:121]=t[0]->t[120]

    # Update sinusoidal functions
    y_1.set_data(t[0:num],D_h1[0:num])
    y_2.set_data(t[0:num],D_h2[0:num])

    # # Update sinusoidal functions: sampling time explanation
    # y_1.set_data(tb[0:num],D_h1[0:num])
    # y_2.set_data(tr[0:num],D_h2[0:num])

    # Update the blocks
    block_1.set_data([D_h1u[num]-0.45,D_h1u[num]+0.45],[3.5,3.5])
    block_2.set_data([D_h2u[num]-0.45,D_h2u[num]+0.45],[1.5,1.5])


    # Update the block functions
    block_1f.set_data([-0.05,-0.05],[D_h1u[num]-0.9,D_h1u[num]+0.9])
    block_2f.set_data([-0.05,-0.05],[D_h2u[num]-0.9,D_h2u[num]+0.9])

    return  y_1,block_1,block_1f,\
            y_2,block_2,block_2f

# Set up your figure properties
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(1,2)
plt.subplots_adjust(left=0.03,bottom=0.035,right=0.99,top=0.97,wspace=0.035,hspace=0.2)

# Function
ax1=fig.add_subplot(gs[:,0],facecolor=(0.9,0.9,0.9))
y_1,=ax1.plot([],[],'-b',linewidth=2,label='lim_t->0[D_h1 = '+str(A1)+'*sin(1/(2π*'+str(f1)+'*t))]=DOES NOT EXIST')
y_2,=ax1.plot([],[],'--r',linewidth=2,label='lim_t->0[D_h2 = '+str(-A2)+'*sin(1/(2π*'+str(f2)+'*t))]=DOES NOT EXIST')
block_1f,=ax1.plot([],[],'-b',linewidth=5)
block_2f,=ax1.plot([],[],'-r',linewidth=5)


plt.xlim(t0-0.25,t_end)
# plt.xlim(t0-0.25,t_end+5*dt)
plt.ylim(min(A1,A2)-1,max(A1,A2)+1)
ax1.spines['bottom'].set_position('center')
ax1.spines['left'].set_position(('data',0))
ax1.xaxis.set_label_coords(0.5, -0.01)
plt.yticks(np.concatenate([np.arange(min(A1,A2)-1,0,1),np.arange(1,max(A1,A2)+2,1)]),size=tick_size)
plt.yticks(np.concatenate([np.arange(-7-1,0,1),np.arange(1,7+2,1)]),size=tick_size)
plt.xlabel('seconds [s]',fontsize=label_size)
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1.01,1.01),fontsize='medium')
copyright=ax1.text(0,(max(A1,A2)+1)*1.01,'© Mark Misin Engineering',size=10)

# Action
ax2=fig.add_subplot(gs[:,1],facecolor=(0.9,0.9,0.9))
block_1,=ax2.plot([],[],'-b',linewidth=36)
block_2,=ax2.plot([],[],'-r',linewidth=36)
plt.xlim(min(A1,A2)-1,max(A1,A2)+1)
plt.ylim(-2,14)
ax2.spines['left'].set_position('center')
ax2.spines['bottom'].set_position(('data',0))
ax2.xaxis.set_label_coords(0.5, -0.01)
ax2.yaxis.set_label_coords(-0.002, 0.5)
plt.xticks(np.concatenate([np.arange(min(A1,A2)-1,0,1),np.arange(1,max(A1,A2)+2,1)]),size=tick_size)
plt.yticks(np.concatenate([np.arange(-2,0,1),np.arange(1,15,1)]),size=tick_size)
plt.xlabel('meters [m]',fontsize=label_size)
plt.ylabel('meters [m]',fontsize=label_size)
plt.grid(True)

ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=False,blit=True)

plt.show()
