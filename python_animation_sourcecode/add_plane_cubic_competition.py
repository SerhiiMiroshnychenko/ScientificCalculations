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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



class AddPlane:
    def __init__(self,finish_dist,step_dist,t):
        self.finish_dist=finish_dist
        self.step_dist=step_dist
        self.t=t
        self.div=800

    def create_arrays(self,power,altitude,a,b):
        distance=a*self.t**power+b
        height=np.ones(int(self.finish_dist/self.step_dist))*altitude

        # Create function for speed
        if power<1:
            self.t[0]=0.0025
        else:
            self.t[0]=0
        speed=power*a*self.t**(power-1)

        return distance,height,speed

    def create_plane(self,dist,plane_trajectory,plane_1,plane_2,plane_3,
        plane_4,plane_5,num,distance,height,speed):

        # Draw planes and their trajectories
        plane_trajectory.set_data(distance[0:num],height[0:num])

        # Draw a plane
        plane_1.set_data([distance[num]-40/self.div,distance[num]+20/self.div],
            [height[num],height[num]])
        # plane_2.set_data([distance[num]-20/self.div,distance[num]],
        #     [height[num]+0.3,height[num]])
        # plane_3.set_data([distance[num]-20/self.div,distance[num]],
        #     [height[num]-0.3,height[num]])
        # plane_4.set_data([distance[num]-40/self.div,distance[num]-30/self.div],
        #     [height[num]+0.15,height[num]])
        # plane_5.set_data([distance[num]-40/self.div,distance[num]-30/self.div],
        #     [height[num]-0.15,height[num]])

        # Distance and speed functions
        dist.set_data(self.t[0:num],distance[0:num])


    def initialize_stuff(self,ax,color):

        plane_trajectory,=ax.plot([],[],color,linewidth=4)
        # Initialize airplane
        plane_1,=ax.plot([],[],'k',linewidth=10)
        plane_2,=ax.plot([],[],'k',linewidth=5)
        plane_3,=ax.plot([],[],'k',linewidth=5)
        plane_4,=ax.plot([],[],'k',linewidth=3)
        plane_5,=ax.plot([],[],'k',linewidth=3)


        return plane_trajectory,plane_1,plane_2,plane_3,plane_4,plane_5

    def initialize_distance(self,fig,gs,gsR,gsC,color,t0,t_end,start_dist,finish_dist,power,a,b):
        ax=fig.add_subplot(gs[gsR,gsC], facecolor=(0.9,0.9,0.9))
        dist,=ax.plot([],[],color,linewidth=3,label='Pos = '+str(a)+'*t^'+str(power)+' + '+str(b))
        # dist,=ax.plot([],[],color,linewidth=3,label='D = '+str(a)+'*'+str(t_end)+'^'+str(power)+' + '+str(b)+' = '+str(a*t_end**power+b))
        plt.xlim(t0,t_end+0.05)
        plt.ylim(start_dist-12,finish_dist+50/self.div)
        plt.grid(True)
        if a>=0:
            plt.legend(loc='lower left',fontsize='x-large')
        else:
            plt.legend(loc='upper left',fontsize='x-large')
        return ax,dist
