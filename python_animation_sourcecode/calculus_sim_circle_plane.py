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

####################################################################
# Task circle
# Set simulation duration time
t0 = 0.0
t_end = 20.0
dt = 0.02
# Create array for time
t = np.arange(t0, t_end + dt, dt)
radius = 50
freq = 1 / 10

# Create array for position_x
start_pos = 0
pos_x_i = 200
ascending_rate = 200
position_x = pos_x_i + 2 * radius * np.cos(2 * np.pi * freq * t)

# Create array for position_y
ground = 0
pos_y_i = 100
descending_rate = 100
position_y = pos_y_i + radius * np.sin(2 * np.pi * freq * t)

# Set up the animation function
frame_amount = int(t_end / dt)
size_constant = 0.2

# List to store arrow objects so we can remove them each frame
arrow_artists = []

def update_plot(num):
    """
    Animation update function. Removes old arrows, draws new ones,
    and updates trajectory/time indicators.
    This is a change compared to the older versions, in which this piece of code was not necessary.
    """
    # Remove old arrows
    global arrow_artists
    for arrow in arrow_artists:
        arrow.remove()
    arrow_artists.clear()

    # Draw new arrows and store them in arrow_artists
    pos_x1 = ax0.arrow(0, 0, 200, 100,
        length_includes_head=True, head_width=5, head_length=10, color='g', linewidth=3)
    arrow_artists.append(pos_x1)
    pos_x2 = ax0.arrow(200, 100, position_x[num] - pos_x_i, position_y[num] - pos_y_i,
        length_includes_head=True, head_width=5, head_length=10, color='m', linewidth=3)
    arrow_artists.append(pos_x2)
    pos_x3 = ax0.arrow(0, 0, position_x[num], position_y[num],
        length_includes_head=True, head_width=5, head_length=10, color='orange', linewidth=3)
    arrow_artists.append(pos_x3)
    pos_x1_x = ax0.arrow(0, 0, 200, 0,
        length_includes_head=True, head_width=5, head_length=10, color='g', linewidth=15)
    arrow_artists.append(pos_x1_x)
    pos_x1_y = ax0.arrow(0, 0, 0, 100,
        length_includes_head=True, head_width=5, head_length=10, color='g', linewidth=15)
    arrow_artists.append(pos_x1_y)
    pos_x2_x = ax0.arrow(200, 0, position_x[num] - pos_x_i, 0,
        length_includes_head=True, head_width=5, head_length=10, color='r', linewidth=8)
    arrow_artists.append(pos_x2_x)
    pos_x2_y = ax0.arrow(0, 100, 0, position_y[num] - pos_y_i,
        length_includes_head=True, head_width=5, head_length=10, color='b', linewidth=8)
    arrow_artists.append(pos_x2_y)
    pos_x3_x = ax0.arrow(0, 0, position_x[num], 0,
        length_includes_head=True, head_width=5, head_length=10, color='orange', linewidth=2)
    arrow_artists.append(pos_x3_x)
    pos_x3_y = ax0.arrow(0, 0, 0, position_y[num],
        length_includes_head=True, head_width=5, head_length=10, color='orange', linewidth=2)
    arrow_artists.append(pos_x3_y)

    # Update lines and trajectories as before
    plane_trajectory.set_data(position_x[0:num], position_y[0:num])
    pos_x_indic_horiz.set_data([pos_x_i, position_x[num]], [pos_y_i, pos_y_i])
    pos_y_indic_vert.set_data([pos_x_i, pos_x_i], [pos_y_i, position_y[num]])
    time_indic_horiz.set_data([t0, t[num]], [position_x[num], position_x[num]])
    pos_x_indic_vert.set_data([t[num], t[num]], [pos_x_i, position_x[num]])
    time_indic_horiz_2.set_data([ground, t[num]], [position_y[num], position_y[num]])
    pos_y_indic_vert_2.set_data([t[num], t[num]], [pos_y_i, position_y[num]])
    pos_x.set_data(t[0:num], position_x[0:num])
    pos_y.set_data(t[0:num], position_y[0:num])

    return (plane_trajectory, pos_x_indic_horiz, pos_x, pos_y, pos_y_indic_vert, pos_x_indic_vert,
            time_indic_horiz, time_indic_horiz_2, pos_y_indic_vert_2)


# Set up your figure properties
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=(0.8, 0.8, 0.8))
gs = gridspec.GridSpec(2, 2)

# Airplane motion
ax0 = fig.add_subplot(gs[0, :], facecolor=(0.9, 0.9, 0.9))
plane_trajectory, = ax0.plot([], [], '-k', linewidth=4)
pos_x_indic_horiz, = ax0.plot([], [], 'r:o', linewidth=2)
pos_y_indic_vert, = ax0.plot([], [], 'b:o', linewidth=2)
plt.xlim(0, 600)
plt.ylim(ground, 200)
plt.xticks(np.arange(0, 600 + 1, 600 / 4), size=15)
plt.yticks(np.arange(ground, 200 + 1, 200 / 4), size=15)
plt.xlabel('position_x [m]', fontsize=15)
plt.ylabel('position_y [m]', fontsize=15)
plt.grid(True)

# Copyright
copyright = ax0.text(0, 200 * 3.2 / 3, '© Mark Misin Engineering', size=15)

# position_x function
ax2 = fig.add_subplot(gs[1, 0], facecolor=(0.9, 0.9, 0.9))
pos_x, = ax2.plot([], [], '-b', linewidth=3, label='Pos_x = ' + str(pos_x_i) + ' + ' + str(2 * radius) + '*cos(2π' + str(freq) + 't)')
time_indic_horiz, = ax2.plot([], [], 'g:o', linewidth=2, label='time spent')
pos_x_indic_vert, = ax2.plot([], [], 'r:o', linewidth=2, label='amplitude')
plt.xlim(t0, t_end + 0.05)
plt.ylim(ground, 400)
plt.xticks(np.arange(t0, t_end + 0.5, 2), size=15)
plt.yticks(np.arange(0, 400 + 1, 400 / 4), size=15)
plt.xlabel('time [s]', fontsize=15)
plt.ylabel('position_x [m]', fontsize=15)
plt.grid(True)
plt.legend(loc='lower right', fontsize='x-large')

# position_y function
ax4 = fig.add_subplot(gs[1, 1], facecolor=(0.9, 0.9, 0.9))
pos_y, = ax4.plot([], [], '-b', linewidth=3, label='Pos_y = ' + str(pos_y_i) + ' + ' + str(radius) + '*sin(2π' + str(freq) + 't)')
time_indic_horiz_2, = ax4.plot([], [], 'g:o', linewidth=2, label='time spent')
pos_y_indic_vert_2, = ax4.plot([], [], 'b:o', linewidth=2, label='amplitude')
plt.xlim(t0, t_end + 0.05)
plt.ylim(ground, 400)
plt.xticks(np.arange(t0, t_end + 0.5, 2), size=15)
plt.yticks(np.arange(0, 400 + 1, 400 / 4), size=15)
plt.xlabel('time [s]', fontsize=15)
plt.ylabel('position_y [m]', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize='x-large')

plane_ani = animation.FuncAnimation(fig, update_plot,
    frames=frame_amount, interval=20, repeat=True, blit=True)

plt.show()
