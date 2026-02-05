'''
LICENSE AGREEMENT

In relation to this Python file:

1. Copyright of this Python file is owned by the author: Mark Misin

2. This Python code can be freely used and distributed

3. The copyright label in this Python file such as

copyright=ax_main.text(x,y,'© Mark Misin Engineering',size=z)

that indicate that the Copyright is owned by Mark Misin MUST NOT be removed.

WARRANTY DISCLAIMER!

This Python file comes with absolutely NO WARRANTY! In no event can the author of this Python file be held responsible for whatever happens in relation to this Python file. For example, if there is a bug in the code and because of that a project, invention, or whatever it is used for fails - the author is NOT RESPONSIBLE!
'''

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np

# Task landing
lab = 1
# Set simulation duration time
t0 = 0.0
t_end = 10.0
dt = 0.02
# Create array for time
t = np.arange(t0, t_end + dt, dt)

# Create array for position_x
start_displ = 0
pos_x_i = 2000
ascending_rate = 200
position_x = pos_x_i + ascending_rate * t

# Create array for altitude
ground = 0
pos_y_i = 1500
descending_rate = 100
position_y = pos_y_i - descending_rate * t

label = 'Displacement vector (black), which is the difference of position vectors (green)'

#####################################################################

# Task sinusoidal function
lab = 0
# Set simulation duration time
t0 = 0.0
t_end = 20.0
dt = 0.02
# Create array for time
t = np.arange(t0, t_end + dt, dt)

# Create array for position_x
start_displ = 0
pos_x_i = 2000
ascending_rate = 200
position_x = pos_x_i + ascending_rate * t

# Create array for altitude
ground = 0
pos_y_i = 1500
descending_rate = 100
sin_ampl = 500
freq = 1 / 20
position_y = pos_y_i + sin_ampl * np.sin(2 * np.pi * freq * t)
label = 'displacement vector (purple) = difference of 2 position vectors (green)'

####################################################################


# Set up the animation function
frame_amount = int(t_end / dt)
size_constant = 1

distance_x = np.zeros(len(t))
distance_y = np.zeros(len(t))
distance_tot = np.zeros(len(t))
distance_x[0] = pos_x_i
distance_y[0] = pos_y_i
for i in range(1, len(t)):
    distance_x[i] = distance_x[i - 1] + abs(position_x[i] - position_x[i - 1])
    if lab == 1:
        distance_y[i] = distance_y[i - 1] - abs(position_y[i] - position_y[i - 1])
    else:
        distance_y[i] = distance_y[i - 1] + abs(position_y[i] - position_y[i - 1])
distance_tot = abs(distance_x - pos_x_i) + abs(distance_y - pos_y_i)

# ------------------- CHANGES START HERE ------------------------
# Global list to keep track of arrow artists for removal each frame
arrow_artists = []

def update_plot(num):
    global arrow_artists
    # Remove old arrows
    for arrow in arrow_artists:
        arrow.remove()
    arrow_artists.clear()

    # Draw new arrows and store references
    displ_x = ax0.arrow(pos_x_i, pos_y_i, position_x[num] - pos_x_i, 0,
                       length_includes_head=True, head_width=40, head_length=80, color='r', linewidth=2)
    arrow_artists.append(displ_x)
    displ_y = ax0.arrow(position_x[num], pos_y_i, 0, position_y[num] - pos_y_i,
                       length_includes_head=True, head_width=40, head_length=80, color='b', linewidth=2)
    arrow_artists.append(displ_y)
    displ_x2 = ax2.arrow(t[num], pos_x_i, 0, position_x[num] - pos_x_i,
                        length_includes_head=True, head_width=0.2, head_length=100, color='r', linewidth=4)
    arrow_artists.append(displ_x2)
    displ_y2 = ax4.arrow(t[num], pos_y_i, 0, position_y[num] - pos_y_i,
                        length_includes_head=True, head_width=0.2, head_length=100, color='b', linewidth=4)
    arrow_artists.append(displ_y2)
    pos_R_1 = ax0.arrow(0, 0, pos_x_i, pos_y_i,
                       length_includes_head=True, head_width=40, head_length=80, color='g', linewidth=2)
    arrow_artists.append(pos_R_1)
    pos_R_2 = ax0.arrow(0, 0, position_x[num], position_y[num],
                       length_includes_head=True, head_width=40, head_length=80, color='g', linewidth=2)
    arrow_artists.append(pos_R_2)
    displ_R = ax0.arrow(pos_x_i, pos_y_i, position_x[num] - pos_x_i, position_y[num] - pos_y_i,
                       length_includes_head=True, head_width=40, head_length=80, color='m', linewidth=2)
    arrow_artists.append(displ_R)

    # Trajectory
    plane_trajectory.set_data(position_x[0:num], position_y[0:num])

    # Draw a plane
    plane_1.set_data([position_x[num] - 40 * size_constant, position_x[num] + 20 * size_constant],
                     [position_y[num], position_y[num]])
    plane_2.set_data([position_x[num] - 15 * size_constant, position_x[num] + 10],
                     [position_y[num], position_y[num]])
    plane_3.set_data([position_x[num] - 45 * size_constant, position_x[num] - 30 * size_constant],
                     [position_y[num] + 80 * size_constant, position_y[num]])
    plane_4.set_data([position_x[num] - 55 * size_constant, position_x[num] - 40 * size_constant],
                     [position_y[num], position_y[num]])

    # Connect plane with axes
    pos_x_indic_vert.set_data([t[num], t[num]],
                             [pos_x_i, position_x[num]])
    alt_indic_vert.set_data([t[num], t[num]],
                           [pos_y_i, position_y[num]])
    pos_x.set_data(t[0:num], position_x[0:num])
    pos_y.set_data(t[0:num], position_y[0:num])

    return (plane_trajectory, pos_x, pos_y, alt_indic_vert, pos_x_indic_vert,
            plane_1, plane_2, plane_3, plane_4,
            pos_R_1, pos_R_2, displ_R,
            displ_x, displ_y, displ_x2, displ_y2)
# ------------------- CHANGES END HERE ------------------------

# Set up your figure properties
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=(0.8, 0.8, 0.8))
gs = gridspec.GridSpec(2, 2)

# Airplane motion
ax0 = fig.add_subplot(gs[0, :], facecolor=(0.9, 0.9, 0.9))
plane_trajectory, = ax0.plot([], [], '-k', linewidth=2, label=label)

# Draw an airplane
plane_1, = ax0.plot([], [], 'k', linewidth=10)
plane_2, = ax0.plot([], [], 'w', linewidth=5)
plane_3, = ax0.plot([], [], 'k', linewidth=4)
plane_4, = ax0.plot([], [], 'w', linewidth=3)

plt.xlim(0, position_x[-1] + 10)
plt.ylim(ground, pos_y_i + pos_y_i / 3)
plt.xticks(np.arange(0, position_x[-1] + 1, (position_x[-1] - position_x[0]) / 4), size=15)
plt.yticks(np.arange(ground, pos_y_i + pos_y_i / 3 + 1, pos_y_i / 3), size=15)
plt.xlabel('position_x [m]', fontsize=15)
plt.ylabel('position_y [m]', fontsize=15)
plt.legend(loc='upper left', fontsize='x-large')
plt.grid(True)

# Copyright
copyright = ax0.text(0, (pos_y_i + pos_y_i / 3 + 1) * 3.2 / 3, '© Mark Misin Engineering', size=15)

# position_x function
ax2 = fig.add_subplot(gs[1, 0], facecolor=(0.9, 0.9, 0.9))
pos_x, = ax2.plot([], [], '-b', linewidth=3, label='Pos_x = ' + str(pos_x_i) + ' + ' + str(ascending_rate) + 't')
pos_x_indic_vert, = ax2.plot([], [], 'r', linewidth=2, label='dispacement_x vector')
plt.xlim(t0, t_end + 0.05)
plt.ylim(ground, position_x[-1] + 10)
plt.xticks(np.arange(t0, t_end + 0.5, 2), size=15)
plt.yticks(np.arange(0, position_x[-1] + 1, (position_x[-1] - position_x[0]) / 4), size=15)
plt.xlabel('time [s]', fontsize=15)
plt.ylabel('position_x [m]', fontsize=15)
plt.grid(True)
plt.legend(loc='lower right', fontsize='x-large')

# Altitude function
ax4 = fig.add_subplot(gs[1, 1], facecolor=(0.9, 0.9, 0.9))
if lab == 1:
    pos_y, = ax4.plot([], [], '-b', linewidth=3, label='Pos_y = ' + str(pos_y_i) + ' - ' + str(descending_rate) + 't')
else:
    pos_y, = ax4.plot([], [], '-b', linewidth=3, label='Pos_y = ' + str(pos_y_i) + ' + ' + str(sin_ampl) + '*sin(2π' + str(freq) + 't)')
alt_indic_vert, = ax4.plot([], [], 'b', linewidth=2, label='displacement_y vector')
plt.xlim(t0, t_end + 0.05)
plt.ylim(ground, pos_y_i + pos_y_i / 3 + 2000 + 1)
plt.xticks(np.arange(t0, t_end + 0.5, 2), size=15)
plt.yticks(np.arange(0, pos_y_i + pos_y_i / 3 + 2000 + 1, pos_y_i / 3), size=15)
plt.xlabel('time [s]', fontsize=15)
plt.ylabel('position_y [m]', fontsize=15)
plt.grid(True)
if lab == 1:
    plt.legend(loc='upper right', fontsize='x-large')
else:
    plt.legend(loc='upper left', fontsize='x-large')

plane_ani = animation.FuncAnimation(fig, update_plot,
                                    frames=frame_amount, interval=20, repeat=True, blit=True)

plt.show()
