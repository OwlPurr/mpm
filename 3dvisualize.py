import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# read file
files = sorted(glob.glob("../output/pos_*.bin"))
positions = [np.fromfile(f, dtype=np.float32).reshape(-1, 3) for f in files]
n_frames = len(positions)

# style
M = 32  # GridSize
dt = 0.01  # Time step
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# color
particle_norm = Normalize(vmin=0, vmax=M)
particle_cmap = cm.turbo

# initialize
def init():
    ax.set_xlim(0, M)
    ax.set_ylim(0, M)
    ax.set_zlim(0, M)
    
    # Set axis labels and colors
    ax.set_xlabel('x [m]', color='white', fontsize=12)
    ax.set_ylabel('y [m]', color='white', fontsize=12)
    ax.set_zlabel('z [m]', color='white', fontsize=12)
    
    # Grid lines
    ax.grid(True, color='white', alpha=0.3)
    
    ax.set_title("Mochi simulation", fontsize=25, color='white')
    ax.xaxis.line.set_color('white')
    ax.yaxis.line.set_color('white')
    ax.zaxis.line.set_color('white')
    ax.tick_params(colors='white')

# update frame
def update(frame):
    ax.cla()
    init()

    pos = positions[frame]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    scatter = ax.scatter(
        x, y, z,
        s=50,
        c=z, 
        cmap=particle_cmap,
        norm=particle_norm,
        alpha=0.4,
        linewidths=0
    )

    # Time display
    time_sec = frame * dt
    ax.set_title(f"Frame {frame} | Time: {time_sec:.2f} [s]", fontsize=20, color='white')
    return scatter

# animation
anim = FuncAnimation(fig, update,
                    frames=n_frames,
                    interval=100,
                    init_func=init,
                    blit=False)

anim.save("../output/mochi.gif", writer="pillow", fps=10, dpi=50)
plt.close()

