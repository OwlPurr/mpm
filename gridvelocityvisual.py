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

# 格子速度データの読み込み
grid_files = sorted(glob.glob("../output/grid_*.bin"))
grid_velocities = [np.fromfile(f, dtype=np.float32).reshape(-1, 3) for f in grid_files]

# 格子拡散データの読み込み
div_files = sorted(glob.glob("../output/div_*.bin"))
div = [np.fromfile(f, dtype=np.float32).reshape(-1) for f in div_files]
print(abs(np.max(div)))

# 格子中心位置を計算（32x32x32）
grid_size = M
grid_coords = np.array([
    [i + 0.5, j + 0.5, k + 0.5]
    for i in range(grid_size)
    for j in range(grid_size)
    for k in range(grid_size)
])

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

    # 格子速度の表示をカラー点群で代用
    velocity = grid_velocities[frame]
    coords = grid_coords

    # 速度ノルム（L2）を計算
    velocity_norm = np.linalg.norm(velocity, axis=1)

    # 断面フィルタ：z座標が M/2 以下の点のみ使用
    mask = coords[:, 1] > (M / 2)
    coords_ = coords[mask]
    norm_ = velocity_norm[mask]

    div_values = div[frame]
    div_ = div_values[mask]

    # 色付け用 colormap
    norm_scale = Normalize(vmin=np.min(div)/4, vmax=np.max(div)/4)  
    grid_cmap = cm.plasma  

    ax.scatter(
        coords_[:, 0], coords_[:, 1], coords_[:, 2],
        s=40,
        c=div_,
        cmap=grid_cmap,
        norm=norm_scale,
        alpha=0.6
    )
    '''
    # 粒子の表示
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
    '''
    # タイトル
    time_sec = frame * dt
    ax.set_title(f"Frame {frame} | Time: {time_sec:.2f} [s]", fontsize=20, color='white')
    return #scatter

# animation
anim = FuncAnimation(fig, update,
                    frames=40,
                    interval=100,
                    init_func=init,
                    blit=False)

anim.save("../output/mochi.gif", writer="pillow", fps=15, dpi=150)
plt.close()