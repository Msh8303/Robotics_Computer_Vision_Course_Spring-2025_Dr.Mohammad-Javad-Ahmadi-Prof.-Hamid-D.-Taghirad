import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# Rotation matrices
# -----------------------------
def Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca]
    ])

def Ry(b):
    cb, sb = np.cos(b), np.sin(b)
    return np.array([
        [cb, 0, sb],
        [0, 1, 0],
        [-sb, 0, cb]
    ])

def rotation_universal(alpha, beta):
    return Rx(alpha) @ Ry(beta)

# -----------------------------
# Parameters
# -----------------------------
L = 1.0
deg = np.pi / 180.0
max_angle = 10 * deg

frames = 120
t = np.linspace(0, 2*np.pi, frames)
alpha_traj = max_angle * np.sin(t)
beta_traj  = max_angle * np.cos(t)

e_x = np.array([1, 0, 0])
e_y = np.array([0, 1, 0])
e_z = np.array([0, 0, 1])

# -----------------------------
# Figure
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def init_ax():
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Universal Joint - Two Orthogonal Revolute Joints')

def draw_base_axes():
    ax.quiver(0,0,0, 1,0,0, color='r', arrow_length_ratio=0.08)
    ax.quiver(0,0,0, 0,1,0, color='g', arrow_length_ratio=0.08)
    ax.quiver(0,0,0, 0,0,1, color='b', arrow_length_ratio=0.08)

init_ax()

# -----------------------------
# Update function
# -----------------------------
def update(i):
    ax.cla()  # پاک‌سازی استاندارد و ایمن
    init_ax()
    draw_base_axes()

    a = alpha_traj[i]
    b = beta_traj[i]
    R = rotation_universal(a, b)

    x_axis = R @ e_x
    y_axis = R @ e_y
    z_axis = R @ e_z

    ax.quiver(0,0,0, *x_axis, color='m', arrow_length_ratio=0.08, linewidth=2)
    ax.quiver(0,0,0, *y_axis, color='c', arrow_length_ratio=0.08, linewidth=2)
    ax.quiver(0,0,0, *z_axis, color='k', arrow_length_ratio=0.08, linewidth=3)

    tool_pts = np.array([[0,0,0], L*z_axis])
    ax.plot(tool_pts[:,0], tool_pts[:,1], tool_pts[:,2], 'k-', lw=4)

    ax.text2D(
        0.05, 0.9,
        f"alpha = {a/deg:.2f} deg\nbeta  = {b/deg:.2f} deg",
        transform=ax.transAxes
    )

    return []

# -----------------------------
# Animation
# -----------------------------
anim = FuncAnimation(fig, update, frames=frames, interval=70, blit=False)

gif_path = "universal_joint.gif"  # مسیر دلخواه
anim.save(gif_path, writer=PillowWriter(fps=15))

print("Saved GIF to:", gif_path)
