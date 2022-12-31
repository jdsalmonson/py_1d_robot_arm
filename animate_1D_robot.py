import numpy as np
from numpy import heaviside
from scipy.interpolate import interp1d
from oneD_finger import OneDFinger

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Rectangle
from matplotlib import animation
import matplotlib.collections as clt


def F_a(self, t, t0=1.0, t1=5.0):
    """a particular applied force profile"""
    return 0.4 * heaviside(t - t0, 0.5) * heaviside(t1 - t, 0.5) + 2.0 * heaviside(
        t - 6.5, 0.5
    ) * heaviside(7.5 - t, 0.5)


OneDFinger.F_a = F_a

f_kinetic = 0.4
odf = OneDFinger(f_kinetic=f_kinetic, v_o_atol=f_kinetic * 2.0e-2)
# odf = OneDFinger(m_object = 0.3, F_static = 0.5, F_kinetic = 0.03, x_merge_of

t, y = odf.integrate()

# Initialize the figure
n_frames = 100
interp_pos = interp1d(t, y[:2, :])
times = np.linspace(0, 10, n_frames)
x = interp_pos(times)
x_f = x[0, :]
x_o = x[1, :]

fig, axs = plt.subplots(3, 1, figsize=(8, 8))
axa = axs[0]
axp = axs[1]
axf = axs[2]

patches = []
e = Ellipse((x_f[0], 0), 1.0, 0.5, color="blue")
rect = Rectangle(
    (x_o[0], -0.5),
    0.5,
    1.0,
    color="green",
)
patches.append(e)
patches.append(rect)

collection = clt.PatchCollection(patches)
axa.add_collection(collection)
axa.set_xlim((-1, 5))
axa.set_ylim((-1, 1))
axa.set_aspect("equal")

axp.plot(t, y[:2, :].T)
vl = axp.axvline(times[0], color="red")
axp.grid(True)

axf.plot(t, odf.F_a(t), "--", label="F_applied")
axf.plot(t, odf.F_sensor_func(t, y), "-", color="magenta", label="F_sensor")
axf.grid(True)
axf.legend(loc="upper right")
axf.set_xlabel("time (s)")
vlf = axf.axvline(times[0], color="red")


def animate(i):

    patches = []
    e = Ellipse((x_f[i], 0), 1.0, 0.5)
    rect = Rectangle(
        (x_o[i], -0.5),
        0.5,
        1.0,
        color="green",
    )

    patches.append(e)
    patches.append(rect)
    collection.set_paths(patches)
    collection.set_facecolors(["blue", "green"])

    vl.set_xdata([times[i], times[i]])
    vlf.set_xdata([times[i], times[i]])


anim = animation.FuncAnimation(
    fig,
    animate,
    # init_func=init,
    frames=n_frames,
    interval=20,
    blit=False,
)

plt.show()
