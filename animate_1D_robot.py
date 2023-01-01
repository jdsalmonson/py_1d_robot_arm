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

x_min = np.min(x)
x_max = np.max(x)

interp_f_sensor = interp1d(t, odf.F_sensor_func(t, y))

fig, axs = plt.subplots(3, 1, figsize=(8, 8))
axa = axs[0]
axp = axs[1]
axf = axs[2]

patches = []
e_x = 1.0
e = Ellipse((x_f[0] - 0.5 * e_x, 0.0), e_x, 0.5, color="blue")
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
axa.set_xlim((x_min - 0.5 - e_x, x_max + 0.5))
axa.set_ylim((-1, 1))
axa.set_aspect("equal")
axa.get_yaxis().set_visible(False)

# position plot
axp.plot(t, y[0, :].T, color="blue")
axp.plot(t, y[1, :].T, color="green")
vl = axp.axvline(times[0], color="red")
(xfp,) = axp.plot(times[0], x_f[0], "o", color="blue")
(xop,) = axp.plot(times[0], x_o[0], "o", color="green")
axp.grid(True)

# force plot
axf.plot(t, interp_f_sensor(t), "-", color="magenta", label="F_sensor")
axf.plot(t, odf.F_a(t), "--", color="cyan", label="F_applied")
vlf = axf.axvline(times[0], color="red")
(fas,) = axf.plot(times[0], interp_f_sensor(times[0]), "o", color="magenta")
(fap,) = axf.plot(times[0], odf.F_a(times[0]), "o", color="cyan")
axf.grid(True)
axf.legend(loc="upper right")
axf.set_xlabel("time (s)")


def animate(i):

    patches = []
    e = Ellipse((x_f[i] - 0.5 * e_x, 0), e_x, 0.5)
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

    # update the position plot
    vl.set_xdata([times[i], times[i]])
    xfp.set_data(times[i], x_f[i])
    xop.set_data(times[i], x_o[i])
    # update the force plot
    vlf.set_xdata([times[i], times[i]])
    fas.set_data(times[i], interp_f_sensor(times[i]))
    fap.set_data(times[i], odf.F_a(times[i]))


anim = animation.FuncAnimation(
    fig,
    animate,
    # init_func=init,
    frames=n_frames,
    interval=20,
    blit=False,
)

save_as = "gif"  # "html"  # "gif"
if save_as == "html":
    # Make HTML output: -------------------
    from matplotlib.animation import HTMLWriter
    import matplotlib

    # Increase size limit for html file:

    matplotlib.rcParams["animation.embed_limit"] = 2**32  # 128
    anim.save("oneD_robot.html", writer=HTMLWriter(embed_frames=True))

    # To open file in web browser:
    # > xdg-open symmetric_pendulum.html
    # --------------------------------------
elif save_as == "gif":
    # Make GIF output: -------------------
    from matplotlib.animation import PillowWriter

    anim.save("oneD_robot.gif", writer=PillowWriter(fps=30))
    # --------------------------------------
else:
    plt.show()
