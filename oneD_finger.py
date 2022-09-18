import numpy as np
from numpy import heaviside
from functools import partial
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt


class OneDFinger:
    def __init__(
        self,
        m_finger=1.0,
        m_object=1.0,
        f_applied=None,
    ):
        """
        Args:
          m_finger (float) mass of finger
          m_object (float) masss of object
        """

        self.m_finger = m_finger
        self.m_object = m_object

    def F_a(self, t, t0=1.0, t1=2.0):
        """a particular applied force profile"""
        return heaviside(t - t0, 0.5) * heaviside(t1 - t, 0.5)

    def F_d(self, v, k=1.0):
        """simple drag Force"""
        return k * v**2

    def oneD_finger_n_obj(self, t, y):

        F_f = 0.0
        y1p = (self.F_a(t) - self.F_d(y[1])) / self.m_finger

        y3p = -F_f / self.m_object

        return [y[1], y1p, y[3], y3p]

    @staticmethod
    def touch(t, y):
        return y[0] - y[2]

    touch.terminal = True

    def oneD_finger_n_obj_merged(self, t, y):
        """objects joined.  Use same equation for both"""

        F_f = 0.0  # F_d(y[3])  # 0.1
        y1p = (self.F_a(t) - self.F_d(y[1]) - F_f) / (self.m_finger + self.m_object)

        y3p = (max(0.0, self.F_a(t) - self.F_d(y[3])) - F_f) / (
            self.m_finger + self.m_object
        )

        y_vector = [y[1], y1p, y[3], y3p]
        return y_vector


'''
def oneD_finger(t, y, F_a, m=1.0, k=1.0):
    y1p = (F_a(t) - k * (y[1] ** 2)) / m
    return [y[1], y1p]


def my_F_a(t, t0=1.0, t1=2.0):
    """a particular applied force profile"""
    return heaviside(t - t0, 0.5) * heaviside(t1 - t, 0.5)


oneD_finger_instance = partial(oneD_finger, F_a=my_F_a)


def F_d(v, k=1.0):
    """simple drag Force"""
    return k * v**2


def dlt(x, X, eps=1.0e-3):
    """delta(x > X) with fuzz"""
    return heaviside(x + eps - X, 0.5)  # 1.0 if (X - x) < eps else 0.0


def oneD_finger_n_obj(t, y, F_a, F_d, mf=1.0, mo=1.0):
    dd = dlt(y[0], y[2])

    F_f = 0.0  # F_d(y[3])  # 0.1
    y1p = (F_a(t) - F_d(y[1]) - dd * F_f) / (mf + dd * mo)

    y3p = (dd * (F_a(t) - F_d(y[1])) - F_f) / (dd * mf + mo)

    """
    # force object to have same x & x':
    if dd:
        y[3] = y[1]
        y3p = y1p
    """

    # Need impulse term: when dd turns from 0 -> 1, y[1] = y[3] = (mf*y[1] + mo*y[3])/(mf+mo) = v_CM for full stickiness

    if dd > oneD_finger_n_obj.dd0:
        print(f"Yo: {t} {y[1]} {y[3]}, {y[0]} {y[2]}")
        y[1] = y[3] = (mf * y[1] + mo * y[3]) / (mf + mo)  # set both to CM velocity
        print(f"Yo2: {t} {y[1]} {y[3]}, {y[0]} {y[2]}")
        # y3p = y1p # equate accelerations?
        oneD_finger_n_obj.dd0 = dd

    # print(f"   Yo3: {t:.4} {y[1]:.4} {y[3]:.4}, {y[0]:.4} {y[2]:.4}")

    return [y[1], y1p, y[3], y3p]


oneD_finger_n_obj_instance = partial(oneD_finger_n_obj, F_a=my_F_a, F_d=F_d)
oneD_finger_n_obj.dd0 = 0


def touch(t, y):
    return y[0] - y[2]


touch.terminal = True


def oneD_finger_n_obj_merged(t, y, F_a, F_d, mf=1.0, mo=1.0):
    """objects joined.  Use same equation"""

    F_f = 0.0  # F_d(y[3])  # 0.1
    y1p = (F_a(t) - F_d(y[1]) - F_f) / (mf + mo)

    y3p = (F_a(t) - F_d(y[3]) - F_f) / (mf + mo)

    y_vector = [y[1], y1p, y[3], y3p]
    return y_vector


oneD_finger_n_obj_merged_instance = partial(
    oneD_finger_n_obj_merged, F_a=my_F_a, F_d=F_d
)
'''

# sol_ola = solve_ivp(oneD_finger_instance, [0, 20], [0.0, 0.0], dense_output=True)
"""
# Solve until objects touch:
sol_ola = solve_ivp(
    oneD_finger_n_obj_instance,
    [0, 10],
    [0.0, 0.0, 1.5, 0.0],
    dense_output=True,
    events=touch,
)

t_end = sol_ola.t_events[0][0]
t = np.linspace(0, t_end, 20)
z = sol_ola.sol(t)

# calculate merger
y_ev = sol_ola.y_events[-1][0]
v_cm = 0.5 * (y_ev[1] + y_ev[3])

print("y_ev = ", y_ev)
print(" event = ", sol_ola.t_events[-1][0])

# Solve merged objects
sol_ola2 = solve_ivp(
    oneD_finger_n_obj_merged_instance,
    [sol_ola.t_events[-1][0], 10],
    [y_ev[0], v_cm, y_ev[2], v_cm],
    dense_output=True,
    # method="LSODA",
    # first_step=0.1,
    # events=touch,
)

t2 = np.linspace(sol_ola2.t[0], sol_ola2.t[-1], 20)
z2 = sol_ola2.sol(t2)

# t_end = sol_ola.t_events[0][0]
it_end = np.abs(t - t_end).argmin() + 1
print("event and solution:")
print(t_end)
print(sol_ola.sol(t_end))

# plt.plot(t[:it_end], z.T[:it_end, ...])
plt.plot(np.hstack([t, t2]), np.hstack([z, z2]).T)

# plt.xlim([0, sol_ola.t_events[0][0]])
# plt.plot(t, heaviside(1.0 - t, 0.5))

plt.show()
"""

if __name__ == "__main__":

    def F_a(self, t, t0=1.0, t1=2.0):
        """a particular applied force profile"""
        return heaviside(t - t0, 0.5) * heaviside(t1 - t, 0.5) + heaviside(
            t - 6.5, 0.5
        ) * heaviside(7.0 - t, 0.5)

    OneDFinger.F_a = F_a

    odf = OneDFinger()

    # Solve until objects touch:
    sol_ola = solve_ivp(
        odf.oneD_finger_n_obj,
        [0, 10],
        [0.0, 0.0, 1.5, 0.0],
        dense_output=True,
        events=odf.touch,
    )

    t_end = sol_ola.t_events[0][0]
    t = np.linspace(0, t_end, 20)
    z = sol_ola.sol(t)

    # calculate merger
    y_ev = sol_ola.y_events[-1][0]
    v_cm = 0.5 * (y_ev[1] + y_ev[3])

    print("y_ev = ", y_ev)
    print(" event = ", sol_ola.t_events[-1][0])

    # Solve merged objects
    sol_ola2 = solve_ivp(
        odf.oneD_finger_n_obj_merged,
        [sol_ola.t_events[-1][0], 10],
        [y_ev[0], v_cm, y_ev[2], v_cm],
        dense_output=True,
        # method="LSODA",
        # first_step=0.1,
        # events=touch,
    )

    t2 = np.linspace(sol_ola2.t[0], sol_ola2.t[-1], 20)
    z2 = sol_ola2.sol(t2)

    plt.plot(np.hstack([t, t2]), np.hstack([z, z2]).T)

    plt.show()
