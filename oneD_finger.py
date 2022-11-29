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
        K_elastic=0.5,
        F_f=0.0,
        f_applied=None,
        t_i: float = 0.0,
        t_f: float = 10.0,
        N: int = 20,
    ):
        """
        Solve 5 differential equations, for the position, y, or velocity, v, from acceleration, a,
        of the finger, f, or object, o, or their combined mass, fo.
          y_f' = v_f
          y_o' = v_o
          v_f' = a_f
          v_o' = a_o
          v_fo' = a_fo

        Args:
          m_finger (float) mass of finger
          m_object (float) masss of object
          K_elastic (float) elastic collision parameter (K = 1: elastic, K = 0: inelastic)
          F_f (float) frictional force on object
          t_i (float) initial time to start integration
          t_f (float) final time to end integration
          N (int) number of points at which to evaluate each sub-solution
        """

        self.m_finger = m_finger
        self.m_object = m_object
        self.K_elastic = K_elastic
        self.F_f = F_f

        self.t_i = t_i
        self.t_f = t_f
        self.N = N

        self.t0 = [t_i, t_f]
        # initial values of [y_f, y_o, v_f, v_o, v_fo]
        self.y0 = [0.0, 1.5, 0.0, 0.0, 0.0]

    def v_cm(self, v_finger: float, v_object: float) -> float:
        """Return center of momentum of finger and object
        Args:
          v_finger (float)
        """
        return (self.m_finger * v_finger + self.m_object * v_object) / (
            self.m_finger + self.m_object
        )

    def F_a(self, t: float, t0: float = 1.0, t1: float = 2.0) -> float:
        """a particular applied force profile"""
        return heaviside(t - t0, 0.5) * heaviside(t1 - t, 0.5)

    def F_d(self, v: float, k: float = 1.0) -> float:
        """simple drag Force"""
        return k * v**2

    def oneD_finger_n_obj(self, t: float, y: list[float]) -> list[float]:
        """Function to return derivatives (RHS) of ODEs to be solved.
        This function integrates the finger and object seperately.
        """

        # v' (acceleration) of finger:
        v_fp = (self.F_a(t) - self.F_d(y[2])) / self.m_finger

        # v' of object:
        v_op = -self.F_f * y[3] / self.m_object

        # v' of finger + object:
        # v_fop = (max(0.0, self.F_a(t) - self.F_d(y[2])) - F_f) / (self.m_finger + self.m_object)
        v_fop = (self.F_a(t) - self.F_d(y[2]) - self.F_f * y[3]) / (
            self.m_finger + self.m_object
        )

        return [y[2], y[3], v_fp, v_op, v_fop]  # [y[1], y1p, y[3], y3p]

    # @staticmethod
    def touch(self, t: float, y: list) -> float:
        return y[0] - y[1]  # y[0] - y[2]

    touch.terminal = True
    # This direction allows finger & object to separate before touching again.
    # It only flags on y[0] - y[1] starting negative (finger, 0, below object, 1)
    touch.direction = 1.0

    def oneD_finger_n_obj_merged(self, t: float, y: list[float]) -> list[float]:
        """Function to return derivatives (RHS) of ODEs to be solved.
        This function integrates the finger and object as if they were joined.
        Uses same equation for both.
        """

        # v' (accleration) of finger:
        v_fp = (self.F_a(t) - self.F_d(y[2])) / self.m_finger

        # v' of object:
        v_op = -self.F_f * y[3] / self.m_object

        # v_fop = (max(0.0, self.F_a(t) - self.F_d(y[2])) - F_f) / (
        #    self.m_finger + self.m_object
        # )
        # v' of finger + object:
        v_fop = (self.F_a(t) - self.F_d(y[2]) - self.F_f * y[3]) / (
            self.m_finger + self.m_object
        )

        # y_vector = [y[1], y1p, y[3], y3p]
        return [y[4], y[4], v_fp, v_op, v_fop]  # y_vector

    def decel(self, t: float, y: list) -> float:
        """If root of force eqn passes thru zero, finger and object go from compression to tension.
        (Vice versa is excluded by initial conditions)
        """
        return self.F_a(t) - self.F_d(y[2]) - self.F_f * y[3]

    decel.terminal = True

    def step(self, t0: list = None, y0: list = None, prev: int = 0):
        """Numerically integrate trajectories between object interaction events
        Args:
          t0 (list): time-span [t_start, t_end]
          y0 (list): initial values
          prev (int): previous state, 0 (start), 1: 2 separate objects, 2: merged objects
        Returns:
          t_sol, y_sol, next
        """
        if t0 is None:
            t0 = self.t0
        if y0 is None:
            y0 = self.y0

        if prev == 0:
            print(f"0 time: {t0[0]:.3}")
            ode = self.oneD_finger_n_obj
            t_range = t0
            y_init = y0
            events = self.touch
            state = 1
        elif prev == 1:
            v_cm = self.v_cm(y0[2], y0[3])
            if self.F_a(t0[0]) - self.F_d(y0[2]) - self.F_f * y0[3] > 0.0:
                print(f"2A {t0[0]:.3}")
                ode = self.oneD_finger_n_obj_merged
                t_range = t0
                y_init = [y0[0], y0[1], v_cm, v_cm, v_cm]
                events = self.decel
                state = 2
            else:
                print(f"1B {t0[0]:.3}")
                ode = self.oneD_finger_n_obj
                # semi-elastic bounce:
                v_f = v_cm + self.K_elastic * (v_cm - y0[2])
                v_o = v_cm + self.K_elastic * (v_cm - y0[3])
                t_range = t0
                y_init = [y0[0], y0[1], v_f, v_o, v_cm]
                events = self.touch
                state = 1
            # if force positive, setup 2
            # else set up 1
        elif prev == 2:
            # case where force between f & o has gone negative, but they are still merged (but not for long)
            print(f"1 {t0[0]:.3}")
            # velocity of merged finger and object
            v_cm = y0[4]
            ode = self.oneD_finger_n_obj
            t_range = t0
            y_init = [y0[0], y0[1], v_cm, v_cm, v_cm]
            events = self.touch
            state = 1

        # Solve until objects touch:
        sol_ola = solve_ivp(
            ode,
            t_range,
            y_init,
            dense_output=True,
            events=events,
        )

        if len(sol_ola.t_events[0]):
            t_end = sol_ola.t_events[0][-1]
            y_end = sol_ola.y_events[0][-1]
            done = True if t_end >= t0[-1] else False
        else:
            done = True
            t_end = t0[-1]
            y_end = sol_ola.sol(t_end)

        t_sol = np.linspace(t0[0], t_end, self.N)
        y_sol = sol_ola.sol(t_sol)

        return [t_end, t0[-1]], y_end, state, done, t_sol, y_sol, sol_ola

    def integrate(self):
        """
        Step thru time range, t0, from initial condition y0.

        Returns:
          t, y: tuple of time and y arrays to be plotted: plt.plot(t, y.T)
        """
        t = self.t0
        y = self.y0
        state = 0
        done = False

        t_sol_lst = []
        y_sol_lst = []
        while done is False:
            t, y, state, done, t_sol, y_sol, sol_ola = self.step(t, y, state)

            t_sol_lst.append(t_sol)
            y_sol_lst.append(y_sol)

        pass

        return np.hstack(t_sol_lst), np.hstack(y_sol_lst)


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

    def F_a(self, t, t0=1.0, t1=5.0):
        """a particular applied force profile"""
        return 0.4 * heaviside(t - t0, 0.5) * heaviside(t1 - t, 0.5) + heaviside(
            t - 6.5, 0.5
        ) * heaviside(7.5 - t, 0.5)

    OneDFinger.F_a = F_a

    odf = OneDFinger()

    # Solve until objects touch:
    sol_ola = solve_ivp(
        odf.oneD_finger_n_obj,
        [0, 10],
        odf.y0,  # [0.0, 0.0, 1.5, 0.0],
        dense_output=True,
        events=odf.touch,
    )

    t_end = sol_ola.t_events[0][0]
    t = np.linspace(0, t_end, 20)
    z = sol_ola.sol(t)

    # calculate merger
    y_ev = sol_ola.y_events[-1][0]
    # v_cm = 0.5 * (y_ev[1] + y_ev[3])
    v_cm = 0.5 * (y_ev[2] + y_ev[3])

    print("y_ev = ", y_ev)
    print(" event = ", sol_ola.t_events[-1][0])
    print("sol_ola events: ", sol_ola.t_events)

    # ---------------------
    # Solve merged objects
    sol_ola2 = solve_ivp(
        odf.oneD_finger_n_obj_merged,
        [sol_ola.t_events[-1][0], 10],
        # [y_ev[0], v_cm, y_ev[2], v_cm],
        [y_ev[0], y_ev[1], v_cm, v_cm, v_cm],
        dense_output=True,
        # method="LSODA",
        # first_step=0.1,
        events=odf.decel,
    )

    print("sol_ola2 events: ", sol_ola2.t_events)

    t2 = np.linspace(sol_ola2.t[0], sol_ola2.t[-1], 20)
    z2 = sol_ola2.sol(t2)

    # calculate unmerger
    y_ev = sol_ola2.y_events[-1][0]
    # v_cm = 0.5 * (y_ev[1] + y_ev[3])
    v_cm = y_ev[4]

    # ---------------------
    # Solve unmerged objects
    sol_ola3 = solve_ivp(
        odf.oneD_finger_n_obj,
        [sol_ola2.t_events[-1][0], 10],
        # [y_ev[0], v_cm, y_ev[2], v_cm],
        [y_ev[0], y_ev[1], v_cm, v_cm, v_cm],
        dense_output=True,
        # method="LSODA",
        # first_step=0.1,
        events=odf.touch,
    )

    print("sol_ola3 events: ", sol_ola3.t_events)
    print("sol_ola3 yevents: ", sol_ola3.y_events)

    t3 = np.linspace(sol_ola3.t[0], sol_ola3.t[-1], 40)
    z3 = sol_ola3.sol(t3)

    # calculate merger
    y_ev = sol_ola3.y_events[-1][0]
    v_cm = 0.5 * (y_ev[2] + y_ev[3])

    ## Calculating this merger depends on if the contact force, F_s = F_a - F_d - F_f > 0.
    ## If F_s <= 0 (i.e. coasting), then this merger should be a semi-elastic bounce
    ##  and the next equation should be the same: odf.oneD_finger_n_obj, i.e. solving two
    ## distinct objects again.

    ## As it is, the force is still on, so the following solve of merged eqns is correct.

    # ---------------------
    # Solve merged objects
    sol_ola4 = solve_ivp(
        odf.oneD_finger_n_obj_merged,
        [sol_ola3.t_events[-1][0], 10],
        # [y_ev[0], v_cm, y_ev[2], v_cm],
        [y_ev[0], y_ev[1], v_cm, v_cm, v_cm],
        dense_output=True,
        # method="LSODA",
        # first_step=0.1,
        events=odf.decel,
    )

    print("sol_ola4 events: ", sol_ola4.t_events)

    t4 = np.linspace(sol_ola4.t[0], sol_ola4.t[-1], 20)
    z4 = sol_ola4.sol(t4)

    plt.plot(np.hstack([t, t2, t3, t4]), np.hstack([z, z2, z3, z4]).T)

    # plt.legend()
    plt.show()
