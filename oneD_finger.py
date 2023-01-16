import math
import numpy as np
from numpy import heaviside
from functools import partial
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

# from rich import print


class OneDFinger:
    def __init__(
        self,
        m_finger=1.0,
        m_object=1.0,
        K_f_drag=1.0,
        K_o_drag=0.02,
        K_elastic=0.5,
        F_static=1.0,
        f_kinetic=0.01,  # 0.1,
        x_merge_offset=1.0e-5,
        v_o_atol=1.0e-3,
        t_i: float = 0.0,
        t_f: float = 10.0,
        y0: np.ndarray = np.array([0.0, 1.5, 0.0, 0.0, 0.0]),
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

        If f_kinetic or x_merge_offset are too big, v_o_atol may need to be increased (e.g. 1e-3 -> 2.0e-3)

        Args:
          m_finger (float) mass of finger
          m_object (float) masss of object
          K_f_drag (float) drag coefficient for finger
          K_o_drag (float) drag coefficient for object
          K_elastic (float) elastic collision parameter (K = 1: elastic, K = 0: inelastic)
          F_static (float) static friction force when object is at rest
          f_kinetic (float) factor multiplied by incident force when object is moving to get kinetic friction force
          x_merge_offset (float) seperate f & o when they merge or bounce under tension
          v_o_atol (float) tolerance to determine if object is at rest, in which case static friction is used.
          t_i (float) initial time to start integration
          t_f (float) final time to end integration
          y0 (np.ndarray) initial values of [y_f, y_o, v_f, v_o, v_fo]
          N (int) number of points at which to evaluate each sub-solution
        """

        self.m_finger = m_finger
        self.m_object = m_object
        self.K_f_drag = K_f_drag
        self.K_o_drag = K_o_drag
        self.K_elastic = K_elastic
        self.F_static = F_static
        self.f_kinetic = f_kinetic
        self.x_merge_offset = x_merge_offset
        self.v_o_atol = v_o_atol

        self.t_i = t_i
        self.t_f = t_f
        self.N = N

        self.t0 = [t_i, t_f]
        self.y0 = y0

        self.step_str = "{:<8.3} {:>15}: v_o0 = {:<8.3} F_i - F_F = {:<8.3}"

    def v_cm(self, v_finger: float, v_object: float) -> float:
        """Return center of momentum of finger and object
        Args:
          v_finger (float) velocity of finger
          v_object (float) velocity of object
        Returns:
          v_cm (float) center of momentum velocity
        """
        return (self.m_finger * v_finger + self.m_object * v_object) / (
            self.m_finger + self.m_object
        )

    def F_a(self, t: float, t0: float = 1.0, t1: float = 2.0) -> float:
        """a particular applied force profile"""
        return heaviside(t - t0, 0.5) * heaviside(t1 - t, 0.5)

    def F_d(self, v: float) -> float:
        """simple drag Force"""
        return self.K_f_drag * v**2

    def F_friction(
        self, y: list[float], F_i: float, v_idx: int = 1, atol: float = 1.0e-3
    ) -> float:
        """Frictional force model
        Args:
          y (list) velocity for kinetic friction term
          F_i (float) incident force for static friction term
          v_idx (int) index of velocity used for friction 0: finger, 1: object, 2: finger-object
          atol (float) absolute tolerance for math.isclose()
        """
        x_f, x_o, _, v_o, v_fo = y

        v_f = y[2 + v_idx]

        # vectorized:
        # If not touching, no friction.  Otherwise, friction matches positive incident force up to static friction limit.
        # Subtract x_merge_offset from x_o to make "touching" more fuzzy, and thus more robust.
        static_f = np.where(
            np.less(x_f, x_o - self.x_merge_offset),
            0.0,
            np.maximum(0.0, np.minimum(self.F_static, F_i)),
        )
        # If object is not moving, friction is static.  Otherwise, friction is kinetic.
        f_f = np.where(
            np.isclose(v_f, 0.0, atol=atol),
            static_f,
            self.f_kinetic * static_f + self.K_o_drag * v_o,
        )
        vf_f = f_f

        return vf_f

    def F_sensor_func(self, t, y):
        """Calculate the force sensor reading at the finger tip.
        Args:
          t (array) time
          y (array) [x_finger, x_object, v_finger, v_object, v_finger-object]
          Returns:
            F_sensor (array) force sensor reading
        """

        F_i = self.F_a(t) - self.F_d(y[4])
        F_f = self.F_friction(y, F_i, v_idx=2)
        F_sensor = np.where(np.isclose(y[0], y[1], atol=1.0e-4), F_f, 0.0)

        # center-of-momentum velocity of the finger and object vs. time:
        vt_cm = np.where(
            np.isclose(y[0], y[1], atol=1.0e-3), y[4], self.v_cm(y[2], y[3])
        )
        # np.where(np.isclose(y[0], y[1], atol=1.e-3), 0.0, self.m_finger * (y[2] - vt_cm) * (1. + self.K_elastic))

        # relative velocity of the finger and object:
        dv = np.where(
            np.isclose(y[0], y[1], atol=1.0e-3), 0.0, np.diff(y[2:4, :], axis=0)
        )[0]

        # Find the indices where the finger and object connect and detach:
        i_connect = np.intersect1d(np.where(dv < 0.0)[0] + 1, np.where(dv == 0)[0])
        i_detach = np.intersect1d(np.where(dv > 0.0)[0] - 1, np.where(dv == 0)[0])

        # If connection is followed by detachment, then it will be a bounce:
        i_intersect = np.intersect1d(i_connect, i_detach - 1)

        # Connections followed by detachments are bounces, otherwise they are merges:
        i_bounce = np.intersect1d(i_connect, i_intersect)
        i_merge = np.setdiff1d(i_connect, i_intersect)
        print("Force sensor bounces: ", t[i_bounce], "merges: ", t[i_merge])

        for i_b in i_bounce:
            # (v_finger - v_cm) * (1 + K_elastic) is the impulse of the finger on the object:
            # = (v_finger - v_cm) going into the bounce + (v_finger - v_cm) * K_elastic coming out of the bounce.
            """
            print(
                "i_b = ",
                i_b,
                (y[2] - vt_cm)[i_b - 2 : i_b + 2],
                vt_cm[i_b - 2 : i_b + 2],
                t[i_b - 2 : i_b + 2],
                np.diff(t)[i_b - 1 : i_b + 2],
            )
            """
            f_sensor_bounce = (
                self.m_finger
                * (y[2] - vt_cm)[i_b]
                / np.sum(np.diff(t)[i_b - 1 : i_b + 2])
                * (1.0 + self.K_elastic)
            )
            F_sensor[i_b] += f_sensor_bounce
            if np.diff(t)[i_b] == 0.0:
                F_sensor[i_b + 1] += f_sensor_bounce

        for i_m in i_merge:

            """
            print(
                "i_m = ",
                i_m,
                (y[2] - vt_cm)[i_m - 2 : i_m + 2],
                vt_cm[i_m - 2 : i_m + 2],
                t[i_m - 2 : i_m + 2],
                np.diff(t)[i_m - 1 : i_m + 2],
            )
            """
            if vt_cm[i_m + 1] == 0.0:
                # If the object is immovable due to static friction, then it is effectively infinite mass,
                # so the impulse velocity change is just the finger velocity before the merger: y[2],
                # since the finger velocity after merger is 0.0.
                # So set center-of-momentum velocity of merger to 0.0:
                print(f"{i_m} merger cm velocity = 0.0")
                vt_cm[i_m] = 0.0
            f_sensor_merge = (
                self.m_finger
                * (y[2] - vt_cm)[i_m]
                / np.sum(np.diff(t)[i_m - 1 : i_m + 2])
            )
            F_sensor[i_m] += f_sensor_merge
            if np.diff(t)[i_m] == 0.0:
                F_sensor[i_m + 1] += f_sensor_merge

        return F_sensor

    def oneD_finger_n_obj(self, t: float, y: list[float]) -> list[float]:
        """Function to return derivatives (RHS) of ODEs to be solved.
        This function integrates the finger and object seperately.
        """

        F_i = self.F_a(t) - self.F_d(y[2])
        F_f = self.F_friction(y, F_i, v_idx=1)

        # v' (acceleration) of finger:
        v_fp = F_i / self.m_finger

        # v' of object:
        v_op = -self.F_friction(y, np.Inf, v_idx=1) / self.m_object

        # v' (acceleration) of finger + object:
        v_fop = (F_i - F_f) / (self.m_finger + self.m_object)

        return [y[2], y[3], v_fp, v_op, v_fop]

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

        # When merged, the drag velocity should be that of the f+o rather than f only:
        v_drag = y[4]  # y[2] would be finger velocity only
        F_i = self.F_a(t) - self.F_d(v_drag)
        F_f = self.F_friction(y, F_i, v_idx=2)

        # v' (accleration) of finger:
        v_fp = (self.F_a(t) - self.F_d(v_drag)) / self.m_finger

        # v' (acceleration) of object:
        v_op = -self.F_friction(y, F_i, v_idx=2) / self.m_object

        # v' (acceleration) of finger + object:
        v_fop = (F_i - F_f) / (self.m_finger + self.m_object)

        # print(
        #    f"  *Merged: t {t:.5}, y {y}, v_fp {v_fp:.5}, v_op {v_op:.5}, v_fop {v_fop:.5} F_i {F_i:.5} F_f {F_f:.5}"
        #
        #
        # )
        # y_vector = [y[1], y1p, y[3], y3p]
        return [y[4], y[4], v_fp, v_op, v_fop]  # y_vector

    def decel(self, t: float, y: list) -> float:
        """If root of force eqn passes thru zero, finger and object go from compression to tension.
        (Vice versa is excluded by initial conditions)
        """

        # Since decel() is only called when f+o are merged, use y[4] for v_fop, instead of y[2] for v_fp:
        F_i = self.F_a(t) - self.F_d(y[4])

        # Subtract small value so zero-crossing is slightly in tension, thus f+o touching w/o force won't be flagged:
        return F_i - 1.0e-5  # 1.0  # F_i - F_f + 1.0e-5  # foo

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

        x_f0, x_o0, v_f0, v_o0, v_fo0 = y0
        v_cm = self.v_cm(v_f0, v_o0)
        # If previous step was merged, us f+o velocity, v_fo0, for drag:
        v_drag = v_f0 if max(prev, 1) == 1 else v_fo0
        F_i = self.F_a(t0[0]) - self.F_d(v_drag)

        F_f = self.F_friction(y0, F_i, v_idx=max(prev, 1))
        # print(f"\t{t0[0]:.3} F_i = {F_i:.3}, v_o0 = {v_o0:.3}, F_f = {F_f:.3} y = {y0}")

        if prev == 0:
            print(self.step_str.format(t0[0], "Start", v_o0, F_i - F_f))
            ode = self.oneD_finger_n_obj
            t_range = t0
            y_init = y0
            events = self.touch
            state = 1
        elif prev == 1:

            if math.isclose(
                v_o0,
                0.0,
                abs_tol=self.v_o_atol,  # TEST if x_merge_offset is too big, this tolerance may need to be increased (e.g. 2.0e-3)
            ):  # abs(v_o0) < 1.0e-3  # v_o0 == 0.0:

                if F_i < F_f:
                    print(self.step_str.format(t0[0], "Static bounce", v_o0, F_i - F_f))
                    ode = self.oneD_finger_n_obj
                    v_f = -self.K_elastic * v_f0
                    v_o = 0.0
                    v_cmb = self.v_cm(v_f, v_o)
                    t_range = t0
                    y_init = [x_f0, x_o0, v_f, v_o, v_cmb]
                    events = self.touch
                    state = 1
                else:  # F_i >= F_f
                    # TEST: since it is low, reset v_o0 to zero and recalc F_f
                    # (Should this recalc of F_f happen before if F_i < F_f?)
                    v_o0 = 0.0
                    y0[3] = 0.0
                    F_f = self.F_friction(y0, F_i, v_idx=max(prev, 1))

                    print(self.step_str.format(t0[0], "Merge1", v_o0, F_i - F_f))
                    ode = self.oneD_finger_n_obj_merged
                    # Below static threshold, set velocity to zero:
                    if F_i == F_f:
                        v_cm = 0.0
                    t_range = t0
                    y_init = [x_f0, x_o0, v_cm, v_cm, v_cm]
                    events = self.decel
                    state = 2
            else:  # v_o0 != 0.0
                # F_min could be a small positive value corresponding to how momentum is inelasticly reflected.
                F_min = 0.0
                if F_i <= F_min:
                    print(self.step_str.format(t0[0], "Bounce", v_o0, F_i - F_f))
                    ode = self.oneD_finger_n_obj
                    # semi-elastic bounce:
                    v_f = v_cm + self.K_elastic * (v_cm - v_f0)
                    v_o = v_cm + self.K_elastic * (v_cm - v_o0)
                    t_range = t0
                    y_init = [x_f0 - self.x_merge_offset, x_o0, v_f, v_o, v_cm]
                    events = self.touch
                    state = 1
                else:  # F_i > F_min
                    print(self.step_str.format(t0[0], "Merge2", v_o0, F_i - F_f))
                    ode = self.oneD_finger_n_obj_merged
                    t_range = t0
                    y_init = [x_f0, x_o0, v_cm, v_cm, v_cm]
                    events = self.decel
                    state = 2

        elif prev == 2:
            # case where force between f & o has gone negative, but they are still merged (but not for long)
            print(
                self.step_str.format(t0[0], "Merged tention", v_o0, F_i - F_f)
                + f"  x_f0 = {x_f0:.3}, x_o0 = {x_o0:.3}, v_f0 = {v_f0:.5}, v_o0 = {v_o0:.5}, v_fo0 = {v_fo0:.5}"
            )
            # velocity of merged finger and object
            v_cm = v_fo0
            ode = self.oneD_finger_n_obj
            t_range = t0
            y_init = [x_f0 - self.x_merge_offset, x_o0, v_cm, v_cm, v_cm]
            events = self.touch
            state = 1

        # Solve until objects touch:
        sol_ola = solve_ivp(
            ode,
            t_range,
            y_init,
            dense_output=True,
            events=events,
            vectorized=True,
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

        # print(f"***t = {sol_ola.t}, message = {sol_ola.message}")
        # print(
        #    f"sol_ola.t_events = {sol_ola.t_events}, sol_ola.y_events = {sol_ola.y_events}"
        # )

        # return t_sol, y_sol, state, done
        return [t_end, t0[-1]], y_end, state, done, t_sol, y_sol, sol_ola

    def integrate(self):
        """
        Step thru time domain, t0, from initial condition y0.

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


if __name__ == "__main__":

    def F_a(self, t, t0=1.0, t1=5.0):
        """a particular applied force profile"""
        return 0.4 * heaviside(t - t0, 0.5) * heaviside(t1 - t, 0.5) + 2.0 * heaviside(
            t - 6.5, 0.5
        ) * heaviside(7.5 - t, 0.5)

    OneDFinger.F_a = F_a

    from scipy.interpolate import interp1d

    def F_a(self, t):
        """a particular applied force profile"""
        kind = "linear"  # 'cubic'
        end_first_push = 2.0  # 4.0
        f_a = interp1d(
            [
                0.0,
                1.0,
                end_first_push,
                end_first_push + 1.2,
                5.0 - 0.0 * 1.0,
                6.0,
                7.0,
                8.0,
                12.0,
            ],
            [0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            kind=kind,
        )(t)

        return f_a

    OneDFinger.F_a = F_a

    # Scale v_o_atol with f_kinetic
    f_kinetic = 0.4
    odf = OneDFinger(f_kinetic=f_kinetic, v_o_atol=f_kinetic * 2.0e-2)
    # odf = OneDFinger(m_object = 0.3, F_static = 0.5, F_kinetic = 0.03, x_merge_of

    t, y = odf.integrate()

    plt.grid(True)
    plt.plot(t, y.T)
    plt.plot(t, odf.F_a(t), "--")
    plt.plot(t, odf.F_sensor_func(t, y), "-")

    # plt.legend()
    plt.show()
