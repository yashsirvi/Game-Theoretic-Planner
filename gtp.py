from track import Splined_Track
import numpy as np
import config
import matplotlib.pyplot as plt
import cvxpy as cp

class SE_IBR():
    def __init__(self, config):
        self.config = config
        self.dt = config.dt
        self.track = Splined_Track(config.track_waypoints, config.track_width)
        self.n_steps = config.n_steps
        self.traj = self.init_traj(0)

    def init_traj(self, i, p_0):
        """
        Initialize the trajectory at the start of the race.
        Assuming that the 1st waypoint is the start point.
        Simply a line following the tangent at the start point.

        :param i: Index of the current track frame
        :return: Initial trajectory
        """
        v_i = self.config.v_max
        traj = np.zeros(shape=(self.n_steps, 2))
        p = p_0
        for k in range(self.n_steps):
            idx, c, t, n = self.track.nearest_trackpoint(p)
            traj[k] = p
            p += self.dt*v_i*t
        return traj

    def best_response(self, i,j, state, trajectories):

        v_i = self.config.v_max
        a_i = self.config.a_max
        p_i = state[i] # position of car i
        p_j = state[j]

        p = cp.Variable((self.n_steps, 2))

        # === hi(θi)=0 ===  Equality constraints only involving player i
        # Dynamic constraints:
        # (p_i)^k - (p_i)^(k-1) - v_i*dt = 0 
        # cp.Zero is a constraint that the expression is equal to zero
        init_dyn_constraints = cp.Zero(p_i - p[0] - cp.Constant(v_i*self.dt)) # p[0] = p_i + v_i*dt
        init_dyn_constraints = [init_dyn_constraints]+[
                                cp.Zero(cp.Constant(v_i*self.dt) - p[k-1] + p[k])
                                for k in range(1, self.n_steps)]
        
        # === γ(θi, θj) <= 0 === Inequality involving both players
        # Collision constraints:
        # (p_i)^k - (p_j)^k <= di for all k
        # cp.SOC is a constraint that the expression is in the second order cone
        # cp.SOC(t, x) is equivalent to ||x||_2 <= t
        collision_constraints = [cp.SOC(cp.Constant(self.config.collision_radius),
                                        p_i[k] - p_j[k]) for k in range(self.n_steps)]

        # === g(θi) <= 0 === Inequality constraints only involving player i
        # Velocity constraints: v_i - v_max <= 0
        vel_constraints = 



if __name__ == "__main__":
    planner = SE_IBR(config)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    planner.track.plot_waypoints_2d(ax)
    planner.track.plot_track(ax, draw_boundaries=True)
    ax.plot(planner.traj[:, 0], planner.traj[:, 1])
    plt.show()

