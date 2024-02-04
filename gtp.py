from track import Splined_Track
import numpy as np
import config
import matplotlib.pyplot as plt
import cvxpy as cp
import time

class SE_IBR():
    def __init__(self, config):
        self.config = config
        self.dt = config.dt
        self.track = Splined_Track(config.track_waypoints, config.track_width)
        self.n_steps = config.n_steps
        self.traj = self.init_traj(0, self.track.waypoints[0])
        self.i_ego = 0

        # These are some parameters that could be tuned.
        # They control how the safety penalty and the relaxed constraints are weighted.
        self.nc_weight = 1
        self.nc_relax_weight = 128.0
        self.track_relax_weight = 128.0  # possibly should be the largest of the gains?

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

    def best_response(self, i, state, trajectories):
        j = (i + 1) % 2
        v_i = self.config.v_max
        a_i = self.config.a_max
        p_i = state[i] # position of car i
        p_j = state[j]
        d_coll = 2*self.config.collision_radius
        d_safe = 2*self.config.collision_radius
        p = cp.Variable((self.n_steps, 2))
        width = self.config.track_width

        # === hi(θi)=0 ===  Equality constraints only involving player i
        # Dynamic constraints:
        # (p_i)^k - (p_i)^(k-1) - v_i*dt = 0 
        # cp.Zero is a constraint that the expression is equal to zero
        #TODO: cross check the use of cp.Zero and cp.SOC, AirSim uses cp.SOC
        #TODO check if it also satisfies velocity constraints
        # ||p_0 - p[0]|| <= v*dt 
        init_dyn_constraint = cp.SOC(cp.Constant(v_i * self.dt), cp.Constant(state[i, :]) - p[0, :])
        # ||p[k+1] - p[k]|| <= v*dt
        dyn_constraints = [init_dyn_constraint] + [
            cp.SOC(cp.Constant(v_i * self.dt), p[k + 1, :] - p[k, :]) for k in range(self.n_steps - 1)]
        
        # === γ(θi, θj) <= 0 === Inequality involving both players
        # Collision constraints:
        # (p_i)^k - (p_j)^k <= di for all k
        # cp.SOC is a constraint that the expression is in the second order cone
        # cp.SOC(t, x) is equivalent to ||x||_2 <= t
        nc_constraints = []
        nc_obj = cp.Constant(0)
        nc_relax_obj = cp.Constant(0)
        non_collision_objective_exp = 0.5  # exponentially decreasing weight
        for k in range(self.n_steps):
            p_opp = trajectories[j][k, :]
            p_ego = trajectories[i][k, :]
            # Compute beta, the normal direction vector pointing from the ego's drone position to the opponent's
            beta = p_opp - p_ego
            if np.linalg.norm(beta) >= 1e-6:
                # Only normalize if norm is large enough
                beta /= np.linalg.norm(beta)
            #     n.T * (p_opp - p_ego) >= d_coll
            nc_constraints.append(beta.dot(p_opp) - beta.T @ p[k, :] >= d_coll)
            #TODO: See wtf is nc_obj and nc_relax_obj
            # For normal non-collision objective use safety distance
            nc_obj += (non_collision_objective_exp ** k) * cp.pos(d_safe - (beta.dot(p_opp) - beta.T @ p[k, :]))
            # For relaxed non-collision objective use collision distance
            nc_relax_obj += (non_collision_objective_exp ** k) * cp.pos(d_coll - (beta.dot(p_opp) - beta.T @ p[k, :]))
                                        
        

        # === g(θi) <= 0 === Inequality constraints only involving player i
        # Velocity constraints: v_i - v_max <= 0
        # vel_constraints = [cp.Zero(v_i - self.config.v_max)]
        # === Track Constraints ===s
        track_constraints = []
        track_obj = cp.Constant(0)
        track_objective_exp = 0.5  #xponentially decreasing weight e
        t = np.zeros((self.n_steps, 2)) # tangent
        n = np.zeros((self.n_steps, 2)) # normal
        for k in range(self.n_steps):
            # query track indices at ego position
            idx, c, t[k, :], n[k, :] = self.track.nearest_trackpoint(trajectories[i][k, :])
            # hortizontal track height constraints
            track_constraints.append(n[k, :].T @ p[k, :] - np.dot(n[k, :], c) <= width - self.config.collision_radius)
            track_constraints.append(n[k, :].T @ p[k, :] - np.dot(n[k, :], c) >= -(width - self.config.collision_radius))
            # track constraints objective
            track_obj += (track_objective_exp ** k) * (
                    cp.pos(n[k, :].T @ p[k, :] - np.dot(n[k, :], c) - (width - self.config.collision_radius)) +
                    cp.pos(-(n[k, :].T @ p[k, :] - np.dot(n[k, :], c) + (width - self.config.collision_radius))))
       
        # === "Win the Race" Objective ===
        # Take the tangent t at the last trajectory point
        # This serves as an approximation to the total track progress
        obj = -t[-1, :].T @ p[-1, :]
        # create the problem in cxvpy and solve it
        prob = cp.Problem(cp.Minimize(obj + self.nc_weight * nc_obj), dyn_constraints + track_constraints + nc_constraints)

        # try to solve proposed problem
        trajectory_result = np.array((self.n_steps, 2))
        try:
            prob.solve()
            # relax track constraints if problem is infeasible
            if np.isinf(prob.value):    
                print("WARN: relaxing track constraints")
                # If the problem is not feasible, relax track constraints
                # Assert it is indeed an infeasible problem and not unbounded (in which case value is -inf).
                # (The dynamical constraints keep the problem bounded.)
                assert prob.value >= 0.0

                # Solve relaxed problem (track constraint -> track objective)
                relaxed_prob = cp.Problem(cp.Minimize(obj + self.nc_weight * nc_obj + self.track_relax_weight * track_obj),
                                        dyn_constraints + nc_constraints)
                relaxed_prob.solve()

                # relax non-collision constraints if problem is still  infeasible
                if np.isinf(relaxed_prob.value):
                    print("WARN: relaxing non collision constraints")
                    # If the problem is still infeasible, relax non-collision constraints
                    # Again, assert it is indeed an infeasible problem and not unbounded (in which case value is -inf).
                    # (The dynamical constraints keep the problem bounded.)
                    assert relaxed_prob.value >= 0.0

                    # Solve relaxed problem (non-collision constraint -> non-collision objective)
                    relaxed_prob = cp.Problem(cp.Minimize(obj + self.nc_weight * nc_obj + self.nc_relax_weight * nc_relax_obj),
                                            dyn_constraints + track_constraints)
                    relaxed_prob.solve()

                    assert not np.isinf(relaxed_prob.value)
            trajectory_result = p.value
        except:  # if cvxpy fails, just return the initialized trajectory to do something
            print("WARN: cvxpy failre: resorting to initial trajectory (no collision constraints!)")
            trajectory_result = trajectories[i]
        return trajectory_result
    
    def iterative_br(self, i_ego, state, n_game_iterations=2, n_sqp_iterations=3):
        trajectories = [
            self.init_traj(i, state[i, :]) for i in [0, 1]
        ]   
        t0 = time.time()
        for i_game in range(n_game_iterations - 1):
            for i in [i_ego, (i_ego + 1) % 2]:
                for i_sqp in range(n_sqp_iterations - 1):
                    trajectories[i] = self.best_response(i, state, trajectories)
        # one last time for i_ego
        for i_sqp in range(n_sqp_iterations):
            trajectories[i_ego] = self.best_response(i_ego, state, trajectories)
        t1 = time.time()
        print('Total IBR solution time: ', t1 - t0)
        # return trajectories[i_ego]
        return np.array(trajectories)


if __name__ == "__main__":
    planner = SE_IBR(config)
    # state = np.array([planner.track.waypoints[0], planner.track.waypoints[0] + [0, 0.1]])
    way_idx = 4
    ego_state = planner.track.waypoints[way_idx]
    # move opponent along normal
    opp_state = planner.track.waypoints[way_idx] + planner.track.track_normals[way_idx]*0.1
    state = np.array([ego_state, opp_state])
    trajectory = planner.iterative_br(0, state)
    # print(trajectory)
    # Plot the track and the trajectory
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # planner.track.plot_waypoints_2d(ax)
    # plot initial position
    ax.plot(state[0, 0], state[0, 1], 'rx')
    ax.plot(state[1, 0], state[1, 1], 'gx')
    planner.track.plot_track(ax, draw_boundaries=True)
    # ax.plot(planner.traj[:, 0], planner.traj[:, 1])
    ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], 'r')
    ax.plot(trajectory[1, :, 0], trajectory[1, :, 1], 'g')
    plt.show()

