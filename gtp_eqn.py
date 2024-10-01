from track_np import Splined_Track
import numpy as np
import config
import matplotlib.pyplot as plt
import cvxpy as cp
import time
import math
from scipy.optimize import curve_fit
from functools import partial

class SE_IBR:
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
        self.max_curvature = config.max_curvature

    def init_traj(self, i, p_0):
        """
        Initialize the trajectory at the start of the race.
        Assuming that the 1st waypoint is the start point.
        Simply a line following the tangent at the start point.

        :param i: Index of the current track frame
        :return: Initial trajectory
        """
        # a0 = [p_0[0], self.config.v_max, 0]
        # b0 = [p_0[1], self.config.v_max, 0]
        # print(a0, b0)
        Ai = np.zeros((self.n_steps, 3))
        Bi = np.zeros((self.n_steps, 3))
        for k in range(self.n_steps):
            idx, c0, t0, n0 = self.track.nearest_trackpoint(p_0)
            p_1 = p_0 + self.config.v_max * t0
            _, c1, t1, n1 = self.track.nearest_trackpoint(p_1)
            p_2 = p_1 + self.config.v_max * t1

            # fit a quadratic to the line between the two points give two eqn for each x and y
            eqn_x = np.polyfit([k, k+1, k+2], [p_0[0], p_1[0], p_2[0]], 2)
            eqn_y = np.polyfit([k, k+1, k+2], [p_0[1], p_1[1], p_2[1]], 2)
            # plt.show()
            Ai[k, :] = [eqn_x[2], eqn_x[1], eqn_x[0]]
            Bi[k, :] = [eqn_y[2], eqn_y[1], eqn_y[0]]
            p_0 = p_1
        return (Ai, Bi)

    def best_response(self, i, state, trajectories):
        j = (i + 1) % 2
        v_max = self.config.v_max
        a_max = self.config.a_max
        p_i = state[i]  # position of car i
        d_coll = 2 * self.config.collision_radius
        d_safe = 2 * self.config.collision_radius
        # p = cp.Variable((self.n_steps, 2))
        strat_A = cp.Variable((self.n_steps, 3)) # for x axis
        strat_B = cp.Variable((self.n_steps, 3)) # for y axis
        width = self.config.track_width

        # === hi(θi)=0 ===  Equality constraints only involving player i
        # Dynamic constraints:
        # (p_i)^k - (p_i)^(k-1) - v_i*dt = 0
        # cp.Zero is a constraint that the expression is equal to zero
        # TODO: cross check the use of cp.Zero and cp.SOC, AirSim uses cp.SOC
        # TODO check if it also satisfies velocity constraints
        # Continuity constraints, [x]
        continuity_constraints = []
        for k in range(self.n_steps - 1):
            a_At, b_At, c_At = strat_A[k, :]
            a_Bt, b_Bt, c_Bt = strat_B[k, :]
            a_Atp1, b_Atp1, c_Atp1 = strat_A[k + 1, :]
            a_Btp1, b_Btp1, c_Btp1 = strat_B[k + 1, :]

            # position continuity
            # [x_n(t_n+1), y_n(t_n+1)] == [x_n+1(t_n+1), y_n+1(t_n+1)] 
            continuity_constraints.append(
                (a_At + b_At * (k+1)  + c_At * (k+1)**2)
                - (a_Atp1 + b_Atp1 * (k + 1) + c_Atp1 * (k + 1) ** 2)
                == 0
            )   
            continuity_constraints.append(
                (a_Bt + b_Bt * (k+1) + c_Bt * (k+1)**2)
                - (a_Btp1 + b_Btp1 * (k + 1) + c_Btp1 * (k + 1) ** 2)
                == 0
            )

            # velocity continuity
            # [ux_n(t_n+1), uy_n(t_n+1)] == [ux_n+1(t_n+1), uy_n+1(t_n+1)]
            continuity_constraints.append(
                b_At + 2 * c_At * (k+1) - (b_Atp1 + 2 * c_Atp1 * (k + 1)) == 0
            )
            continuity_constraints.append(
                b_Bt + 2 * c_Bt * (k+1) - (b_Btp1 + 2 * c_Btp1 * (k + 1)) == 0
            )

        # pt1 = [strat_A[0, 0], strat_B[0, 0]]
        # continuity_constraints.append(p_i[0] - pt1[0] == 0)
        # continuity_constraints.append(p_i[1] - pt1[1] == 0)
        continuity_constraints.append(strat_A[0, 0] - p_i[0] == 0)
        continuity_constraints.append(strat_B[0, 0] - p_i[1] == 0)
        # === γ(θi, θj) <= 0 === Inequality involving both players
        # Collision constraints:
        # (p_i)^k - (p_j)^k <= di for all k
        # cp.SOC is a constraint that the expression is in the second order cone
        # cp.SOC(t, x) is equivalent to ||x||_2 <= t
        nc_constraints = []
        nc_obj = cp.Constant(0)
        nc_relax_obj = cp.Constant(0)
        non_collision_objective_exp = 0.5  # exponentially decreasing weight
        ks = np.arange(self.n_steps)
        # p_egos = np
        for k in range(self.n_steps):
            A_opp = trajectories[j][0][k, :]
            B_opp = trajectories[j][1][k, :]
            A_ego = trajectories[i][0][k, :]
            B_ego = trajectories[i][1][k, :]
            # A_ego = strat_A[k, :]
            # B_ego = strat_B[k, :]
            p_ego = [
                A_ego[0] + A_ego[1] *k + A_ego[2] *(k ** 2),
                B_ego[0] + B_ego[1] *k + B_ego[2] *(k ** 2),
            ]
            p_opp = [
                A_opp[0] + A_opp[1] *k + A_opp[2] *(k ** 2),
                B_opp[0] + B_opp[1] *k + B_opp[2] *(k ** 2),
            ]
            # Compute beta, the normalized direction vector pointing from the ego's drone position to the opponent's
            beta = [p_opp[0] - p_ego[0], p_opp[1] - p_ego[1]]
            # beta = cp.vstack(beta)
            if np.linalg.norm(beta, 2) >= 1e-6:
                # Only normalize if norm is large enough
                beta /= np.linalg.norm(beta)
            p_curr = [
                strat_A[k, 0] + strat_A[k, 1] * (k) + strat_A[k, 2] * (k ** 2),
                strat_B[k, 0] + strat_B[k, 1] * (k) + strat_B[k, 2] * (k ** 2),
            ]
            nc_constraints.append(beta.dot(p_opp) - beta.T @ p_curr >= d_coll)
            # TODO: See wtf is nc_obj and nc_relax_obj
            # For normal non-collision objective use safety distance
            nc_obj += (non_collision_objective_exp**k) * cp.pos(
                d_safe - (beta.dot(p_opp) - beta.T @ p_curr)
            )
            # For relaxed non-collision objective use collision distance
            nc_relax_obj += (non_collision_objective_exp**k) * cp.pos(
                d_coll - (beta.dot(p_opp) - beta.T @ p_curr)
            )

        # bound the constant term in the quadratic
        c_constraint = [strat_A[k, 0] <= 100 for k in range(self.n_steps)]
        c_constraint += [strat_A[k, 0] >= -100 for k in range(self.n_steps)]
        c_constraint += [strat_B[k, 0] <= 100 for k in range(self.n_steps)]
        c_constraint += [strat_B[k, 0] >= -100 for k in range(self.n_steps)]

        # === g(θi) <= 0 === Inequality constraints only involving player i
        # Speed constraints: v_i - v_max <= 0
        vel_constraints = []
        for k in range(self.n_steps):
            _, b_A, c_A = strat_A[k, :]
            _, b_B, c_B = strat_B[k, :]
            vel_x = b_A + 2 * c_A * k
            vel_y = b_B + 2 * c_B * k
            vel_constraints.append(vel_x**2 + vel_y**2 <= v_max**2)
        # Acceleration constraints: a_i - a_max <= 0
        acc_constraints = []
        for k in range(self.n_steps):
            t = k 
            _, _, c_A = strat_A[k, :]
            _, _, c_B = strat_B[k, :]
            acc_constraints.append(c_A**2 + c_B**2 <= a_max)    
    
        # TODO: curvature constraints
        # curvature constraints

        # === Track Constraints ===s
        track_constraints = []
        track_obj = cp.Constant(0)
        track_objective_exp = 0.5  # xponentially decreasing weight e
        t = np.zeros((self.n_steps, 2))  # tangent
        n = np.zeros((self.n_steps, 2))  # normal
        for k in range(self.n_steps):
            # query track indices at ego position
            A_ego = trajectories[i][0][k]
            B_ego = trajectories[i][1][k]
            p_cur = [
                A_ego[0] + A_ego[1] * k + A_ego[2] * k ** 2,
                B_ego[0] + B_ego[1] * k + B_ego[2] * k ** 2,
            ]
            _, c, t[k, :], n[k, :] = self.track.nearest_trackpoint(p_cur)
            p_new = [
                strat_A[k, 0] + strat_A[k, 1] * k + strat_A[k, 2] * k ** 2,
                strat_B[k, 0] + strat_B[k, 1] * k + strat_B[k, 2] * k ** 2,
            ]
            
            track_constraints.append(   
                n[k, :].T @ p_new - np.dot(n[k, :], c)
                <= width - self.config.collision_radius
            )   
            track_constraints.append(
                n[k, :].T @ p_new - np.dot(n[k, :], c)
                >= -(width - self.config.collision_radius)
            )

            # track constraints objective
            track_obj += (track_objective_exp**k) * (
                cp.pos(
                    n[k, :].T @ p_new
                    - np.dot(n[k, :], c)
                    - (width - self.config.collision_radius)
                )
                + cp.pos(
                    -(
                        n[k, :].T @ p_new
                        - np.dot(n[k, :], c)
                        - (width - self.config.collision_radius)
                    )
                )
            )

        # === "Win the Race" Objective ===
        # Take the tangent t at the last trajectory point
        # This serves as an approximation to the total track progress
        ns = self.n_steps
        pT = [
            strat_A[-1, 0] + strat_A[-1, 1] * (ns) + strat_A[-1, 2] * (ns) ** 2,
            strat_B[-1, 0] + strat_B[-1, 1] * (ns) + strat_B[-1, 2] * (ns) ** 2,
        ]
                        
        obj = -t[-1, :].T @ pT
        # create the problem in cxvpy and solve it
        prob = cp.Problem(
            cp.Minimize(obj),
            track_constraints + nc_constraints + vel_constraints + acc_constraints + c_constraint
            + continuity_constraints,
        )

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
                # assert prob.value >= 0.0
                if (prob.value < 0):
                    print("WARN: infeasible problem")
                    return trajectories[i]

                # Solve relaxed problem (track constraint -> track objective)
                relaxed_prob = cp.Problem(
                    cp.Minimize(
                        obj
                        + self.nc_weight * nc_obj
                        + self.track_relax_weight * track_obj
                    ),
                    nc_constraints
                    + vel_constraints
                    + acc_constraints
                    + continuity_constraints
                    + c_constraint,
                )
                relaxed_prob.solve()

                # relax non-collision constraints if problem is still  infeasible
                if np.isinf(relaxed_prob.value):
                    print("WARN: relaxing non collision constraints")
                    # If the problem is still infeasible, relax non-collision constraints
                    # Again, assert it is indeed an infeasible problem and not unbounded (in which case value is -inf).
                    # (The dynamical constraints keep the problem bounded.)
                    assert relaxed_prob.value >= 0.0

                    # Solve relaxed problem (non-collision constraint -> non-collision objective)
                    relaxed_prob = cp.Problem(
                        cp.Minimize(
                            obj
                            + self.nc_weight * nc_obj
                            + self.nc_relax_weight * nc_relax_obj
                        ),
                        track_constraints,
                    )
                    relaxed_prob.solve()

                    assert not np.isinf(relaxed_prob.value)
            trajectory_result = [strat_A.value, strat_B.value]
        # print why cvxpy failed
        except cp.error.SolverError as e:
            print(e)
            print(
                "WARN: cvxpy failre: resorting to initial trajectory (no collision constraints!)"
            )
            exit(0)
        trajectory_result = trajectories[i]
        return trajectory_result

    def iterative_br(self, i_ego, state, n_game_iterations=2, n_sqp_iterations=5):
        trajectories = [self.init_traj(i, state[i, :]) for i in [0, 1]]

        t0 = time.time()
        for i_game in range(n_game_iterations - 1):
            for i in [i_ego, (i_ego + 1) % 2]:
                for i_sqp in range(n_sqp_iterations - 1):
                    trajectories[i] = self.best_response(i, state, trajectories)
        # one last time for i_ego
        for i_sqp in range(n_sqp_iterations):
            trajectories[i_ego] = self.best_response(i_ego, state, trajectories)
        t1 = time.time()
        print("Total IBR solution time: ", t1 - t0)
        # return trajectories[i_ego]
        return np.array(trajectories)

import matplotlib.animation as animation

# = np.array([[0, 0], [0, 0]])
# ...
# Assuming SE_IBR and config are defined elsewhere in your code

if __name__ == "__main__":
    planner = SE_IBR(config)
    way_idx = 3
    ego_state = planner.track.waypoints[way_idx]
    opp_state = (
        planner.track.waypoints[way_idx]
        - planner.track.track_normals[way_idx] * 0.1
        - planner.track.track_tangent[way_idx] * 0.1
    )
    state = np.array([ego_state, opp_state])

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    planner.track.plot_track(ax, draw_boundaries=True)
    ego_point, = ax.plot([], [], "rx")
    opp_point, = ax.plot([], [], "bx")
    ego_path, = ax.plot([], [], "r")
    opp_path, = ax.plot([], [], "b")

    def init():
        ego_point.set_data([], [])
        opp_point.set_data([], [])
        ego_path.set_data([], [])
        opp_path.set_data([], [])
        return ego_point, opp_point, ego_path, opp_path

    def update_frame(i, planner, state):
        trajectory = planner.iterative_br(0, state)
        velocities = []
        pathi = []
        pathj = []
        pathi.append(state[0])
        pathj.append(state[1])
        for k in range(planner.n_steps):
            t = k
            (Ai, Bi), (Aj, Bj) = trajectory
            Ai, Bi, Aj, Bj = Ai[k], Bi[k], Aj[k], Bj[k]
            pathi.append(
                [Ai[0] + Ai[1] * t + Ai[2] * t**2, Bi[0] + Bi[1] * t + Bi[2] * t**2]
            )
            pathj.append(
                [Aj[0] + Aj[1] * t + Aj[2] * t**2, Bj[0] + Bj[1] * t + Bj[2] * t**2]
            )
            vix = Ai[1] + 2 * Ai[2] * t
            viy = Bi[1] + 2 * Bi[2] * t
            vi = math.sqrt(vix**2 + viy**2)
            velocities.append(vi)
        pathi = np.array(pathi)
        pathj = np.array(pathj)
        ego_point.set_data([state[0][0]], [state[0][1]])
        opp_point.set_data([state[1][0]], [state[1][1]])
        ego_path.set_data(pathi[:, 0], pathi[:, 1])
        opp_path.set_data(pathj[:, 0], pathj[:, 1])
        state[0] = pathi[2]
        state[1] = pathj[2]
        return ego_point, opp_point, ego_path, opp_path

    update_frame_partial = partial(update_frame, planner=planner, state=state)
    ani = animation.FuncAnimation(fig, update_frame_partial, frames=20, init_func=init, interval=200, blit=True)

    plt.show()