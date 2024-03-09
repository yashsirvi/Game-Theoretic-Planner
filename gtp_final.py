from track import Splined_Track
import numpy as np
import config
import matplotlib.pyplot as plt
import cvxpy as cp
import time
import math

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
        self.max_curvature = config.max_curvature

    def init_traj(self, i, p_0):
        """
        Initialize the trajectory at the start of the race.
        Assuming that the 1st waypoint is the start point.
        Simply a line following the tangent at the start point.

        :param i: Index of the current track frame
        :return: Initial trajectory
        """
        a0 = [p_0[0], self.config.v_max, 0]
        b0 = [p_0[1], self.config.v_max, 0]
        print(a0, b0)
        Ai = np.zeros((self.n_steps, 3))
        Bi = np.zeros((self.n_steps, 3))
        for k in range(self.n_steps):
            t = k+1
            p = [a0[0] + a0[1]*t + a0[2]*t**2, b0[0] + b0[1]*t + b0[2]*t**2]
            idx, c, t, n = self.track.nearest_trackpoint(p)
            v_x = config.v_max * t[0]
            v_y = config.v_max * t[1]
            Ai[k] = [p[0], v_x, 0]
            Bi[k] = [p[1], v_y, 0]
        return (Ai, Bi)

 
    
    def best_response(self, i, state, trajectories):
        j = (i + 1) % 2
        v_max = self.config.v_max
        a_max = self.config.a_max
        p_i = state[i]  # position of car i
        p_j = state[j]  # position of car j
        d_coll = 2*self.config.collision_radius
        d_safe = 2*self.config.collision_radius
        # p = cp.Variable((self.n_steps, 2))
        strat_A = cp.Variable((self.n_steps, 3))
        strat_B = cp.Variable((self.n_steps, 3))
        width = self.config.track_width


        # === hi(θi)=0 ===  Equality constraints only involving player i
        # Dynamic constraints:
        # (p_i)^k - (p_i)^(k-1) - v_i*dt = 0 
        # cp.Zero is a constraint that the expression is equal to zero
        #TODO: cross check the use of cp.Zero and cp.SOC, AirSim uses cp.SOC
        #TODO check if it also satisfies velocity constraints
        # Continuity constraints, [x]
        continuity_constraints = []
        for k in range(self.n_steps - 1):
            t = k+1
            a_At, b_At, c_At = strat_A[k, :]
            a_Bt, b_Bt, c_Bt = strat_B[k, :]
            a_Atp1, b_Atp1, c_Atp1 = strat_A[k+1, :]
            a_Btp1, b_Btp1, c_Btp1 = strat_B[k+1, :]

            continuity_constraints.append(a_At + b_At*t + c_At*t**2 - (a_Atp1 + b_Atp1*(t+1) + c_Atp1*(t+1)**2) == 0)
            continuity_constraints.append(a_Bt + b_Bt*t + c_Bt*t**2 - (a_Btp1 + b_Btp1*(t+1) + c_Btp1*(t+1)**2) == 0)
            
        pt1 = [strat_A[0, 0]  + strat_A[0, 1] + strat_A[0, 2], strat_B[0, 0]  + strat_B[0, 1] + strat_B[0, 2]]
        dist1 = cp.norm(cp.vstack([p_i[0] - pt1[0], p_i[1] - pt1[1]]))
        continuity_constraints.append(v_max >= dist1)
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
            A_opp = trajectories[j][0][k, :]
            B_opp = trajectories[j][1][k, :]
            A_ego = trajectories[i][0][k, :]
            B_ego = trajectories[i][1][k, :]
            ts = k+1
            p_ego = [A_ego[0] + A_ego[1] * (ts) + A_ego[2] * (ts)**2, B_ego[0] + B_ego[1] * (ts) + B_ego[2] * (ts)**2]
            p_opp = [A_opp[0] + A_opp[1] * (ts) + A_opp[2] * (ts)**2, B_opp[0] + B_opp[1] * (ts) + B_opp[2] * (ts)**2]
            # Compute beta, the normal direction vector pointing from the ego's drone position to the opponent's
            beta = [p_opp[0] - p_ego[0], p_opp[1] - p_ego[1]]
            if np.linalg.norm(beta) >= 1e-6:
                # Only normalize if norm is large enough
                beta /= np.linalg.norm(beta)
            p_curr = [strat_A[k, 0] + strat_A[k,1] * (ts) + strat_A[k,2] * (ts)**2, strat_B[k,0] + strat_B[k,1] * (ts) + strat_B[k,2] * (ts)**2]
            nc_constraints.append(beta.dot(p_opp) - beta.T @ p_curr >= d_coll)
            #TODO: See wtf is nc_obj and nc_relax_obj
            # For normal non-collision objective use safety distance
            nc_obj += (non_collision_objective_exp ** k) * cp.pos(d_safe - (beta.dot(p_opp) - beta.T @ p_curr))
            # For relaxed non-collision objective use collision distance
            nc_relax_obj += (non_collision_objective_exp ** k) * cp.pos(d_coll - (beta.dot(p_opp) - beta.T @ p_curr))

        for k in range(self.n_steps):
            t = k+1
            a_A, b_A, c_A = strat_A[k, :]
            a_B, b_B, c_B = strat_B[k, :]
            oa_A, ob_A, oc_A = trajectories[j][0][k]
            oa_B, ob_B, oc_B = trajectories[j][1][k]
            dist = cp.norm(cp.vstack([a_A + b_A*t + c_A*t**2 - (oa_A + ob_A*t + oc_A*t**2), a_B + b_B*t + c_B*t**2 - (oa_B + ob_B*t + oc_B*t**2)]))
            nc_constraints.append(dist - d_coll <= 0)


        # === g(θi) <= 0 === Inequality constraints only involving player i
        # Speed constraints: v_i - v_max <= 0
        vel_constraints = []
        for k in range(self.n_steps):
            t = k+1 
            a_A, b_A, c_A = strat_A[k, :]
            a_B, b_B, c_B = strat_B[k, :]
            # v_i - v_max <= 0
            vel_x = b_A + 2*c_A*t
            vel_y = b_B + 2*c_B*t
            vel = cp.norm(cp.vstack([vel_x, vel_y]))
            vel_constraints.append(vel - v_max <= 0)
        # Acceleration constraints: a_i - a_max <= 0
        acc_constraints = []
        for k in range(self.n_steps):
            t = k+1
            a_A, b_A, c_A = strat_A[k, :]
            a_B, b_B, c_B = strat_B[k, :]
            # a_i - a_max <= 0
            acc_x = 2*c_A
            acc_y = 2*c_B
            acc = cp.norm(cp.vstack([acc_x, acc_y]))
            acc_constraints.append(acc - a_max <= 0)
        #TODO: curvature constraints
        # curvature constraints

        # === Track Constraints ===s
        track_constraints = []
        track_obj = cp.Constant(0)
        track_objective_exp = 0.5  #xponentially decreasing weight e
        t = np.zeros((self.n_steps, 2)) # tangent
        n = np.zeros((self.n_steps, 2)) # normal
        for k in range(self.n_steps):
            # query track indices at ego position
            A_ego = trajectories[i][0][k]
            B_ego = trajectories[i][1][k]
            ts = k+1
            p_cur = [A_ego[0] + A_ego[1] * (ts) + A_ego[2] * (ts)**2, B_ego[0] + B_ego[1] * (ts) + B_ego[2] * (ts)**2]
            _, c, t[k, :], n[k, :] = self.track.nearest_trackpoint(p_cur)
            p_new = [strat_A[k, 0] + strat_A[k,1] * (ts) + strat_A[k,2] * (ts)**2, strat_B[k,0] + strat_B[k,1] * (ts) + strat_B[k,2] * (ts)**2]
            
            track_constraints.append(n[k, :].T @ p_new - np.dot(n[k, :], c) <= width - self.config.collision_radius)
            track_constraints.append(n[k, :].T @ p_new - np.dot(n[k, :], c) >= -(width - self.config.collision_radius))

            # track constraints objective
            track_obj += (track_objective_exp ** k) * ( cp.pos(n[k, :].T @ p_new - np.dot(n[k, :], c) - (width - self.config.collision_radius)) +
                                                        cp.pos(-(n[k, :].T @ p_new - np.dot(n[k, :], c) + (width - self.config.collision_radius))))


        # === "Win the Race" Objective ===
        # Take the tangent t at the last trajectory point
        # This serves as an approximation to the total track progress
        ns = self.n_steps
        pT = [strat_A[-1,0] + strat_A[-1,1] * (ns) + strat_A[-1,2] * (ns)**2, strat_B[-1,0] + strat_B[-1,1] * (ns) + strat_B[-1,2] * (ns)**2]
        obj = -t[-1, :].T @ pT
        print("tanget:", t[-1, :].T, "pT:", pT)
        tangent = [[pT[0], pT[1]]]
        tangent +=[[pT[0] + t[-1, 0], pT[1] + t[-1, 1]]]
        # print(tangent)
        # plot the tangent
        # plt.plot(tangent[0], tangent[1], 'g')
        # plt.plot([pT[0], pT[0] + t[-1, 0]], [pT[1], pT[1] + t[-1, 1]], 'r')
        # print("OBJ:", obj)
        # print(self.nc_weight * nc_obj)
        # print(self.track_relax_weight * track_obj)
        # create the problem in cxvpy and solve it
        prob = cp.Problem(cp.Minimize(obj), 
                        track_constraints 
                        + nc_constraints
                        + vel_constraints
                        + acc_constraints
                        +continuity_constraints
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
                assert prob.value >= 0.0

                # Solve relaxed problem (track constraint -> track objective)
                relaxed_prob = cp.Problem(cp.Minimize(obj + self.nc_weight * nc_obj + self.track_relax_weight * track_obj),
                                        nc_constraints)
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
                                            track_constraints)
                    relaxed_prob.solve()

                    assert not np.isinf(relaxed_prob.value)
            trajectory_result = [strat_A.value, strat_B.value]
        # print why cvxpy failed
        except cp.error.SolverError as e:
            print(e)
            print("WARN: cvxpy failre: resorting to initial trajectory (no collision constraints!)")
            exit(0)
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
    # print(planner.track.track_normals[way_idx])
    opp_state = planner.track.waypoints[way_idx] + planner.track.track_normals[way_idx]*0.1 - planner.track.track_tangent[way_idx]*0.1
    state = np.array([ego_state, opp_state])
    print("STATE:",  state)
    for i in range(20):
        trajectory = planner.iterative_br(0, state)
        print(trajectory)
        pathi = []
        pathj = []
        pathi.append(state[0])
        pathj.append(state[1])
        for k in range(planner.n_steps):
            t = k+1
            (Ai, Bi), (Aj, Bj) = trajectory
            Ai, Bi, Aj, Bj = Ai[k], Bi[k], Aj[k], Bj[k]
            pathi.append([Ai[0] + Ai[1]*t + Ai[2]*t**2, Bi[0] + Bi[1]*t + Bi[2]*t**2])
            pathj.append([Aj[0] + Aj[1]*t + Aj[2]*t**2, Bj[0] + Bj[1]*t + Bj[2]*t**2])
        pathi = np.array(pathi)
        pathj = np.array(pathj)
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        planner.track.plot_track(ax, draw_boundaries=True)
        ax.plot(state[1][0], state[1][1], 'bx')
        ax.plot(state[0][0], state[0][1], 'rx')
        ax.plot(pathi[:, 0], pathi[:, 1], 'r')
        ax.plot(pathj[:, 0], pathj[:, 1], 'b')
        plt.show()
        state = np.array([pathi[-1], pathj[-1]])
