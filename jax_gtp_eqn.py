from track import Splined_Track
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
import config
import matplotlib.pyplot as plt
import cvxpy as cp
import time
import math
from scipy.optimize import curve_fit
from functools import partial
import timeit
from functools import partial
import mdmm_jax
import optax

class SE_IBR:
    def __init__(self, config):
        self.config = config
        self.dt = config.dt
        self.track = Splined_Track(config.track_waypoints, config.track_width)
        self.n_steps = config.n_steps
        self.traj = self.init_traj(self.track.waypoints[0])
        self.i_ego = 0

        # These are some parameters that could be tuned.
        # They control how the safety penalty and the relaxed constraints are weighted.
        self.nc_weight = 1
        self.nc_relax_weight = 128.0
        self.track_relax_weight = 128.0  # possibly should be the largest of the gains?
        self.max_curvature = config.max_curvature
        self.v_max = self.config.v_max
        self.a_max = self.config.a_max
        self.d_coll = 2 * self.config.collision_radius
        self.d_safe = 2 * self.config.collision_radius
        self.t_width = self.track.track_width
        self.prngkey = jax.random.PRNGKey(0)  # Initialize a random key
        self.optimizer = optax.chain(optax.sgd(1e-3), mdmm_jax.optax_prepare_update())
        
    def init_traj(self, p_0):
        """
        Initialize the trajectory at the start of the race.
        Assuming that the 1st waypoint is the start point.
        Simply a line following the tangent at the start point.

        :param p_0: Start point of the trajectory
        :return: Initial trajectory
        """
        Ai = np.zeros((self.n_steps, 3))
        Bi = np.zeros((self.n_steps, 3))
    
        
        for k in range(self.n_steps):
            _, _, t0, _ = self.track.nearest_trackpoint(p_0)
            p_1 = p_0 + self.config.v_max * t0
            # _, c1, t1, n1 = self.track.nearest_trackpoint(p_1)
            p_2 = p_1 + self.config.v_max * t0 * 2
            
            ks = np.array([k, k+1.0, k+2.0])
            psx = np.array([p_0[0], p_1[0], p_2[0]])
            psy = np.array([p_0[1], p_1[1], p_2[1]])
            eqn_x = np.polyfit(ks, psx, 2) # TAKES ABOUT 1e-4s
            eqn_y = np.polyfit(ks, psy, 2)
            
            Ai[k, :] = eqn_x[::-1]
            Bi[k, :] = eqn_y[::-1]
            
            p_0 = p_1
            
        return (Ai, Bi) 

    def best_response(self, i, state=None, trajectories=None):
        j = (i + 1) % 2
        ego_traj = trajectories[i]
        A_eqn = jnp.array(trajectories[i][0])
        B_eqn = jnp.array(trajectories[i][1])
        # self.p_i = state[i]  # position of car i
        
        """=================Equality connstraints only involving the ego vehicle=================
            : h_i(θ_i) == 0 , ∀ i∈N
            (p_i)_k - (p_i)_k-1 - v_i*dt = 0
        """
        # eq_continuity = self.continuity_constraints(A_eqn, B_eqn, state[i])
        eq_continuity = mdmm_jax.eq(lambda A_eqn, B_eqn: self.continuity_constraints(A_eqn, B_eqn, state[i]))
        """=================Inequality constraints involving the ego vehicles===================
            : g_i(θ_i) <= 0 , ∀ i∈N
        """
        ineq_vel = mdmm_jax.ineq(lambda A_eqn, B_eqn: self.vel_constraints(A_eqn, B_eqn, self.v_max), weight=10)
        ineq_acc = mdmm_jax.ineq(lambda A_eqn, B_eqn: self.acc_constraints(A_eqn, B_eqn, self.a_max))
        ineq_track = mdmm_jax.ineq(lambda A_eqn, B_eqn: self.track_constraints(A_eqn, B_eqn, trajectories[i]))
        
        """=================Inequality constraints involving both the vehicles===================
            : γ(θ_i, θ_j) <= 0 , ∀ i,j∈N
        """
        ineq_collision = mdmm_jax.ineq(lambda A_eqn, B_eqn: self.collision_constraints(A_eqn, B_eqn, trajectories))
        
        # ineq_vel_test = mdmm_jax.ineq(lambda A_eqn, B_eqn: self.v_max - jnp.linalg.norm(jnp.array([A_eqn[-1, 1] + 2 * A_eqn[-1, 2] * (self.n_steps - 1), B_eqn[-1, 1] + 2 * B_eqn[-1, 2] * (self.n_steps - 1)]), ord=2))
        
        constraints = mdmm_jax.combine(
                                       eq_continuity,
                                       ineq_vel, 
                                       ineq_acc, 
                                       ineq_track, 
                                    #    ineq_collision
                                       )
        
        mdmm_params = constraints.init(A_eqn, B_eqn)
        params = (A_eqn, B_eqn), mdmm_params
        opt_state = self.optimizer.init(params)
        
        
        # Assuming self.track.plot_track and self.get_trajectory are defined methods
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        self.track.plot_track(ax, draw_boundaries=True)
        trajectory_line, = ax.plot([], [], "r")
        start_marker, = ax.plot([], [], "bx")  # Blue cross marker for the start point
        # init_state_marker, = ax.plot(state[i][0], state[i][1], "rx")  # Red cross marker for the initial state
        
        for itr in range(1000):
            params, opt_state, info = self.update(params, opt_state, constraints, ego_traj)
            # stop if the change in the objective is less than 1e-3
            if info[0] < 1e-3:
                break
            # print(opt_state)
            # print(params)
            
            traj_A = params[0][0]
            traj_B = params[0][1]
            # print(traj_A)
            path_i = []
            velocities = []
            for t in range(traj_A.shape[0]):
                path_i.append([traj_A[t, 0] + traj_A[t, 1]*t + traj_A[t, 2]*t**2, traj_B[t, 0] + traj_B[t, 1]*t + traj_B[t, 2]*t**2])   
                velocities.append(jnp.linalg.norm(jnp.array([traj_A[t, 1] + 2 * traj_A[t, 2] * t, traj_B[t, 1] + 2 * traj_B[t, 2] * t]), ord=2))
            print(list(velocities))
            # Convert path_i to a numpy array for easier indexing
            path_i = np.array(path_i)
            cross = ([path_i[0, 0]], [path_i[0, 1]])
            # print(cross)
            # Update the plot
            trajectory_line.set_data([path_i[:, 0]], [path_i[:, 1]])
            start_marker.set_data(*cross)  # Set the start marker at the first point
            plt.pause(0.001)  # Pause to update the plot
        
        plt.show()
        
    @partial(jit, static_argnums=(0,))
    def objective(self, A_eqn, B_eqn, ego_traj):
        nT = self.n_steps - 1
        pos_T = jnp.array([A_eqn[-1][0] + A_eqn[-1][1]*nT + A_eqn[-1][2]*nT**2, B_eqn[-1][0] + B_eqn[-1][1]*nT + B_eqn[-1][2]*nT**2])
        cur_traj_pos_T = jnp.array([ego_traj[0][-1][0] + ego_traj[0][-1][1]*nT + ego_traj[0][-1][2]*nT**2, ego_traj[1][-1][0] + ego_traj[1][-1][1]*nT + ego_traj[1][-1][2]*nT**2])
        _, _, tangent_T, _ = self.track.nearest_trackpoint(cur_traj_pos_T)
        return -tangent_T@pos_T
    
    @partial(jit, static_argnums=(0,))
    def continuity_constraints(self, A_eqn, B_eqn, state):
        """
        1. CONTINUITY CONSTRAINTS
        - a_t + b_t(t+1) + c_t(t+1)^2 == a_t+1 + b_t+1(t+1) + c_t+1(t+1)^2
        """
        i = jnp.arange(A_eqn.shape[0] - 1)
        # Compute individual constraints
        constraint1 = A_eqn[i, 0] + A_eqn[i, 1] * (i + 1) + A_eqn[i, 2] * (i + 1) ** 2 - A_eqn[i + 1, 0] - A_eqn[i + 1, 1] * (i + 1) - A_eqn[i + 1, 2] * (i + 1) ** 2
        constraint2 = B_eqn[i, 0] + B_eqn[i, 1] * (i + 1) + B_eqn[i, 2] * (i + 1) ** 2 - B_eqn[i + 1, 0] - B_eqn[i + 1, 1] * (i + 1) - B_eqn[i + 1, 2] * (i + 1) ** 2
        constraint3 = A_eqn[0, 0] - state[0] + B_eqn[0, 0] - state[1]
        constraint4 = A_eqn[i, 1] + 2 * A_eqn[i, 2] * (i + 1) - A_eqn[i + 1, 1] - 2 * A_eqn[i + 1, 2] * (i + 1)
        constraint5 = B_eqn[i, 1] + 2 * B_eqn[i, 2] * (i + 1) - B_eqn[i + 1, 1] - 2 * B_eqn[i + 1, 2] * (i + 1)
        
        # Stack constraints vertically
        stacked_constraints = jnp.hstack([
            constraint1,
            constraint2,
            constraint3,
            constraint4,
            constraint5
        ])
        # Create a single mdmm_jax.eq constraint
        # cont_constraints = mdmm_jax.eq(lambda A_eqn, B_eqn: stacked_constraints, 0)
        
        # return cont_constraints
        return stacked_constraints
    
    @partial(jit, static_argnums=(0,))
    def vel_constraints(self, A_eqn, B_eqn, v_max):
        ks = jnp.arange(A_eqn.shape[0])
        return -jnp.linalg.norm(jnp.array([A_eqn[:, 1] + 2 * A_eqn[:, 2] * ks, B_eqn[:, 1] + 2 * B_eqn[:, 2] * ks]), axis=0) + jnp.repeat(v_max, A_eqn.shape[0])

    @partial(jit, static_argnums=(0,))
    def acc_constraints(self, A_eqn, B_eqn, a_max):
        return -A_eqn[:,2]**2 + B_eqn[:,2]**2 + a_max
    
    @partial(jit, static_argnums=(0,))
    def track_constraints(self, A_eqn, B_eqn, ego_traj):
        A_ego, B_ego = ego_traj
        ks = jnp.arange(A_eqn.shape[0])
        ksq = ks**2
        p_prev = jnp.array([A_ego[:,0] + A_ego[:,1]*ks + A_ego[:,2]*ksq, B_ego[:,0] + B_ego[:,1]*ks + B_ego[:,2]*ksq]).T
        p_new = jnp.array([A_eqn[:,0] + A_eqn[:,1]*ks + A_eqn[:,2]*ksq, B_eqn[:,0] + B_eqn[:,1]*ks + B_eqn[:,2]*ksq]).T
        # for every p_prev, run and gather returns from self.track.nearest_trackpoint(p_prev)
        _, cs, ts, ns = jax.vmap(self.track.nearest_trackpoint)(p_prev)
        track_const = -jnp.einsum('ij,ij->i', ns, p_new) + jnp.einsum('ij,ij->i', ns, cs) + self.t_width - self.config.collision_radius
        track_const += jnp.einsum('ij,ij->i', ns, p_new) - jnp.einsum('ij,ij->i', ns, cs) + self.t_width - self.config.collision_radius
        return track_const
    
    @partial(jit, static_argnums=(0,))
    def collision_constraints(self, A_eqn, B_eqn, trajs):
        A_ego, B_ego = trajs[self.i_ego]
        A_opp, B_opp = trajs[(self.i_ego + 1) % 2]
        ks = jnp.arange(A_eqn.shape[0])
        ksq = ks**2
        p_prev = jnp.array([A_eqn[:,0] + A_eqn[:,1]*ks + A_eqn[:,2]*ksq, B_eqn[:,0] + B_eqn[:,1]*ks + B_eqn[:,2]*ksq]).T
        p_ego = jnp.array([A_ego[:,0] + A_ego[:,1]*ks + A_ego[:,2]*ksq, B_ego[:,0] + B_ego[:,1]*ks + B_ego[:,2]*ksq]).T
        p_opp = jnp.array([A_opp[:,0] + A_opp[:,1]*ks + A_opp[:,2]*ksq, B_opp[:,0] + B_opp[:,1]*ks + B_opp[:,2]*ksq]).T
        beta = p_opp - p_ego
        # print(beta.shape)
        beta /= jnp.linalg.norm(beta, axis=0) # TODO: see if we can conditionally normalize
        # return jnp.sum(beta@p_opp - beta@p_prev - self.d_coll)
        print("COLL: ", (jnp.einsum('ij,ij->i', beta, p_opp) - jnp.einsum('ij,ij->i', beta, p_prev) - self.d_coll).shape)
        return jnp.einsum('ij,ij->i', beta, p_opp) - jnp.einsum('ij,ij->i', beta, p_prev) - self.d_coll
    
    # @partial(jit, static_argnums=(0,))
    def system(self, params, constraint, ego_traj):
        main_params, mdmm_params = params
        loss = self.objective(*main_params, ego_traj)
        mdmm_loss, inf = constraint.loss(mdmm_params, *main_params)
        return loss + mdmm_loss, (loss, inf)
    
    # @partial(jit, static_argnums=(0,))
    def update(self, params, opt_state, constraint, ego_traj):
        grad, info = jax.grad(self.system, has_aux=True)(params, constraint, ego_traj)
        updates, opt_state = self.optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, info
        
    def iterative_br(self, i_ego, state, n_game_iterations=2, n_sqp_iterations=3):
        trajectories = [self.init_traj(state[i, :]) for i in [0, 1]]

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
        return jnp.array(trajectories)

import matplotlib.animation as animation
import argparse

if __name__ == "__main___":
    parser = argparse.ArgumentParser(description="Run the planner simulation.")
    parser.add_argument("--animate", action="store_true", help="Run the simulation with animation.")
    args = parser.parse_args()

    planner = SE_IBR(config)
    way_idx = 3
    ego_state = planner.track.waypoints[way_idx]
    opp_state = (
        planner.track.waypoints[way_idx]
        + planner.track.track_normals[way_idx] * 0.3
        - planner.track.track_tangent[way_idx] * 0.3
    )
    state = jnp.array([ego_state, opp_state])

    if args.animate:
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        planner.track.plot_track(ax, draw_boundaries=True)
        ego_point, = ax.plot([], [], "rx")
        opp_point, = ax.plot([], [], "bx")
        ego_path, = ax.plot([], [], "r")
        opp_path, = ax.plot([], [], "b")

        def init():
            ego_point.set_data([], [])[[1,2,3],[0,1,0]], [[0,0,0], [0,0,0]]
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
            pathi = jnp.array(pathi)
            pathj = jnp.array(pathj)
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
    else:
        for i in range(100):
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
            pathi = jnp.array(pathi)
            pathj = jnp.array(pathj)
            state[0] = pathi[2]
            state[1] = pathj[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the planner simulation.")
    parser.add_argument("--animate", action="store_true", help="Run the simulation with animation.")
    args = parser.parse_args()

    planner = SE_IBR(config)
    way_idx = 3
    ego_state = planner.track.waypoints[way_idx]
    opp_state = (
        planner.track.waypoints[way_idx]
        + planner.track.track_normals[way_idx] * 0.3
        - planner.track.track_tangent[way_idx] * 0.3
    )
    state = jnp.array([ego_state, opp_state])
    print(state)
    trajectories = [planner.init_traj(ego_state), planner.init_traj(opp_state)]
    planner.best_response(0, state, trajectories)
    # time = timeit.timeit(lambda: planner.best_response(0, state, trajectories), number=100)
