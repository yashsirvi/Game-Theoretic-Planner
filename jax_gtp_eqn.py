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
        self.c_cont_constraint = jnp.array([2*i+1 for i in range(self.n_steps-1)]).reshape(-1, 1)
        
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

        A_eqn = jax.random.normal(self.prngkey, (self.n_steps, 3))
        B_eqn = jax.random.normal(self.prngkey, (self.n_steps, 3))
        # self.p_i = state[i]  # position of car i
        
        """=================Equality connstraints only involving the ego vehicle=================
            : h_i(θ_i) == 0 , ∀ i∈N
            (p_i)_k - (p_i)_k-1 - v_i*dt = 0
        """
        
        eq_contiuity = mdmm_jax.eq(lambda: self.continuity_constraints(A_eqn, B_eqn, self.n_steps, self.c_cont_constraint, state), 0)

        """=================Inequality constraints involving the ego vehicles=================
            : g_i(θ_i) <= 0 , ∀ i∈N
        """
        ineq_vel = mdmm_jax.ineq(lambda: self.vel_constraints(A_eqn, B_eqn, self.v_max), 0)
        ineq_acc = mdmm_jax.ineq(lambda: self.acc_constraints(A_eqn, B_eqn, self.a_max), 0)
        ineq_track = mdmm_jax.ineq(lambda: self.track_constraints(A_eqn, B_eqn, trajectories[j]), 0)
        
        """=================Inequality constraints involving both the vehicles=================
            : γ(θ_i, θ_j) <= 0 , ∀ i,j∈N
        """
        ineq_collision = mdmm_jax.ineq(lambda: self.collision_constraints(A_eqn, B_eqn, trajectories), 0)
        
    
    @staticmethod
    @jit
    def continuity_constraints(A_eqn, B_eqn, nsteps, c_cont_constraint, state):
        """
        1. CONTINUITY CONSTRAINTS
        - a_t + b_t(t+1) + c_t(t+1)^2 == a_t+1 + b_t+1(t+1) + c_t+1(t+1)^2
        - for combined constraint, we can add up the constraints
        - i.e min f(x) s.t. h_i(x) = 0, ∀ i∈N => min f(x) + Σλ_ih_i(x)
        - Sum comes to be
        - for a_i terms -> a_(n-1) - a 
        - for b_i terms -> (n-1)̇⋅b_(n-1) - Σ_i=0^n-2 b_i
        - for c_i terms -> (n-1)^2⋅c_(n-1) - Σ_i=0^n-2 (2i+1)c_i
        - can precompute 2i+1 for i=0 to n-2
        """
        total_sum = A_eqn[-1][0] - A_eqn[0][0]
        total_sum += (nsteps - 1) * A_eqn[-1][1] - jnp.sum(A_eqn[:-1, 1])
        total_sum += (nsteps - 1)**2 * A_eqn[-1][2] - jnp.sum(c_cont_constraint * A_eqn[:-1, 2])
        total_sum += B_eqn[-1][0] - B_eqn[0][0]
        total_sum += (nsteps - 1) * B_eqn[-1][1] - jnp.sum(B_eqn[:-1, 1])
        total_sum += (nsteps - 1)**2 * B_eqn[-1][2] - jnp.sum(c_cont_constraint * B_eqn[:-1, 2])
        #vel
        total_sum += A_eqn[-1][1] - A_eqn[0][1]
        total_sum += 2*(nsteps - 1) * A_eqn[-1][2] - 2*jnp.sum(A_eqn[:-1, 2])
        total_sum += A_eqn[0][0] - state[i][0] + B_eqn[0][0] - state[i][1] 
        return total_sum
    
    @staticmethod
    @jit
    def vel_constraints(A_eqn, B_eqn, v_max):
            ks = jnp.arange(A_eqn.shape[0])
            return jnp.sum(jnp.linalg.norm([A_eqn[:, 1] + 2 * A_eqn[:, 2] * ks, B_eqn[:, 1] + 2 * B_eqn[:, 2] * ks], axis=0) - v_max)
    @staticmethod
    @jit
    def acc_constraints(A_eqn, B_eqn, a_max):
        return jnp.sum(A_eqn[:,2]**2 + B_eqn[:,2]**2 - a_max)
    
    @partial(jit, static_argnums=(0,))
    def track_constraints(self, A_eqn, B_eqn, ego_traj):
        A_ego, B_ego = ego_traj
        ks = jnp.arange(A_eqn.shape[0])
        ksq = ks**2
        p_prev = jnp.array([A_ego[:,0] + A_ego[:,1]*ks + A_ego[:,2]*ksq, B_ego[:,0] + B_ego[:,1]*ks + B_ego[:,2]*ksq]).T
        p_new = jnp.array([A_eqn[:,0] + A_eqn[:,1]*ks + A_eqn[:,2]*ksq, B_eqn[:,0] + B_eqn[:,1]*ks + B_eqn[:,2]*ksq]).T
        # for every p_prev, run and gather returns from self.track.nearest_trackpoint(p_prev)
        _, cs, ts, ns = jax.vmap(self.track.nearest_trackpoint)(p_prev)
        track_cost = jnp.sum(-jnp.einsum('ij,ij->i', ns, p_new) + jnp.einsum('ij,ij->i', ns, cs) + self.t_width - self.config.collision_radius)
        track_cost += jnp.sum(jnp.einsum('ij,ij->i', ns, p_new) - jnp.einsum('ij,ij->i', ns, cs) + self.t_width - self.config.collision_radius)
        return track_cost
    
    def collision_constraints(self, A_eqn, B_eqn, trajs):
        A_ego, B_ego = trajs[self.i_ego]
        A_opp, B_opp = trajs[(self.i_ego + 1) % 2]
        ks = jnp.arange(A_eqn.shape[0])
        ksq = ks**2
        p_prev = jnp.array([A_eqn[:,0] + A_eqn[:,1]*ks + A_eqn[:,2]*ksq, B_eqn[:,0] + B_eqn[:,1]*ks + B_eqn[:,2]*ksq]).T
        p_ego = jnp.array([A_ego[:,0] + A_ego[:,1]*ks + A_ego[:,2]*ksq, B_ego[:,0] + B_ego[:,1]*ks + B_ego[:,2]*ksq]).T
        p_opp = jnp.array([A_opp[:,0] + A_opp[:,1]*ks + A_opp[:,2]*ksq, B_opp[:,0] + B_opp[:,1]*ks + B_opp[:,2]*ksq]).T
        beta = jnp.array([p_opp[0,:] - p_ego[0,:], p_opp[1,:] - p_ego[1,:]])
        beta /= jnp.linalg.norm(beta, axis=0) # TODO: see if we can conditionally normalize
        return jnp.sum(beta.T@p_opp - beta.T@p_prev - self.d_coll)
        
        
        
                
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

# = np.array([[0, 0], [0, 0]])
# ...
# Assuming SE_IBR and config are defined elsewhere in your code

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
    trajectories = [planner.init_traj(ego_state), planner.init_traj(opp_state)]
    planner.best_response(0, state, trajectories)
    # ego_traj = np.array([[1,2,3], [1,1,0]])
    # ego_traj2 = ego_traj
    # print(ego_traj.shape)
    # planner.track_constraints(ego_traj, ego_traj,ego_traj=jnp.array([ego_traj, ego_traj2]))