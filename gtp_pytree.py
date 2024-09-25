import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax.tree_util import register_pytree_node_class
from track_jnp import Splined_Track
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

@struct.dataclass
class SE_IBR:
    config: any
    dt: float
    track: any
    n_steps: int
    traj: any
    i_ego: int = 0
    nc_weight: float = 1.0
    nc_relax_weight: float = 128.0
    track_relax_weight: float = 128.0
    max_curvature: float = 0.0
    v_max: float = 0.0
    a_max: float = 0.0
    d_coll: float = 0.0
    d_safe: float = 0.0
    t_width: float = 0.0
    prngkey: jnp.ndarray = jax.random.PRNGKey(0)
    optimizer: any = struct.field(pytree_node=False)  # Excluded from pytree

    @classmethod
    def create(cls, config):
        track = Splined_Track(config.track_waypoints, config.track_width)
        traj = cls.init_traj(cls, track.waypoints[0])
        optimizer = optax.chain(optax.sgd(1e-3), mdmm_jax.optax_prepare_update())
        return cls(
            config=config,
            dt=config.dt,
            track=track,
            n_steps=config.n_steps,
            traj=traj,
            max_curvature=config.max_curvature,
            v_max=config.v_max,
            a_max=config.a_max,
            d_coll=2 * config.collision_radius,
            d_safe=2 * config.collision_radius,
            t_width=track.track_width,
            prngkey=jax.random.PRNGKey(0),
            optimizer=optimizer
        )
    
    def init_traj(self, p_0):
        Ai = jnp.zeros((self.n_steps, 3))
        Bi = jnp.zeros((self.n_steps, 3))
    
        for k in range(self.n_steps):
            _, _, t0, _ = self.track.nearest_trackpoint(p_0)
            p_1 = p_0 + self.config.v_max * t0
            p_2 = p_1 + self.config.v_max * t0 * 2
            
            ks = jnp.array([k, k+1.0, k+2.0])
            psx = jnp.array([p_0[0], p_1[0], p_2[0]])
            psy = jnp.array([p_0[1], p_1[1], p_2[1]])
            eqn_x = jnp.polyfit(ks, psx, 2)
            eqn_y = jnp.polyfit(ks, psy, 2)
            
            Ai = Ai.at[k, :].set(eqn_x[::-1])
            Bi = Bi.at[k, :].set(eqn_y[::-1])
            
            p_0 = p_1
            
        return (Ai, Bi)
    
    @partial(jax.jit, static_argnums=(0,))
    def objective(self, A_eqn, B_eqn, ego_traj):
        nT = self.n_steps - 1
        pos_T = jnp.array([A_eqn[-1][0] + A_eqn[-1][1]*nT + A_eqn[-1][2]*nT**2, 
                           B_eqn[-1][0] + B_eqn[-1][1]*nT + B_eqn[-1][2]*nT**2])
        cur_traj_pos_T = jnp.array([ego_traj[0][-1][0] + ego_traj[0][-1][1]*nT + ego_traj[0][-1][2]*nT**2, 
                                    ego_traj[1][-1][0] + ego_traj[1][-1][1]*nT + ego_traj[1][-1][2]*nT**2])
        _, _, tangent_T, _ = self.track.nearest_trackpoint(cur_traj_pos_T)
        return -tangent_T @ pos_T

    @partial(jax.jit, static_argnums=(0,))
    def continuity_constraints(self, A_eqn, B_eqn, state):
        i = jnp.arange(A_eqn.shape[0] - 1)
        constraint1 = A_eqn[i, 0] + A_eqn[i, 1] * (i + 1) + A_eqn[i, 2] * (i + 1) ** 2 - A_eqn[i + 1, 0] - A_eqn[i + 1, 1] * (i + 1) - A_eqn[i + 1, 2] * (i + 1) ** 2
        constraint2 = B_eqn[i, 0] + B_eqn[i, 1] * (i + 1) + B_eqn[i, 2] * (i + 1) ** 2 - B_eqn[i + 1, 0] - B_eqn[i + 1, 1] * (i + 1) - B_eqn[i + 1, 2] * (i + 1) ** 2
        constraint3 = A_eqn[0, 0] - state[0] + B_eqn[0, 0] - state[1]
        constraint4 = A_eqn[i, 1] + 2 * A_eqn[i, 2] * (i + 1) - A_eqn[i + 1, 1] - 2 * A_eqn[i + 1, 2] * (i + 1)
        constraint5 = B_eqn[i, 1] + 2 * B_eqn[i, 2] * (i + 1) - B_eqn[i + 1, 1] - 2 * B_eqn[i + 1, 2] * (i + 1)
        
        stacked_constraints = jnp.hstack([
            constraint1,
            constraint2,
            constraint3,
            constraint4,
            constraint5
        ])
        return stacked_constraints

    @partial(jax.jit, static_argnums=(0,))
    def vel_constraints(self, A_eqn, B_eqn, v_max):
        ks = jnp.arange(A_eqn.shape[0])
        return -jnp.linalg.norm(jnp.array([A_eqn[:, 1] + 2 * A_eqn[:, 2] * ks, B_eqn[:, 1] + 2 * B_eqn[:, 2] * ks]), axis=0) + jnp.repeat(v_max, A_eqn.shape[0])

    # Define other constraints and methods similarly...
    
    @partial(jax.jit, static_argnums=(0,))
    def update(self, params, opt_state, constraint, ego_traj):
        grad, info = jax.grad(self.system, has_aux=True)(params, constraint, ego_traj)
        updates, opt_state = self.optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, info

# Now register the class as a pytree
register_pytree_node_class(SE_IBR)
