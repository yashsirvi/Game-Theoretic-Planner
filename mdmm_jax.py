import mdmm_jax
import jax
import jax.numpy as jnp
import optax


def fx(x, y):
    return x**2 + y**2

def loss_fn(params):
    return fx(*params)

constraint_fn = lambda v: v[0] + v[1] - 1
constraint=mdmm_jax.eq(constraint_fn, 0)
v = jnp.array([0., 0.])
mdmm_params = constraint.init(v)
params = v, mdmm_params
opt = optax.chain(optax.adam(1e-1), mdmm_jax.optax_prepare_update())
opt_state= opt.init(params)

def system(params):
    main_params, mdmm_params = params
    loss = loss_fn(main_params)
    mdmm_loss, inf = constraint.loss(mdmm_params, main_params)
     # f(x) + \lambda g(x) + c/2 g(x)^2
     # f(x) = loss
     # \lambda g(x) + c/2 g(x)^2 = mdmm_loss
    return loss + mdmm_loss, (loss, inf)

@jax.jit
def update(params, opt_state):
    grad, info = jax.grad(system, has_aux=True)(params)
    updates, opt_state = opt.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, info

for i in range(100):
    params, opt_state, info = update(params, opt_state)
