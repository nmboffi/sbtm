"""
Nicholas M. Boffi
7/29/22

Updates to parameters and particles for score-based transport modeling.
"""

from jax import jit, vmap
import jax
from functools import partial
import jax.numpy as np
import numpy as onp
import haiku as hk
from typing import Callable, Tuple, Union
from tqdm.auto import tqdm as tqdm


@partial(jit, static_argnums=(5, 6))
def update_particles(
    particle_pos: np.ndarray,
    t: float,
    params: hk.Params,
    D: Union[np.ndarray, float],
    dt: float,
    forcing: Callable[[np.ndarray, float], np.ndarray],
    apply_score: Callable[[hk.Params, np.ndarray], np.ndarray],
    mask: np.ndarray = None
) -> np.ndarray:
    """Take a forward Euler step and update the particles."""
    if mask is not None:
        score_term = -D * mask * apply_score(params, particle_pos)
    else:
        score_term = -D * apply_score(params, particle_pos)

    return particle_pos + dt*(forcing(particle_pos, t) + score_term)


@partial(jit, static_argnums=(5, 6))
def update_particles_EM(
    particle_pos: np.ndarray,
    t: float,
    D_sqrt: Union[np.ndarray, float],
    dt: float,
    key: np.ndarray,
    forcing: Callable[[np.ndarray, float], np.ndarray],
    noisy: bool = True,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Take a step forward via Euler-Maruyama to update the particles."""

    if noisy:
        noise = np.sqrt(2*dt) * jax.random.normal(key, shape=particle_pos.shape)
        if mask is not None:
            brownian = -D_sqrt * mask * noise
        else:
            brownian = -D_sqrt * noise

        return particle_pos + dt*forcing(particle_pos, t) + brownian
    else:
        return particle_pos + dt*forcing(particle_pos, t)


def rollout_EM_trajs(
    x0s: np.ndarray,
    nsteps: int,
    t0: float,
    dt: float,
    key: np.ndarray,
    forcing: Callable[[np.ndarray, float], np.ndarray],
    D_sqrt: Union[np.ndarray, float],
    noisy: bool = True
) -> np.ndarray:
    """Given a set of initial conditions, create a stochastic trajectory 
    via Euler-Maruyama. Useful for constructing a baseline against which to compare
    the moments.

    Args:
    ------
    x0s: Initial condition. Dimension = n x d where n is the number of samples 
         and d is the dimension of the system.
    nsteps: Number of steps of Euler-Maruyama to take.
    t0: initial time.
    dt: Timestep.
    key: jax PRNG key.
    forcing: Forcing to apply to the particles.
    D_sqrt: Square root of the diffusion matrix.
    """
    n, d = x0s.shape
    trajs = onp.zeros((nsteps+1, n, d))
    trajs[0] = x0s
    step_sample = \
            lambda sample, t, key: update_particles_EM(sample, t, D_sqrt, 
                                                       dt, key, forcing, noisy)
    step_samples = vmap(step_sample, in_axes=(0, None, 0), out_axes=0)

    for curr_step in tqdm(range(nsteps)):
        t = t0 + curr_step*dt
        keys = jax.random.split(key, num=n)
        trajs[curr_step+1] = step_samples(trajs[curr_step], t, keys)
        key = keys[-1]

    return trajs, key
