"""
Nicholas M. Boffi
7/29/22

Drift terms for score-based transport modeling.
"""


from jax import vmap
from jax.lax import stop_gradient
import jax.numpy as np
from typing import Callable, Tuple


compute_particle_diffs = vmap(
        vmap(lambda x, y: x - y, in_axes=(0, None), out_axes=0),
        in_axes=(None, 0), out_axes=1
    )


def active_swimmer(
    xv: np.ndarray,
    t: float,
    gamma: float
) -> np.ndarray:
    """Active swimmer example."""
    del t
    x, v = xv
    return np.array([-x**3 + v, -gamma*v])


def harmonic_trap(
    particle_pos: np.ndarray,
    t: float,
    compute_mut: Callable[[float], np.ndarray]
) -> np.ndarray:
    """Forcing for particles in a harmonic trap with harmonic repulsion."""
    mut = compute_mut(t)
    particle_forces = -0.5*particle_pos + mut[None, :] \
            - 0.5*np.mean(particle_pos, axis=0)

    return particle_forces


def gaussian_interaction(
    xs: np.ndarray,
    A: float,
    r: float,
) -> np.ndarray:
    particle_diffs = compute_particle_diffs(xs, xs)
    gauss_facs = np.exp(-np.sum(particle_diffs**2, axis=2) / (2*r**2))
    interaction = A/(r**2)*np.mean(particle_diffs * gauss_facs[:, :, None], axis=1)
    return interaction


def anharmonic_gaussian(
    x: np.ndarray,
    t: float,
    A: float,
    r: float,
    B: float,
    N: int,
    d: int,
    compute_mut: Callable[[float], np.ndarray],
    print_info=False
) -> np.ndarray:
    """Gaussian short-range force with anharmonic attraction to the origin."""
    # trap force
    particle_pos = x.reshape((N, d))
    diff = particle_pos - compute_mut(t)[None, :]
    diff_norms = np.linalg.norm(diff, axis=1)**2

    # repulsive interaction
    interaction = gaussian_interaction(particle_pos, A, r)
    particle_forces = -B*diff*diff_norms[:, None] + interaction

    if print_info:
        print('interaction:', interaction)
        print('potential:', particle_forces - interaction)
        print()

    return particle_forces.ravel()



def anharmonic(
    x: np.ndarray,
    t: float,
    compute_mut: Callable[[float], np.ndarray]
) -> np.ndarray:
    """Single particle in an anharmonic trap."""
    diff = x - compute_mut(t)
    return -diff * (diff @ diff)


def anharmonic_harmonic(
    x: np.ndarray,
    t: float,
    A: float,
    B: float,
    N: int,
    d: int,
    compute_mut: Callable[[float], np.ndarray],
) -> np.ndarray:
    """Harmonically-interacting particles in an anharmonic trap."""
    particle_pos = x.reshape((N, d))
    diff = particle_pos - compute_mut(t)[None, :]
    diff_norms = np.sum(diff**2, axis=1)
    xbar = np.mean(particle_pos, axis=0)
    return np.ravel(-B*diff*diff_norms[:, None] + A*(particle_pos - xbar[None, :]))
