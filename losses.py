"""
Nicholas M. Boffi
7/29/22

Loss functions for score-based transport modeling.
"""


import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from typing import Tuple, Callable, Union, Optional
from jax import jit, vmap, value_and_grad
from jaxopt.linear_solve import solve_cg
from functools import partial
import haiku as hk
import optax


def sqnorm(
    x: np.ndarray,
    axis: int
) -> np.ndarray:
    """Compute the square norm of an array along an axis."""
    return np.sum(x**2, axis=axis)



@partial(jit, static_argnums=(4, 5))
def denoising_loss(
    params: hk.Params,
    samples: np.ndarray,
    noises: np.ndarray,
    noise_fac: float,
    batch_score: Callable[[hk.Params, np.ndarray], np.ndarray],
    split_num: int=1
) -> float:
    """Compute the denoising loss function. Apply antithetic 
    sampling for variance reduction.

    Note: for hypo-elliptic problems, always assume that the coordinates with
    noise are last!
    
    Args:
        params:      Haiku parameter mapping for the neural network.
        samples:     Samples of particles. dimension = n X d with d the
                     system dimension and n the number of samples from \rho_t.
        noises:      Noise samples. dimension = N x n x d with N the number of noise samples
                     per sample from \rho_t.
        noise_fac:   Standard deviation of the noise used in the stochastic approximation.
        apply_score: Haiku transform for computing the score.
        split_point: Where to split the data, if the score is defined as the derivative
                     over a subset of the input dimensions.
    """
    loss = 0
    for sign in [-1, 1]:
        perturbed_samples = samples[None, :, :] + noise_fac*sign*noises
        if split_num != 1:
            split_samples = np.split(perturbed_samples, split_num, axis=-1)
            in_axes = (0,)*split_num
            scores = vmap(
                    lambda *args: batch_score(params, *args),
                    in_axes=in_axes, out_axes=0
                )(*split_samples)
        else:
            scores = vmap(lambda xs: batch_score(params, xs))(perturbed_samples)

        d_score = scores.shape[-1]
        loss += np.sum(noise_fac*scores**2 + 2*scores*sign*noises[:, :, d_score:])
    
    return loss / (noises.shape[0]*noises.shape[1]*d_score)


@partial(jit, static_argnums=5)
def sm_loss(
    params: hk.Params,
    samples: np.ndarray,
    div_noises: np.ndarray,
    reg_noises: np.ndarray,
    lam: float,
    apply_score: Callable[[hk.Params, np.ndarray], np.ndarray],
) -> float:
    """Compute the standard score-matching loss function.
    Apply regularization on the Frobenius norm of the Jacobian.
    Approximate the divergence term via Skilling-Hutchinson.

    Args:
        params: Haiku parameters for the network.
        sample: Samples on which to perform the minimization.
        div_noises: Noises to use in the computation of the divergence.
        reg_noises: Noises to use in the computation of the regularization term.
        lam: Regularization parameter.
        apply_score: Function that computes the score on a sample.
        batch_score: Function that computes the score on a batch.
    """
    calc_score = lambda x: apply_score(params, x)
    apply_score_jac = lambda x, eta: jax.jvp(calc_score, primals=(x,), tangents=(eta,))
    map_jac = vmap(apply_score_jac, in_axes=(0, 0), out_axes=0)

    scores, nabla_s_divs = map_jac(samples, div_noises)
    _, nabla_s_regs = map_jac(samples, reg_noises)

    divs = np.sum(div_noises * nabla_s_divs)
    regs = np.sum(np.square(reg_noises * nabla_s_regs))
    norms = np.sum(np.square(scores))

    return (norms + 2*divs + lam * regs) / samples.size


def grad_log_rho0(
    sample: np.ndarray,
    sig0: float,
    mu0: np.ndarray
) -> np.ndarray:
    """Compute the initial potential. Assumed to be an isotropic Gaussian."""
    return -(sample - mu0) / sig0**2


@partial(jit, static_argnums=(4, 5, 7))
def init_loss(
    params: np.ndarray,
    samples: np.ndarray,
    sig0: float,
    mu0: np.ndarray,
    apply_score: Callable[[hk.Params, np.ndarray, Optional[float]], np.ndarray],
    time_dependent: bool = False,
    frame_end: float = 0,
    nt: int = 0
) -> np.ndarray:
    """Compute the initial loss, assuming access to \nabla \log \rho0."""

    grad_log_rho_evals = vmap(
        lambda sample: grad_log_rho0(sample, sig0, mu0)
    )(samples)
    
    if time_dependent:
        ts = np.linspace(0, frame_end, nt)
        dt = ts[1] - ts[0]

        space_batch_score = vmap(
            lambda sample, t: apply_score(params, sample, t),
            in_axes=(0, None),
            out_axes=0
        )

        batch_score = vmap(
                space_batch_score,
                in_axes=(None, 0),
                out_axes=0
            )

        score_evals = batch_score(samples, ts)

        return np.sum((score_evals - grad_log_rho_evals[None, :, :])**2) \
                    / (nt*np.sum(grad_log_rho_evals**2))
    else:
        score_evals = vmap(
            lambda sample: apply_score(params, sample)
        )(samples)

        return np.sum((score_evals - grad_log_rho_evals)**2) \
                / np.sum(grad_log_rho_evals**2)


@jit
def compute_grad_norm(
    grads: hk.Params
) -> float:
    """ Computes the norm of the gradient, where the gradient is input
    as an hk.Params object (treated as a PyTree)."""
    flat_params = ravel_pytree(grads)[0]
    return np.linalg.norm(flat_params) / np.sqrt(flat_params.size)


@partial(jit, static_argnums=(2, 3))
def update(
    params: hk.Params,
    opt_state: optax.OptState,
    opt: optax.GradientTransformation,
    loss_func: Callable[[hk.Params], float],
    loss_func_args: Tuple = tuple(), 
) -> Tuple[hk.Params, optax.OptState, float, hk.Params]:
    """Update the neural network.

    Args:
        params: Parameters to optimize over.
        opt_state: State of the optimizer.
        opt: Optimizer itself.
        loss_func: Loss function for the parameters.
    """
    loss_value, grads = value_and_grad(loss_func)(params, *loss_func_args)
    updates, opt_state = opt.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, grads
