"""
Nicholas M. Boffi
8/4/22

Training loops for score-based transport modeling.
"""


from jax import jit, vmap
from jax.tree_util import tree_map
import jax
from functools import partial
import jax.numpy as np
import numpy as onp
import haiku as hk
from typing import Callable, Tuple, Union
from . import losses
from . import updates
import optax
import dill as pickle
import time
from tqdm.auto import tqdm
from jaxlib.xla_extension import Device

Time = float


def fit_initial_condition(
    n_max_opt_steps: int,
    ltol: float,
    params: hk.Params,
    sig0: float,
    mu0: np.ndarray,
    score_network: Callable[[hk.Params, np.ndarray], np.ndarray],
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    samples: np.ndarray,
    time_dependent: bool = False,
    frame_end: float = 0,
    nt: int = 0
) -> hk.Params:
    """Fit the score for the initial condition.

    Args:
        n_opt_steps: Number of optimization steps before the norm of the gradient 
                     is checked.
        gtol: Tolerance on the norm of the gradient.
        ltol: Tolerance on the relative error.
        params: Parameters to optimize over.
        sig0: Standard deviation of the target initial condition.
        mu0: Mean of the target initial condition.
        score_network: Function mapping parameters and a sample to the network output.
        opt: Optimizer.
        opt_state: State of the optimizer.
        samples: Samples to optimizer over.
    """
    apply_score = score_network.apply
    loss_func = lambda params: \
            losses.init_loss(params, samples, sig0, mu0, apply_score, 
                             time_dependent, frame_end, nt)

    loss_val = np.inf
    with tqdm(range(n_max_opt_steps)) as pbar:
        pbar.set_description("Initial optimization")
        for curr_step in pbar:
            params, opt_state, loss_val, grads \
                    = losses.update(params, opt_state, opt, loss_func)
            pbar.set_postfix(loss=loss_val)
            if loss_val < ltol:
                break


    return params
