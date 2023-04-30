"""
Nicholas M. Boffi
7/29/22

Neural networks for score-based transport modeling.
"""


import haiku as hk
import jax.numpy as np
import numpy as onp
from typing import Optional, Tuple, Callable, Union
import jax
from jax import vmap
from dataclasses import dataclass


def construct_mlp_layers(
    n_hidden: int,
    n_neurons: int,
    act: Callable[[np.ndarray], np.ndarray],
    output_dim: int,
    residual_blocks: bool = True
) -> list:
    """Make a list containing the layers of an MLP.

    Args:
        n_hidden: Number of hidden layers in the MLP.
        n_neurons: Number of neurons per hidden layer.
        act: Activation function.
        output_dim: Dimension of the output.
        residual_blocks: Whether or not to use residual blocks.
    """
    layers = []

    resid_act = lambda x: x + act(x)
    for layer in range(n_hidden):
        ## construct layer
        if layer == 0 or not residual_blocks:
            layers = layers + [
                    hk.Linear(n_neurons),
                    act
                ]
        else:
            layers = layers + [
                    hk.Linear(n_neurons),
                    resid_act
                ]


    ## construct output layer
    layers = layers + [hk.Linear(output_dim)]
    return layers


def construct_interacting_particle_system_network(
    n_hidden: int,
    n_neurons: int,
    N: int,
    d: int,
    act: Callable[[np.ndarray], np.ndarray],
    residual_blocks: bool = False,
) -> Tuple[Callable, Callable]:
    """Construct a neural network useful for
    modeling interacting particle systems.

    Args:
        n_hidden: Number of hidden layers for each network.
        n_neurons: Number of neurons per hidden layer.
        N: Number of particles.
        d: Ambient dimension.
        act: Activation function.
        residual_blocks: Whether or not to use residual blocks.

    Returns
        Haiku transforms that compute the score and the potential.
    """
    def one_particle_term(
            xi: np.ndarray,
            t: Optional[float] = None
        ) -> float: 
        if t is not None:
            xi = np.append(xi, t)

        net = hk.Sequential(
                construct_mlp_layers(
                    n_hidden, 
                    n_neurons, 
                    act, 
                    output_dim=1, 
                    residual_blocks=residual_blocks
                    )
                )

        return net(xi)

    def two_particle_term(
            xi: np.ndarray, 
            xj: np.ndarray,
            t: Optional[float] = None
        ) -> float:
        net = hk.Sequential(
                construct_mlp_layers(
                    n_hidden, 
                    n_neurons, 
                    act, 
                    output_dim=1, 
                    residual_blocks=residual_blocks
                    )
                )

        if t is not None:
            xit, xjt = np.append(xi, t), np.append(xj, t)
            forward_input = np.concatenate((xi, xjt))
            reverse_input = np.concatenate((xj, xit))
        else:
            forward_input = np.concatenate((xi, xj))
            reverse_input = np.concatenate((xj, xi))

        return 0.5*(net(forward_input) + net(reverse_input))

    def potential_energy(
            xs: np.ndarray, 
            t: Optional[float] = None
        ) -> float:
        xs_mat = xs.reshape((N, d))

        one_particle_energy = np.sum(
            vmap(one_particle_term, in_axes=(0, None), out_axes=0)(xs_mat, t)
        )

        if N > 1:
            two_particle_all = vmap(
                    vmap(
                        two_particle_term, 
                        in_axes=(0, None, None), 
                        out_axes=0
                    ),
                in_axes=(None, 0, None), 
                out_axes=0
            )(xs_mat, xs_mat, t)

            two_particle_energy = 0.5*(np.sum(two_particle_all) \
                    - np.trace(two_particle_all)) / N
        else:
            two_particle_energy = 0

        return np.squeeze(
                one_particle_energy + two_particle_energy
            )

    score = jax.grad(potential_energy)
    potential_network = hk.without_apply_rng(hk.transform(potential_energy))
    score_network = hk.without_apply_rng(hk.transform(score))

    return score_network, potential_network


def construct_score_network(
    d: int,
    n_hidden: int,
    n_neurons: int,
    act: Callable[[np.ndarray], np.ndarray],
    residual_blocks: bool = True,
    is_gradient: bool = True
) -> Tuple[Callable, Callable]:
    """Construct a score network for a simpler system
    that does not consist of interacting particles.

    Args:
        d: System dimension.
        n_hidden: Number of hidden layers in the network.
        n_neurons: Number of neurons per layer.
        act: Activation function.
        residual_blocks: Whether or not to use residual blocks.
        is_gradient: Whether or not to compute the score as the gradient of a potential.
    """
    output_dim = 1 if is_gradient else d
    net = lambda x: hk.Sequential(
        construct_mlp_layers(n_hidden, n_neurons, act, output_dim, residual_blocks)
    )(x)


    if is_gradient:
        # need to squeeze so the output of the potential is a scalar, rather than
        # a 1-dimensional array.
        potential = lambda x: np.squeeze(net(x))
        score = jax.grad(potential)
        return hk.without_apply_rng(hk.transform(score)), \
                hk.without_apply_rng(hk.transform(potential))
    else:
        return hk.without_apply_rng(hk.transform(net)), None
