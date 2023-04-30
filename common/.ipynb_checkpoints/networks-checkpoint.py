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


@dataclass
class IPSTransformer(hk.Module):
    """A simple SetTransformer for modeling an interacting particle system."""
    n_neurons_encode: int
    n_layers_attn: int
    n_heads: int
    n_neurons_decode: int
    N: int
    d: int
    is_gradient: bool
    residual_blocks: bool
    embed_particles: bool
    act: Callable[[np.ndarray], np.ndarray]


    def __call__(
        self,
        xs: np.ndarray
    ) -> np.ndarray:
        Z = xs.reshape((self.N, self.d))

        # optional embedding of particles into a latent space
        if self.embed_particles:
            encode_mlp = hk.Sequential([
                hk.Linear(self.n_heads*self.n_neurons_encode),
                hk.LayerNorm(
                    axis=-1, 
                    param_axis=-1, 
                    create_scale=True, 
                    create_offset=True
                ),
                self.act,
            ])
            Z = encode_mlp(Z)
            init_fac = 1.0 / np.sqrt(self.n_neurons_encode)
        else:
            init_fac = 1.0 / np.sqrt(self.d)


        ## encoder: attention
        encode_init = hk.initializers.TruncatedNormal(init_fac)
        for kk in range(self.n_layers_attn):
            # attention block
            attn_block = hk.MultiHeadAttention(
                    num_heads=self.n_heads, 
                    key_size=self.n_neurons_encode,
                    w_init=encode_init
                )
            ln = hk.LayerNorm(
                    axis=-1, 
                    param_axis=-1, 
                    create_scale=True, 
                    create_offset=True
                )
            if self.residual_blocks:
                Z = ln(Z + attn_block(Z, Z, Z))
            else:
                Z = ln(attn_block(Z, Z, Z))

            # MLP block
            dense_block = hk.Sequential([
                hk.Linear(self.n_heads*self.n_neurons_encode),
                self.act
            ])
            ln = hk.LayerNorm(
                    axis=-1, 
                    param_axis=-1, 
                    create_scale=True, 
                    create_offset=True
                )
            if self.residual_blocks:
                Z = ln(Z + dense_block(Z))
            else:
                Z = ln(dense_block(Z))


        # output the potential
        if self.is_gradient:
            ## decoder: pooling
            S = hk.get_parameter(
                    "S",
                    shape=[1, self.n_heads*self.n_neurons_encode],
                    dtype=xs.dtype,
                    init=encode_init
                )
            attn_pool = hk.MultiHeadAttention(
                    num_heads=self.n_heads,
                    key_size=self.n_neurons_encode,
                    w_init=encode_init
                )
            ln = hk.LayerNorm(
                    axis=-1, 
                    param_axis=-1, 
                    create_scale=True, 
                    create_offset=True
                )
            Z = ln(attn_pool(S, Z, Z))


            ## decoder: output
            output_block = hk.Sequential([
                hk.Linear(self.n_heads*self.n_neurons_decode),
                self.act,
                hk.Linear(1)
            ])

            return np.squeeze(output_block(Z))

        # output the velocity field
        else:
            attn_block = hk.MultiHeadAttention(
                                num_heads=1, 
                                key_size=self.d,
                                w_init=encode_init
                            )

            return attn_block(Z, Z, Z).ravel()


class Quadratic(hk.Module):
    """Simple module implementing a (confining) quadratic in the state space."""

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        d = x.shape[-1]
        L_init = hk.initializers.TruncatedNormal(1.0 / onp.sqrt(d))
        L = hk.get_parameter("L", shape=[d, d], dtype=x.dtype, init=L_init)
        center = hk.get_parameter("center", shape=[d], dtype=x.dtype, init=np.zeros)

        # note the sign: potential = -E, so we want a concave quadratic.
        return -(x - center) @ (L @ L.T) @ (x - center)


def construct_ips_transformer(
    act: Callable[[np.ndarray], np.ndarray],
    n_neurons_encode: int,
    n_layers_attn: int,
    n_heads: int,
    n_neurons_decode: int,
    N: int,
    d: int,
    is_gradient: bool,
    embed_particles: bool,
    residual_blocks: bool
) -> Tuple[
        Callable[[hk.Params, np.ndarray], np.ndarray],  # score
        Callable[[hk.Params, np.ndarray], np.ndarray]   # potential
    ]:
    net = lambda xs: \
            IPSTransformer(
                    n_neurons_encode=n_neurons_encode, 
                    n_layers_attn=n_layers_attn, 
                    n_heads=n_heads, 
                    n_neurons_decode=n_neurons_decode, 
                    N=N, 
                    d=d,
                    is_gradient=is_gradient,
                    embed_particles=embed_particles,
                    residual_blocks=residual_blocks,
                    act=act
                )(xs)

    if is_gradient:
        score = jax.grad(net)
        potential_network = hk.without_apply_rng(hk.transform(net))
        score_network = hk.without_apply_rng(hk.transform(score))
    else:
        potential_network = None
        score_network = hk.without_apply_rng(hk.transform(net))

    return score_network, potential_network


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
    is_gradient: bool = True,
    confining_quadratic: bool = True,
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

    # only want to add a quadratic if we are modeling the potential
    if confining_quadratic and is_gradient:
        net = lambda x: hk.Sequential(
            construct_mlp_layers(n_hidden, n_neurons, act, output_dim, residual_blocks)
        )(x) + Quadratic()(x)
    else:
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
