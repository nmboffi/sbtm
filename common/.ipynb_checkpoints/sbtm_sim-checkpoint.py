"""
Base class for score-based transport modeling.

Nicholas M. Boffi
10/27/22
"""


from dataclasses import dataclass
from typing import Callable, Tuple, Union
import haiku as hk
import jax
import numpy as onp
from jaxlib.xla_extension import Device
import optax


State = onp.ndarray
Time = float


from . import networks
from . import rollouts


@dataclass
class SBTMSim:
    """
    Base class for all SBTM simulations.
    Contains simulation parameters common to all SBTM approaches.
    """
    # initial condition fitting
    n_max_init_opt_steps: int
    init_learning_rate: float
    init_ltol: float
    sig0: float
    mu0: onp.ndarray

    # system parameters
    drift: Callable[[State, Time], State]
    force_args: Tuple
    amp: Callable[[Time], float]
    freq: float
    dt: float
    D: onp.ndarray
    D_sqrt: onp.ndarray
    n: int
    d: int
    N: int

    # timestepping
    ltol: float
    gtol: float
    n_opt_steps: int
    learning_rate: float

    # network parameters
    n_hidden: int
    n_neurons: int
    act: Callable[[State], State]
    residual_blocks: bool
    interacting_particle_system: bool

    # general simulation parameters
    key: onp.ndarray
    params_list: list
    all_samples: dict

    # output information
    output_folder: str
    output_name: str


    def __init__(self, data_dict: dict) -> None:
        self.__dict__ = data_dict.copy()


    def initialize_forcing(self) -> None:
        self.forcing = lambda x, t: self.drift(x, t, *self.force_args)


    def initialize_network_and_optimizer(self) -> None:
        """Initialize the network parameters and optimizer."""
        if self.interacting_particle_system:
            self.score_network, self.potential_network = \
                    networks.construct_interacting_particle_system_network(
                            self.n_hidden, 
                            self.n_neurons, 
                            self.N, 
                            self.d, 
                            self.act, 
                            self.residual_blocks
                        )

            ## TODO: allow for time-dependence.
            example_x = onp.zeros(self.N*self.d)
        else:
            self.score_network, self.potential_network = \
                networks.construct_score_network(
                    self.d,
                    self.n_hidden,
                    self.n_neurons,
                    self.act,
                    is_gradient=True,
                    confining_quadratic=False
                )
            
            example_x = onp.zeros(self.d)
            
        self.key, sk = jax.random.split(self.key)
        init_params = self.score_network.init(self.key, example_x)
        self.params_list = [init_params]
        network_size = jax.flatten_util.ravel_pytree(init_params)[0].size
        print(f'Number of parameters: {network_size}')
        print(f'Number of parameters needed for overparameterization: ' \
                + f'{self.n*example_shape[0]}')

        # set up the optimizer
        self.opt = optax.radam(self.learning_rate)
        self.opt_state = self.opt.init(init_params)

        # set up batching for the score
        self.batch_score = jax.vmap(self.score_network.apply, in_axes=(None, 0))


    def fit_init(self, cpu: Device, gpu: Device) -> None:
        """Fit the initial condition."""
        # draw samples
        samples_shape = (self.n, self.N*self.d)
        init_samples = self.sig0*onp.random.randn(*samples_shape) + self.mu0[None, :]

        # set up optimizer
        init_params = jax.device_put(self.params_list[0], gpu)
        opt = optax.adabelief(self.init_learning_rate)
        opt_state = opt.init(init_params)


        ## TODO: Make this more general to allow for time-dependent initial condition.
        init_params = rollouts.fit_initial_condition(
                            self.n_max_init_opt_steps,
                            self.init_ltol,
                            init_params,
                            self.sig0,
                            self.mu0,
                            self.score_network,
                            opt,
                            opt_state,
                            init_samples
                        )


        self.params_list = [jax.device_put(init_params, device=cpu)]
        self.all_samples = {'SDE': [init_samples], 'learned': [init_samples]}
