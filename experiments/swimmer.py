import jax
from jax import numpy as np
from jax import vmap
import optax
import numpy as onp
from typing import Callable, Tuple
import sys
import argparse
sys.path.append('../')


import common.drifts as drifts
import common.networks as networks
import common.rollouts as rollouts
import common.sbtm_sequential as sbtm_sequential


from tqdm.auto import tqdm
import dill as pickle


print(jax.devices())
gpu = jax.devices('gpu')[0]
cpu = jax.devices('cpu')[0]


######## Configuation Parameters #########
d      = 2
gamma  = 0.1
D      = gamma
D_sqrt = onp.sqrt(D)
mask   = onp.array([0, 1])
dt     = 1e-3
tf     = 10 / gamma
n      = 10000
n_time_steps = int(tf / dt)
store_fac = 25


## configure random seed
repeatable_seed = False
if repeatable_seed:
    key = jax.random.PRNGKey(42)
    onp.random.seed(42)
else:
    key = jax.random.PRNGKey(onp.random.randint(1000))


## set up forcing parameters
drift      = drifts.active_swimmer
force_args = (gamma,)


## initial distribution parameters
sig0 = 1.0
mu0  = np.zeros(d)

### setup optimizer
init_learning_rate = 5e-3
init_ltol = 1e-6
ltol = np.inf
gtol = 0.5
n_opt_steps = 25
n_max_init_opt_steps = int(1e4)


### Set up neural network
n_hidden = 3
n_neurons = 32
act = jax.nn.swish
residual_blocks = False


### output data
base_folder = '/scratch/nb3397/sbtm_results'
system_folder = 'active_swimmer'
output_folder = f'{base_folder}/{system_folder}'
################################################


def construct_simulation(
    learning_rate: float,
    noise_fac: float,
    name_str: str
):
    output_name = f'{name_str}.npy'
    sim = sbtm_sequential.DenoisingSequentialSBTM(
        n_max_init_opt_steps=n_max_init_opt_steps,
        init_learning_rate=init_learning_rate,
        init_ltol=init_ltol,
        sig0=sig0,
        mu0=mu0,
        drift=drift,
        force_args=force_args,
        amp=None,
        freq=None,
        dt=dt,
        D=D,
        D_sqrt=D_sqrt,
        n=n,
        N=1,
        d=d,
        ltol=ltol,
        gtol=gtol,
        n_opt_steps=n_opt_steps,
        learning_rate=learning_rate,
        n_hidden=n_hidden,
        n_neurons=n_neurons,
        act=act,
        residual_blocks=residual_blocks,
        interacting_particle_system=False,
        key=key,
        params_list=[],
        all_samples=dict(),
        output_folder=output_folder,
        output_name=output_name,
        n_time_steps=n_time_steps,
        noise_fac=noise_fac,
        use_ODE=True,
        use_SDE=False,
        store_fac=store_fac,
        save_fac=250,
        means={'SDE': [], 'learned': []},
        covs={'SDE': [], 'learned': []},
        entropies=[],
        mask=mask
    )


    return sim


def get_simulation_parameters():
    """Process command line arguments and set up associated simulation parameters."""
    parser = argparse.ArgumentParser(description='Run an SBTM simulation from the command line.')
    parser.add_argument('learning_rate', type=float,  help='Learning rate?')
    parser.add_argument('noise_fac', type=float, help='Denoising parameter.')
    args = parser.parse_args()

    name_str = f'lr={args.learning_rate}_nf={args.noise_fac}'
    return args.learning_rate, args.noise_fac, name_str


if __name__ == '__main__':
    sim = construct_simulation(*get_simulation_parameters())
    sim.initialize_network_and_optimizer()
    sim.fit_init(cpu=cpu, gpu=gpu)
    sim.initialize_forcing()
    sim.solve_fpe_sequential(cpu=cpu, gpu=gpu)
