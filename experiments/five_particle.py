import jax
from jax import numpy as np
from jax import vmap
import optax
import numpy as onp
from typing import Callable, Tuple
import sys
import argparse
sys.path.append('../')


import drifts
import networks
import rollouts
import sbtm_sequential


from tqdm.auto import tqdm
import dill as pickle


print(jax.devices())
gpu = jax.devices('gpu')[0]
cpu = jax.devices('cpu')[0]


######## Configuation Parameters #########
## physical, non-forcing parameters
d = 2
D = 0.25
dt = 1e-3
tf = 5
n = 10000
n_time_steps = int(tf / dt)
store_fac = 25


## configure random seed
repeatable_seed = False
if repeatable_seed:
    key = jax.random.PRNGKey(42)
    onp.random.seed(42)
else:
    key = jax.random.PRNGKey(onp.random.randint(1000))


## anharmonic w/ Gaussian interaction system
N = 5
r = 0.5
A = 2.0
gamma = 5
R = (gamma*N)**0.5*r
B = 4*D/R**2
mask = onp.ones(N*d)


## configure trap motion
amp = lambda t: 3
freq = np.pi


## set up forcing parameters
drift = drifts.anharmonic_gaussian


## initial distribution parameters
sig0 = 0.5


### setup optimizer
init_learning_rate = 5e-3
learning_rate = 5e-3
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
interacting_particle_system = True


### output data
base_folder   = '/scratch/nb3397/results/sbtm_results'
system_folder = '5particle_test'
output_folder = f'{base_folder}/{system_folder}'
################################################


def construct_simulation(
    use_ODE: bool,
    use_SDE: bool,
    noise_fac: float,
    compute_mut: Callable,
    force_args: Tuple,
    mu0: np.ndarray,
    name_str: str
):
    output_name = f'denoising/{name_str}.npy'
    sim = sbtm_sequential.DenoisingSequentialSBTM(
        n_max_init_opt_steps=n_max_init_opt_steps,
        init_learning_rate=init_learning_rate,
        init_ltol=init_ltol,
        sig0=sig0,
        mu0=mu0,
        drift=drift,
        force_args=force_args,
        amp=amp,
        freq=freq,
        dt=dt,
        D=D,
        D_sqrt=onp.sqrt(D),
        n=n,
        N=N,
        d=d,
        ltol=ltol,
        gtol=gtol,
        n_opt_steps=n_opt_steps,
        learning_rate=learning_rate,
        n_hidden=n_hidden,
        n_neurons=n_neurons,
        act=act,
        residual_blocks=residual_blocks,
        interacting_particle_system=interacting_particle_system,
        key=key,
        params_list=[],
        all_samples=dict(),
        output_folder=output_folder,
        output_name=output_name,
        n_time_steps=n_time_steps,
        noise_fac=noise_fac,
        use_ODE=use_ODE,
        use_SDE=use_SDE,
        save_fac=250,
        store_fac=store_fac,
        mask=mask,
        entropies=[],
        means={'SDE': [], 'learned': []},
        covs={'SDE': [], 'learned': []}
    )

    return sim


def get_simulation_parameters():
    """Process command line arguments and set up associated simulation parameters."""
    parser = argparse.ArgumentParser(
            description='Run an SBTM simulation from the command line.'
        )
    parser.add_argument('--use_ODE',   type=int,  help='Train from ODE samples?')
    parser.add_argument('--use_SDE',   type=int,  help='Train from SDE samples?')
    parser.add_argument('--noise_fac', type=float, help='Denoising parameter.')
    parser.add_argument('--circular',  type=int,  help='Circular or linear trap motion?')
    args = parser.parse_args()

    if bool(args.circular):
        compute_mut = lambda t: amp(t)*np.array([np.cos(freq*t), np.sin(freq*t)])
        name_str = f'circular/nf={args.noise_fac}_ODE={args.use_ODE}_SDE={args.use_SDE}'
    else:
        compute_mut = lambda t: amp(t)*np.array([np.cos(freq*t), 0])
        name_str = f'linear/nf={args.noise_fac}_ODE={args.use_ODE}_SDE={args.use_SDE}'

    force_args = (A, r, B, N, d, compute_mut)
    mu0 = onp.tile(compute_mut(0), N)

    return bool(args.use_ODE), bool(args.use_SDE), args.noise_fac, compute_mut, force_args, mu0, name_str


if __name__ == '__main__':
    sim = construct_simulation(*get_simulation_parameters())
    sim.initialize_network_and_optimizer()
    sim.fit_init(cpu=cpu, gpu=gpu)
    sim.initialize_forcing()
    sim.solve_fpe_sequential(cpu=cpu, gpu=gpu)
