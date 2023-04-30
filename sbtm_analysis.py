"""
Code for analyzing the ouput of an SBTM simulation.

Nicholas M. Boffi
11/2/22
"""


import sbtm_sim
from jax import jit, jacfwd, vmap
import jax.numpy as np
import numpy as onp
from functools import partial
from tqdm.notebook import tqdm as tqdm
import haiku as hk
from typing import Callable, Union, Tuple
from jax import grad, jacfwd, device_put
from jaxlib.xla_extension import Device


State, Time = np.ndarray, float
import updates


class SBTMAnalysis(sbtm_sim.SBTMSim):
    times: onp.ndarray
    ngt_fac: int


    def __init__(self, data_dict: dict, ngt_fac: int) -> None:
        self.__dict__ = data_dict.copy()
        self.nsteps = len(self.params_list)
        self.times = self.dt*onp.arange(self.nsteps)
        self.ngt_fac = ngt_fac


    def compute_noise_free(self) -> None:
        """Roll out noise-free trajectories."""
        traj, self.key = \
                updates.rollout_EM_trajs(
                        np.copy(self.all_samples['SDE'][0]),
                        self.nsteps-1,
                        0,
                        self.dt,
                        self.key,
                        self.forcing,
                        self.D_sqrt,
                        noisy=False
                    )
        self.all_samples['noise_free'] = onp.array(traj)


    def compute_ground_truth(self) -> None:
        """Roll out ground truth trajectories for comparison."""
        x0s = self.mu0 + self.sig0*onp.random.randn(self.ngt_fac*self.n, self.N*self.d)
        traj, self.key = \
                updates.rollout_EM_trajs(
                        x0s,
                        self.nsteps-1,
                        0,
                        self.dt,
                        self.key,
                        self.forcing,
                        self.D_sqrt,
                        noisy=True
                        )

        self.all_samples['ground_truth'] = onp.array(traj)


    def compute_moments(self) -> None:
        self.compute_means()
        self.compute_covs()


    def compute_means(self) -> None:
        self.means = {}
        for key in self.all_samples.keys():
            self.means[key] = onp.zeros((self.nsteps, self.N*self.d))

        for curr_step in tqdm(range(self.nsteps)):
            for key in self.means.keys():
                self.means[key][curr_step] = \
                        np.mean(self.all_samples[key][curr_step], axis=0)


    def compute_covs(self) -> None:
        self.covs = {}
        for key in self.all_samples.keys():
            self.covs[key] = onp.zeros((self.nsteps, self.N*self.d, self.N*self.d))

        for curr_step in tqdm(range(self.nsteps)):
            for key in self.covs.keys():
                self.covs[key][curr_step] = \
                        np.cov(self.all_samples[key][curr_step], rowvar=False)


    def compute_entropy_rate_trajectories(
        self, 
        nt_compute: int,  
        div: bool,
        sigmas: list
    ) -> None:
        self.entropy_rates = {
                'learned': onp.zeros(nt_compute),
                'noise_free': onp.zeros(nt_compute)
                }

        sigma_keys = [f'kde_sigma{sigma}' for sigma in sigmas]
        for key in sigma_keys:
            self.entropy_rates[key] = onp.zeros(nt_compute)

        ## compute the learned estimate.
        ts = np.arange(nt_compute)*self.dt
        for tt, curr_t in enumerate(tqdm(ts)):
            curr_params = self.params_list[tt]
            self.entropy_rates['learned'][tt] = \
                    compute_entropy_rate(
                        self.all_samples['learned'][tt], 
                        curr_t, 
                        curr_params,
                        self.D,
                        self.forcing,
                        self.score_network,
                        noise_free=False,
                        div=div
                    )

            self.entropy_rates['noise_free'][tt] = \
                    compute_entropy_rate(
                        self.all_samples['noise_free'][tt], 
                        curr_t, 
                        curr_params, 
                        self.D,
                        self.forcing,
                        self.score_network,
                        noise_free=True,
                        div=div
                    )


            for sigma, sigma_key in zip(sigmas, sigma_keys):
                self.entropy_rates[sigma_key][tt] = \
                        compute_kde_entropy_rate(
                            self.all_samples['learned'][tt],
                            curr_t,
                            self.all_samples['learned'][tt],
                            sigma,
                            self.D,
                            self.forcing
                        )


###### Entropy Calculation #######
@partial(jit, static_argnums=(4, 5, 6, 7))
def compute_sample_entropy_rate(
    sample: np.ndarray,
    t: float,
    params: hk.Params,
    D: Union[np.ndarray, float],
    forcing: Callable[[State, Time], State],
    score_network: Callable[[hk.Params, State], State],
    noise_free: bool,
    div: bool,
) -> float:
    if div:
        if noise_free:
            drift = lambda x: forcing(x, t)
        else:
            drift = lambda x: forcing(x, t) - D*score_network.apply(params, x)

        return np.trace(jacfwd(drift)(sample))

    else:
        st = score_network.apply(params, sample)
        
        if noise_free:
            vt = forcing(sample, t)
        else:
            vt = forcing(sample, t) - D*st

        return -np.sum(st*vt)


@partial(jit, static_argnums=(4, 5, 6, 7))
def compute_entropy_rate(
    samples: np.ndarray, 
    t: float, 
    params: hk.Params, 
    D: np.ndarray,
    forcing: Callable[[State, Time], State],
    score_network: Callable,
    noise_free: bool,
    div: bool
) -> float:
    Nones = (None,)*7
    return np.mean(
        vmap(
            compute_sample_entropy_rate, in_axes=(0, *Nones)
        )(samples, t, params, D, forcing, score_network, noise_free, div)
    )


@jit
def kde_score(
    x: np.ndarray,
    samples: np.ndarray,
    sigma: float
) -> np.ndarray:
    dists = x[None, :] - samples # n x d 
    sqdists = np.sum(dists**2, axis=1) # n
    exps = np.exp(-sqdists / (2*sigma**2)) # n

    return np.sum(dists * exps[:, None], axis=0) / np.sum(exps) / sigma**2


@jit
def kde(
    x: np.ndarray,
    samples: np.ndarray,
    sigma: float
) -> float:
    dists = x[None, :] - samples # n x d 
    sqdists = np.sum(dists**2, axis=1) # n
    exps = np.exp(-sqdists / (2*sigma**2)) # n

    return (2*np.pi*sigma**2)**(-1/2) * np.mean(exps)


@partial(jit, static_argnums=5)
def compute_kde_sample_entropy_rate(
    eval_sample: np.ndarray,
    t: float,
    samples: np.ndarray,
    sigma: float,
    D: Union[np.ndarray, float],
    forcing: Callable[[State, Time], State]
) -> float:
    drift = lambda x: forcing(x, t) - D*kde_score(x, samples, sigma)
    return np.trace(jacfwd(drift)(eval_sample))


@partial(jit, static_argnums=5)
def compute_kde_entropy_rate(
    eval_samples: np.ndarray,
    t: float,
    samples: np.ndarray,
    sigma: float,
    D: np.ndarray,
    forcing: Callable[[State, Time], State]
) -> float:
    Nones = (None,)*5
    return np.mean(
        vmap(
            compute_kde_sample_entropy_rate, in_axes=(0, *Nones)
        )(eval_samples, t, samples, sigma, D, forcing)
    )


@jit
def compute_kde_entropy(
    eval_samples: np.ndarray,
    samples: np.ndarray,
    sigma: float
) -> float:
    kdes = vmap(kde, in_axes=(0, None, None))(eval_samples, samples, sigma)
    return -np.mean(np.log(kdes))


compute_kde_entropy_traj = vmap(compute_kde_entropy, in_axes=(0, 0, None))
