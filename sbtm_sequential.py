"""
Code for sequential SBTM.

Nicholas M. Boffi
10/27/22
"""


from dataclasses import dataclass
import jax
import jax.numpy as np
from jax import vmap
import numpy as onp
from tqdm.auto import tqdm as tqdm
import dill as pickle
import time
from jaxlib.xla_extension import Device
from haiku import Params
from typing import Tuple


import sbtm_sim
import updates
import losses
from sbtm_analysis import compute_entropy_rate


@dataclass
class SequentialSBTM(sbtm_sim.SBTMSim):
    n_time_steps: int
    use_SDE: bool
    use_ODE: bool
    save_fac: int
    store_fac: int
    means: dict
    covs: dict
    entropies: list
    mask: np.ndarray


    def setup_loss(self):
        """Define the loss function. """
        raise NotImplementedError("Please implement in the inheriting class.")

    
    def setup_loss_fn_args(self, gpu: Device) -> Tuple:
        """Define the arguments to the loss function other than parameters."""
        raise NotImplementedError("Please implement in the inheriting class.")
        

    def setup_batched_steppers(self):
        """Construct convenience functions to step the particles."""
        self.step_learned = vmap(
                lambda params, t, sample: updates.update_particles(
                    sample, 
                    t, 
                    params, 
                    self.D, 
                    self.dt,  
                    self.forcing, 
                    self.score_network.apply,
                    self.mask
                ),
                in_axes=(None, None, 0),
                out_axes=0
        )

        self.step_SDE = vmap(
                lambda t, sample, key: updates.update_particles_EM(
                    sample, 
                    t, 
                    self.D_sqrt, 
                    self.dt, 
                    key,  
                    self.forcing,
                    True,
                    self.mask
                ),
                in_axes=(None, 0, 0),
                out_axes=0
        )


    def step_samples(
        self,
        step,
        params: Params,
        t: float,
        samples: np.ndarray,
        SDE_samples: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Step and save both the SDE and ODE samples."""
        # step learned and SDE particles
        samples = self.step_learned(params, t, samples)
        keys = jax.random.split(self.key, num=self.n)
        SDE_samples = self.step_SDE(t, SDE_samples, keys)
        self.key = keys[-1]

        # save new samples
        if (step+1) % self.store_fac == 0:
            self.all_samples['learned'].append(onp.array(samples))
            self.all_samples['SDE'].append(onp.array(SDE_samples))

        return samples, SDE_samples


    def setup_learning_samples(
        self,
        samples: np.ndarray,
        SDE_samples: np.ndarray
    ) -> Tuple[np.ndarray]:
        """Set up samples to optimize over."""
        if self.use_ODE and self.use_SDE:
            opt_samples = (np.vstack((samples, SDE_samples)),)
        elif self.use_ODE:
            opt_samples = (samples,)
        elif self.use_SDE:
            opt_samples = (SDE_samples,)
        else:
            raise ValueError('Need to specify learning from ODE or SDE.')

        return opt_samples


    def compute_moments_and_entropy_production(
        self,
        params: Params,
        t: float,
        samples: np.ndarray, 
        SDE_samples: np.ndarray
    ) -> None:
        ## entropy
        self.entropies.append(
                compute_entropy_rate(samples, t, params, self.D, self.forcing, 
                                     self.score_network, noise_free=False, div=True)
            )

        ## moments
        self.means['SDE'].append(np.mean(SDE_samples, axis=0))
        self.covs['SDE'].append(np.cov(SDE_samples, rowvar=False))

        self.means['learned'].append(np.mean(samples, axis=0))
        self.covs['learned'].append(np.cov(samples, rowvar=False))


    def solve_fpe_sequential(self, cpu: Device, gpu: Device):
        # set up some convenience variables
        self.setup_loss()
        self.setup_batched_steppers()
        nt = len(self.params_list) - 1

        # move needed data to the GPU for speed.
        params = jax.device_put(self.params_list[-1], gpu)
        opt_state = jax.device_put(self.opt_state, gpu)
        samples = jax.device_put(self.all_samples['learned'][-1], gpu)
        SDE_samples = jax.device_put(self.all_samples['SDE'][-1], gpu)

        # store the initial moments and entropy.
        if nt == 0:
            self.compute_moments_and_entropy_production(params, t=0, 
                                                        samples=samples, 
                                                        SDE_samples=SDE_samples)

        ## output progress bar
        with tqdm(range(self.n_time_steps)) as pbar:
            for step in pbar:
                t = (nt*self.store_fac + step)*self.dt
                pbar.set_description(f"Dynamics: t={t:.3f}")
                samples, SDE_samples = self.step_samples(step, params, t, samples, SDE_samples)
                opt_samples = self.setup_learning_samples(samples, SDE_samples)

                ## perform the optimization
                loss_value, grad_norm = np.inf, np.inf
                num_steps_taken = 0
                while (grad_norm > self.gtol):
                    for curr_opt_step in range(self.n_opt_steps):
                        loss_func_args = opt_samples + self.setup_loss_fn_args(gpu)
                        start_time = time.time()
                        params, opt_state, loss_value, grads \
                                = losses.update(
                                        params, 
                                        opt_state, 
                                        self.opt, 
                                        self.loss_func, 
                                        loss_func_args
                                    )
                        end_time = time.time()

                    grad_norm = losses.compute_grad_norm(grads)
                    pbar.set_postfix(
                        loss=loss_value, ltol=self.ltol,
                        grad_norm=grad_norm, gtol=self.gtol,
                        step_time=end_time-start_time
                    )

                if (step+1) % self.store_fac == 0:
                    self.params_list.append(jax.device_put(params, cpu))
                    self.compute_moments_and_entropy_production(params, t, samples, SDE_samples)

                if (step+1) % self.save_fac == 0:
                    self.save_data()

        self.save_data()


    def save_data(self):
        data = vars(self).copy()
        pickle.dump(data, open(f'{self.output_folder}/{self.output_name}', 'wb'))



@dataclass
class DenoisingSequentialSBTM(SequentialSBTM):
    noise_fac: float

    def setup_loss(self):
        def sample_denoising_loss(
            params: Params,
            sample: np.ndarray,
            noise: np.ndarray,
        ) -> float:
            """
            Compute the denoising loss on a single sample, using antithetic sampling
            over the noise for variance reduction.
            """
            loss = 0
            for sign in [-1, 1]:
                perturbed_sample = sample + self.noise_fac*sign*noise
                score = self.mask*self.score_network.apply(params, perturbed_sample)
                loss += np.sum(self.noise_fac*score**2 + 2*sign*score*noise)

            return np.squeeze(loss / 2)

        self.loss_func = lambda params, samples, noise: np.mean(
                vmap(sample_denoising_loss, in_axes=(None, 0, 0))(params, samples, noise)
            )

        if self.use_SDE and self.use_ODE:
            self.n_train_samples = 2*self.n
        else:
            self.n_train_samples = self.n


    def setup_loss_fn_args(self, gpu: Device) -> Tuple:
        """Set up noise arguments for the loss function. """
        noises = onp.random.randn(self.n_train_samples, self.d*self.N)
        loss_func_args = (jax.device_put(noises, gpu),)
        return loss_func_args


@dataclass
class RegularizedSequentialSBTM(SequentialSBTM):
    lam: float


    def setup_loss(self):
        """Define the loss function. """
        self.loss_func = \
                lambda params, samples, div_noises, reg_noises: \
                    losses.sm_loss(
                            params, 
                            samples, 
                            div_noises, 
                            reg_noises,
                            self.lam,
                            self.score_network.apply
                    )
        
        if self.use_SDE and self.use_ODE:
            self.n_train_samples = 2*self.n
        else:
            self.n_train_samples = self.n


    def setup_loss_fn_args(self, gpu: Device) -> Tuple:
        """Set up noise arguments for the loss function. """
        div_noises = onp.random.randn(self.n_train_samples, self.d*self.N)
        reg_noises = onp.random.randn(self.n_train_samples, self.d*self.N)
        return (jax.device_put(div_noises, gpu), jax.device_put(reg_noises, gpu))
