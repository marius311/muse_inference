
# adapted from https://github.com/junpenglao/Planet_Sakaar_Data_Science/blob/main/PyMC3QnA/discourse_8528%20(MUSE).ipynb
# special thanks to Junpeng Lao

import aesara
import aesara.tensor as at
import arviz as az
import numpy as np
import scipy.stats as st

import pymc as pm
from pymc.distributions import logpt as joint_logpt

from . import MuseProblem


class PyMCMuseProblem(MuseProblem):

    def __init__(self, model):
        
        self.model = model

        # automatically figure out which variables correspond to
        # (x,z,θ). the x are observed, the θ are priors with no
        # parent, and remaining free variables are z
        model_graph = pm.model_graph.ModelGraph(model)
        self.x_RVs = self.model.observed_RVs
        self.θ_RVs = [var for var in model.basic_RVs if model_graph.get_parents(var) == set()]
        self.z_RVs = [var for var in model.free_RVs if var not in self.θ_RVs]

        # compile aesara functions for the likelihood gradients we need, as a function of (x,z,θ)
        rvs_to_values, logpt = unconditioned_logpt(model)
        ordered_values = [rvs_to_values[v] for v in (self.x_RVs + self.z_RVs + self.θ_RVs)]
        θ_vals = [rvs_to_values[v] for v in self.θ_RVs]
        logp = aesara.function(ordered_values, logpt)
        dθlogpt = pm.gradient(logpt, θ_vals)
        dzlogpt = pm.gradient(logpt, [rvs_to_values[v] for v in self.z_RVs])
        self.dθlogp = aesara.function(ordered_values, [dθlogpt])
        self.logp_dzlogp = aesara.function(ordered_values, [logpt, dzlogpt])

        # compile aseara function for prior gradient/hessian
        logpriort = at.sum([logpt.sum() for (logpt, var) in zip(model.logp_elemwiset(), model.basic_RVs) if var in self.θ_RVs])
        dlogpriort = pm.gradient(logpriort, θ_vals)
        d2logpriort = pm.hessian(logpriort, θ_vals)
        self.dlogprior_d2logprior = aesara.function(θ_vals, [dlogpriort, d2logpriort])



    def sample_x_z(self, rng, θ):
        self.model.rng_seeder = rng
        for rng in self.model.rng_seq:
            state = self.model.next_rng().get_value().get_state()
            self.model.rng_seq.pop() # the call to next_rng undesiredly added it to rng_seq
            rng.get_value(borrow=True).set_state(state)
        _sample_x_z = aesara.function(self.θ_RVs, self.x_RVs + self.z_RVs)
        return _sample_x_z(θ)

    def gradθ_logLike(self, x, z, θ):
        return self.dθlogp(x, z, θ)

    def logLike_and_gradz_logLike(self, x, z, θ):
        return self.logp_dzlogp(x, z, θ)

    def grad_hess_θ_logPrior(self, θ):
        return self.dlogprior_d2logprior(θ)



def unconditioned_logpt(model, jacobian: bool = True):
    
    if model.potentials:
        raise Exception("Does not work with model that contains potentials")

    rvs_to_values = {}
    for var in model.rvs_to_values.keys():
        if var in model.observed_RVs:
            value_var = var.type()
            value_var.name = var.name
        else:
            value_var = model.rvs_to_values[var]
        if value_var is not None:
            rvs_to_values[var] = value_var
        else:
            raise ValueError(
                f"Requested variable {var} not found among the model variables"
            )
            
    logpt = joint_logpt(list(rvs_to_values.keys()), rvs_to_values, sum=True, jacobian=jacobian)
    return rvs_to_values, logpt

