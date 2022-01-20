
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

        super().__init__()
        
        self.model = model

        # extract/save actual observed data
        self.x = [model.rvs_to_values[v].data for v in model.observed_RVs]

        # automatically figure out which variables correspond to
        # (x,z,θ). the x have observed values, the θ are paramaters
        # with no parent, and remaining free variables are z
        model_graph = pm.model_graph.ModelGraph(model)
        self.x_RVs = self.model.observed_RVs
        self.θ_RVs = [var for var in model.basic_RVs if model_graph.get_parents(var) == set()]
        self.z_RVs = [var for var in model.free_RVs if var not in self.θ_RVs]

        # remove conditioning on observed variables so everything is free
        rvs_to_values, logpt = unconditioned_logpt(model)

        # create variables for the raveled versions of all the z and θ variables
        z_RV_vals = [rvs_to_values[v] for v in self.z_RVs]
        z_RVs_raveled, z_RVs_unraveled = self.ravel_unravel_RVs(z_RV_vals, "z_raveled")
        θ_RV_vals = [rvs_to_values[v] for v in self.θ_RVs]
        θ_RVs_raveled, θ_RVs_unraveled = self.ravel_unravel_RVs(θ_RV_vals, "θ_raveled")
        
        # create likelihood function and gradients in terms of the raveled z and θ
        raveled_logpt = aesara.clone_replace(logpt, dict(zip(z_RV_vals+θ_RV_vals, z_RVs_unraveled+θ_RVs_unraveled)))
        raveled_inputs = [rvs_to_values[v] for v in self.x_RVs] + [z_RVs_raveled, θ_RVs_raveled]
        dθlogpt = aesara.grad(raveled_logpt, wrt=θ_RVs_raveled)
        dzlogpt = aesara.grad(raveled_logpt, wrt=z_RVs_raveled)
        self.dθlogp = aesara.function(raveled_inputs, dθlogpt)
        self.logp_dzlogp = aesara.function(raveled_inputs, [raveled_logpt, dzlogpt])
        
        # prior gradient and hessian
        logpriort = at.sum([logpt.sum() for (logpt, var) in zip(model.logp_elemwiset(), model.basic_RVs) if var in self.θ_RVs])
        raveled_logpriort = aesara.clone_replace(logpriort, dict(zip(θ_RV_vals, θ_RVs_unraveled)))
        dlogpriort = aesara.grad(raveled_logpriort, wrt=θ_RVs_raveled)
        d2logpriort = aesara.gradient.hessian(raveled_logpriort, wrt=θ_RVs_raveled)
        self.dlogprior_d2logprior = aesara.function([θ_RVs_raveled], [dlogpriort, d2logpriort])

    def sample_x_z(self, rng, θ):
        self.model.rng_seeder = rng
        for rng in self.model.rng_seq:
            state = self.model.next_rng().get_value().get_state()
            self.model.rng_seq.pop() # the call to next_rng undesiredly (for this) added it to rng_seq
            rng.get_value(borrow=True).set_state(state)
        x_z = aesara.function(self.θ_RVs, self.x_RVs + self.z_RVs)(*np.atleast_1d(θ))
        x, z = (x_z[:len(self.x_RVs)], x_z[len(self.x_RVs):])
        ravel = self.ravel_unravel(z)[0]
        return (x, ravel(z))

    def gradθ_logLike(self, x, z, θ):
        return self.dθlogp(*x, z, np.atleast_1d(θ))

    def logLike_and_gradz_logLike(self, x, z, θ):
        return self.logp_dzlogp(*x, z, np.atleast_1d(θ))

    def gradθ_and_hessθ_logPrior(self, θ):
        return self.dlogprior_d2logprior(np.atleast_1d(θ))

    def ravel_unravel_RVs(self, RVs, name):
        RVs_raveled = at.vector(name=name)
        test_point = self.model.recompute_initial_point()
        RVs_val = [test_point[v.name] for v in RVs]
        split_point = np.cumsum([0] + [v.size for v in RVs_val])
        RVs_unraveled = []
        for (i, val) in enumerate(RVs_val):
            RVs_unraveled.append(at.reshape(RVs_raveled[split_point[i]:split_point[i+1]], val.shape))
        return RVs_raveled, RVs_unraveled




def unconditioned_logpt(model, jacobian: bool = True):
    """
    Given a model where some of the variables have been "conditioned",
    ie an observed value has been provided, return a tuple of
    `(rvs_to_values, logpt)` where all of the variables have been made
    free again.
    """
    
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

