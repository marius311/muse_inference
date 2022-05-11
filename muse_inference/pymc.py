
# adapted from https://github.com/junpenglao/Planet_Sakaar_Data_Science/blob/main/PyMC3QnA/discourse_8528%20(MUSE).ipynb
# special thanks to Junpeng Lao

from copy import copy

import aesara
import aesara.tensor as at
import arviz as az
import numpy as np
import scipy.stats as st

import pymc as pm
from pymc.distributions import joint_logpt

from . import MuseProblem


class PyMCMuseProblem(MuseProblem):

    def __init__(self, model, params=None):

        super().__init__()
        
        # dev note: "RVs" are the Aesara tensor variables PyMC uses for
        # simulation. "vals" are the (possibly transformed) Aesara
        # tensor variables PyMC uses for posterior evaluation, and
        # which are tagged with the transform function

        self.model = model

        # extract/save actual observed data
        self.x = [model.rvs_to_values[v].data for v in model.observed_RVs]

        # get log-posterior density, removing conditioning on x so we
        # can swap in other values later
        rvs_to_values, logpt = unconditioned_logpt(model)

        # automatically figure out which variables correspond to
        # (x,z,θ). the x have observed values, the θ are paramaters
        # with no parent, and remaining free variables are z
        model_graph = pm.model_graph.ModelGraph(model)
        if params: 
            θ_RVs = [var for var in model.basic_RVs if var.name in params]
        else:
            θ_RVs = [var for var in model.basic_RVs if model_graph.get_parents(var) == set()]
        x_RVs = self.model.observed_RVs
        z_RVs = [var for var in model.free_RVs if var not in θ_RVs]
        x_vals = [rvs_to_values[v] for v in x_RVs]
        θ_vals = [rvs_to_values[v] for v in θ_RVs]
        z_vals = [rvs_to_values[v] for v in z_RVs]

        # get log-prior density
        logpriort = at.sum([logpt.sum() for (logpt, var) in zip(model.logpt(sum=False), model.basic_RVs) if var in θ_RVs])

        # apply forward or backward transformation stored in val to
        # the variable x, accounting for case where val has no transform
        forward_transform  = lambda val, x: val.tag.transform.forward(x)  if hasattr(val.tag, "transform") else x
        backward_transform = lambda val, x: val.tag.transform.backward(x) if hasattr(val.tag, "transform") else x

        # θ transforms
        self._transform_θ     = aesara.function(θ_vals, [forward_transform(val, val)  for val in θ_vals])
        self._inv_transform_θ = aesara.function(θ_vals, [backward_transform(val, val) for val in θ_vals])

        # create function for sampling x and transformed + raveled z given untransformed θ
        z_RVs_trans_vec = at.concatenate([forward_transform(val, rv) for (rv,val) in zip(z_RVs, z_vals)], axis=0)
        self._sample_x_z = aesara.function(θ_RVs, x_RVs + [z_RVs_trans_vec])

        # create variables for the raveled versions of all the z and θ variables
        z_vec, z_unvec = self._ravel_unravel_tensors(z_vals, "z_vec")
        θ_vec, θ_unvec = self._ravel_unravel_tensors(θ_vals, "θ_vec")

        # create necessary functions, gradients, and hessians, in
        # terms of the transformed + raveled z and transformed or
        # untransformed + raveled θ
        def get_gradients(θ_unvec):
            logpt_vec = aesara.clone_replace(logpt, dict(zip(z_vals+θ_vals, z_unvec+θ_unvec)))
            logpriort_vec = aesara.clone_replace(logpriort, dict(zip(θ_vals, θ_unvec)))

            dθlogpt_vec = aesara.grad(logpt_vec, wrt=θ_vec)
            dzlogpt_vec = aesara.grad(logpt_vec, wrt=z_vec)
            dlogpriort_vec = aesara.grad(logpriort_vec, wrt=θ_vec)
            d2logpriort_vec = aesara.gradient.hessian(logpriort_vec, wrt=θ_vec)

            dθlogp = aesara.function(x_vals + [z_vec, θ_vec], dθlogpt_vec)
            logp_dzlogp = aesara.function(x_vals + [z_vec, θ_vec], [logpt_vec, dzlogpt_vec])
            dlogprior_d2logprior = aesara.function([θ_vec], [dlogpriort_vec, d2logpriort_vec])

            return dθlogp, logp_dzlogp, dlogprior_d2logprior

        θ_unvec_untrans = [forward_transform(val, x) for (val, x) in zip(θ_vals, θ_unvec)]
        self._dθlogp, self._logp_dzlogp, self._dlogprior_d2logprior = get_gradients(θ_unvec)
        self._dθlogp_untransθ, self._logp_dzlogp_untransθ, self._dlogprior_d2logprior_untransθ = get_gradients(θ_unvec_untrans)


    def transform_θ(self, θ):
        return self._transform_θ(*θ)

    def inv_transform_θ(self, θ):
        return self._inv_transform_θ(*θ)

    def sample_x_z(self, rng, θ):
        self.model.rng_seeder = rng
        for rng in self.model.rng_seq:
            state = self.model.next_rng().get_value(borrow=True).get_state()
            self.model.rng_seq.pop() # the call to next_rng undesiredly (for this) added it to rng_seq, so remove it
            rng.get_value(borrow=True).set_state(state)
        *x, z = self._sample_x_z(*np.atleast_1d(θ))
        return (x, z)

    def gradθ_logLike(self, x, zvec, θvec, transformed_θ):
        _dθlogp = self._dθlogp if transformed_θ else self._dθlogp_untransθ
        return _dθlogp(*x, zvec, np.atleast_1d(θvec))

    def logLike_and_gradz_logLike(self, x, zvec, θvec):
        return self._logp_dzlogp(*x, zvec, np.atleast_1d(θvec))

    def gradθ_and_hessθ_logPrior(self, θ, transformed_θ):
        _dlogprior_d2logprior = self._dlogprior_d2logprior if transformed_θ else self._dlogprior_d2logprior_untransθ
        return _dlogprior_d2logprior(np.atleast_1d(θ))

    def _ravel_unravel_tensors(self, RVs, name):
        RVs_raveled = at.vector(name=name)
        test_point = self.model.recompute_initial_point()
        RVs_val = [test_point[v.name] for v in RVs]
        split_point = np.cumsum([0] + [v.size for v in RVs_val])
        RVs_unraveled = []
        for (i, val) in enumerate(RVs_val):
            RVs_unraveled.append(at.reshape(RVs_raveled[split_point[i]:split_point[i+1]], val.shape))
        return RVs_raveled, RVs_unraveled

    def _split_rng(self, rng: np.random.SeedSequence, N):
        return [np.random.RandomState(s) for s in copy(rng).generate_state(N)]


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

