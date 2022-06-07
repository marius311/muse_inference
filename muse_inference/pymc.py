
# adapted from https://github.com/junpenglao/Planet_Sakaar_Data_Science/blob/main/PyMC3QnA/discourse_8528%20(MUSE).ipynb
# special thanks to Junpeng Lao

from copy import copy
from numbers import Number

import aesara
import aesara.tensor as at
import arviz as az
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize

import pymc as pm
from pymc.aesaraf import find_rng_nodes, reseed_rngs
from pymc.distributions import joint_logpt

from .muse_inference import MuseProblem


class PyMCMuseProblem(MuseProblem):

    def __init__(self, model, params=None):

        super().__init__()
        
        # dev note: "RVs" are the Aesara tensor variables PyMC uses for
        # simulation. "vals" are the (transformed, if needed) Aesara
        # tensor variables PyMC uses for posterior evaluation, and
        # which are tagged with the transform function

        self.model = model

        # extract/save actual observed data
        self.x = [model.rvs_to_values[v].data for v in model.observed_RVs]

        # get log-posterior density, removing conditioning on x so we
        # can swap in other values later
        rvs_to_values, logpt = unconditioned_logpt(model)

        # if not provided, automatically figure out which variables
        # correspond to (x,z,θ). the x have observed values, the θ are
        # parameters with no parent, and remaining free variables are z
        model_graph = pm.model_graph.ModelGraph(model)
        if params:
            self.θ_RVs = θ_RVs = [var for var in model.basic_RVs if var.name in params]
        else:
            self.θ_RVs = θ_RVs = [var for var in model.basic_RVs if model_graph.get_parent_names(var) == set()]
        x_RVs = self.model.observed_RVs
        z_RVs = [var for var in model.free_RVs if var not in θ_RVs]
        x_vals = [rvs_to_values[v] for v in x_RVs]
        θ_vals = [rvs_to_values[v] for v in θ_RVs]
        z_vals = [rvs_to_values[v] for v in z_RVs]

        # for seeding sample_x_z
        self.rng_nodes = find_rng_nodes(x_RVs + z_RVs)

        # create ravel / unravel functions for parameter dict
        θ_dict = dict(zip([v.name for v in θ_RVs], aesara.function([], θ_RVs)()))
        self._ravel_θ, self._unravel_θ = self._ravel_unravel(θ_dict)
        # raveling / unraveling z is handled in aesara
        self._ravel_z = self._unravel_z = lambda x: x

        # get log-prior density
        logpriort = at.sum([logpt.sum() for (logpt, var) in zip(model.logpt(sum=False, jacobian=True), model.basic_RVs) if var in θ_RVs])

        # figure out if any transforms are needed and if so, create
        # functions to apply forward or backward transformation stored
        # in val to the variable x, accounting for case where val has
        # no transform
        if any(hasattr(val.tag, "transform") for val in z_vals+θ_vals):
            self._has_θ_transform = True
            forward_transform  = lambda val, x: val.tag.transform.forward(x)  if hasattr(val.tag, "transform") else x
            backward_transform = lambda val, x: val.tag.transform.backward(x) if hasattr(val.tag, "transform") else x
        else:
            self._has_θ_transform = False
            forward_transform = backward_transform = lambda val, x: x

        # create variables for the raveled versions of the RVs and vals that we will need
        z_vals_vec, z_vals_unvec = self._ravel_unravel_tensors(z_vals, "z_vals_vec", is_RV=False)
        θ_vals_vec, θ_vals_unvec = self._ravel_unravel_tensors(θ_vals, "θ_vals_vec", is_RV=False)
        θ_RVs_vec,  θ_RVs_unvec  = self._ravel_unravel_tensors(θ_RVs,  "θ_RVs_vec",  is_RV=True)

        # θ transforms
        self._transform_θ = aesara.function(
            [θ_vals_vec], 
            aesara.clone_replace(cat_flatten([forward_transform(val, val)  for val in θ_vals]), dict(zip(θ_vals, θ_vals_unvec)))
        )
        self._inv_transform_θ = aesara.function(
            [θ_vals_vec],
            aesara.clone_replace(cat_flatten([backward_transform(val, val) for val in θ_vals]), dict(zip(θ_vals, θ_vals_unvec)))
        )

        # create function for sampling x and transformed + raveled z given untransformed + raveled θ
        z_RVs_trans_vec = cat_flatten([forward_transform(val, rv) for (rv,val) in zip(z_RVs, z_vals)])
        self._sample_x_z = aesara.function(
            [θ_RVs_vec], 
            aesara.clone_replace(x_RVs + [z_RVs_trans_vec], dict(zip(θ_RVs, θ_RVs_unvec)))
        )

        # create necessary functions, gradients, and hessians, in
        # terms of the transformed + raveled z and transformed or
        # untransformed + raveled θ
        def get_gradients(θ_unvec):
            logpt_vec = aesara.clone_replace(logpt, dict(zip(z_vals+θ_vals, z_vals_unvec+θ_unvec)))
            logpriort_vec = aesara.clone_replace(logpriort, dict(zip(θ_vals, θ_unvec)))

            dθlogpt_vec = aesara.grad(logpt_vec, wrt=θ_vals_vec)
            dzlogpt_vec = aesara.grad(logpt_vec, wrt=z_vals_vec)
            dlogpriort_vec = aesara.grad(logpriort_vec, wrt=θ_vals_vec)
            d2logpriort_vec = aesara.gradient.hessian(logpriort_vec, wrt=θ_vals_vec)

            logp_dzθlogp = aesara.function(x_vals + [z_vals_vec, θ_vals_vec], [logpt_vec, dzlogpt_vec, dθlogpt_vec])
            dlogprior_d2logprior = aesara.function([θ_vals_vec], [dlogpriort_vec, d2logpriort_vec])

            return logp_dzθlogp, dlogprior_d2logprior

        θ_unvec_untrans = [forward_transform(val, x) for (val, x) in zip(θ_vals, θ_vals_unvec)]
        self._logp_dzθlogp_transθ,   self._dlogprior_d2logprior_transθ   = get_gradients(θ_vals_unvec)
        self._logp_dzθlogp_untransθ, self._dlogprior_d2logprior_untransθ = get_gradients(θ_unvec_untrans)

    def standardize_θ(self, θ):
        is_scalar_θ = len(self.θ_RVs) == 1 and aesara.function([], self.θ_RVs[0].size)() == 1
        if isinstance(θ, Number) and is_scalar_θ:
            return [θ]
        elif isinstance(θ, dict) and set(θ.keys()) == set(v.name for v in self.θ_RVs):
            return θ
        else:
            raise Exception(
                "θ should be " + 
                ("a number or " if is_scalar_θ else "") +
                "a dict with keys: " + 
                ", ".join([var.name for var in self.θ_RVs])
            )

    def transform_θ(self, θ_dict):
        return self.unravel_θ(self._transform_θ(self.ravel_θ(θ_dict)))

    def inv_transform_θ(self, θ_dict):
        return self.unravel_θ(self._inv_transform_θ(self.ravel_θ(θ_dict)))

    def has_θ_transform(self):
        return self._has_θ_transform

    def val_gradz_gradθ_logLike(self, x, z_vec, θ_dict, transformed_θ):
        _logp_dzθlogp = self._logp_dzθlogp_transθ if transformed_θ else self._logp_dzθlogp_untransθ
        logLike, gradz_logLike, gradθ_logLike = _logp_dzθlogp(*x, z_vec, self.ravel_θ(θ_dict))
        return logLike, gradz_logLike, self._unravel_θ(gradθ_logLike)

    def sample_x_z(self, rng: int, θ_dict):
        reseed_rngs(self.rng_nodes, rng)
        *x, z = self._sample_x_z(self.ravel_θ(θ_dict))
        return (x, z)

    def gradθ_hessθ_logPrior(self, θ_dict, transformed_θ):
        _dlogprior_d2logprior = self._dlogprior_d2logprior_transθ if transformed_θ else self._dlogprior_d2logprior_untransθ
        return _dlogprior_d2logprior(self.ravel_θ(θ_dict))

    def _ravel_unravel_tensors(self, tensors, name, is_RV=True):
        tensors_raveled = at.vector(name=name)
        if is_RV:
            shapes = [aesara.function([], RV.shape)() for RV in tensors]
            sizes  = [aesara.function([], RV.size)() for RV in tensors]
        else:
            test_point = self.model.initial_point()
            shapes = [test_point[v.name].shape for v in tensors]
            sizes  = [test_point[v.name].size  for v in tensors]
        split_point = np.cumsum([0] + sizes)
        tensors_unraveled = []
        for (i, shape) in enumerate(shapes):
            tensors_unraveled.append(at.reshape(tensors_raveled[split_point[i]:split_point[i+1]], shape))
        return tensors_raveled, tensors_unraveled

    def _split_rng(self, rng: np.random.SeedSequence, N):
        return copy(rng).generate_state(N)


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


def cat_flatten(tensors):
    return at.concatenate([at.flatten(t) for t in tensors], axis=0)
