
from functools import partial

import jax
from jax.scipy.optimize import minimize
from jax.numpy import concatenate, atleast_1d, atleast_2d
from jax.flatten_util import ravel_pytree

from . import MuseProblem


class JaxMuseProblem(MuseProblem):

    def __init__(self):
        super().__init__()
        self.np = jax.numpy

    def logLike(self, x, z, θ):
        raise NotImplementedError()

    def logPrior(self, θ):
        raise NotImplementedError()

    def gradθ_logLike(self, x, z, θ):
        return jax.grad(lambda θ: self.logLike(x, z, θ))(θ)

    def logLike_and_gradz_logLike(self, x, z, θ):
        logLike = self.logLike(x, z, θ)
        gradz_logLike = jax.grad(lambda z: self.logLike(x, z, θ))(z)
        return (logLike, gradz_logLike)

    def zMAP_at_θ(self, x, z0, θ, gradz_logLike_atol=None):
        ravel, unravel = self.ravel_unravel(z0)
        soln = minimize(
            lambda z_vec: -self.logLike(x, unravel(z_vec), θ), ravel(z0), 
            method="l-bfgs-experimental-do-not-rely-on-this", 
            options=dict(gtol=gradz_logLike_atol),
        )
        return (unravel(soln.x), soln)

    def gradθ_and_hessθ_logPrior(self, θ):
        g = jax.grad(self.logPrior)(θ)
        H = jax.hessian(self.logPrior)(θ)
        return (g,H)

    def ravel_unravel(self, x):
        ravel = lambda x_tree: ravel_pytree(x_tree)[0]
        unravel = ravel_pytree(x)[1]
        return (ravel, unravel)


class JittedJaxMuseProblem(JaxMuseProblem):

    @partial(jax.jit, static_argnums=(0,))
    def gradθ_logLike(self, x, z, θ):
        return super().gradθ_logLike(x, z, θ)

    @partial(jax.jit, static_argnums=(0,))
    def logLike_and_gradz_logLike(self, x, z, θ):
        return super().logLike_and_gradz_logLike(x, z, θ)

    @partial(jax.jit, static_argnums=(0,))
    def zMAP_at_θ(self, x, z0, θ, gradz_logLike_atol=None):
        return super().zMAP_at_θ(x, z0, θ, gradz_logLike_atol)

    @partial(jax.jit, static_argnums=(0,))
    def gradθ_and_hessθ_logPrior(self, θ):
        return super().gradθ_and_hessθ_logPrior(θ)
