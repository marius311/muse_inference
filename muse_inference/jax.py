
from functools import partial

import jax
from jax.scipy.optimize import minimize
from jax.numpy import concatenate, atleast_1d, atleast_2d
from jax.flatten_util import ravel_pytree

from .muse_inference import MuseProblem, ScoreAndMAP


class JaxMuseProblem(MuseProblem):

    def __init__(self):
        super().__init__()
        self.np = jax.numpy

    def logLike(self, x, z, θ):
        raise NotImplementedError()

    def logPrior(self, θ):
        raise NotImplementedError()

    def logLike_and_gradzθ_logLike(self, x, z, θ, transformed_θ=None):
        logLike, (gradz_logLike, gradθ_logLike) = jax.value_and_grad(self.logLike, argnums=(1, 2))(x, z, θ)
        return (logLike, gradz_logLike, gradθ_logLike)

    def gradθ_logLike_at_zMAP(
        self, 
        x, 
        z_guess, 
        θ, 
        method = None,
        options = dict(),
        z_tol = None,
        θ_tol = None,
    ):

        if z_tol is not None:
            options = dict(gtol=z_tol, **options)
        if method is None:
            method = "l-bfgs-experimental-do-not-rely-on-this"

        ravel, unravel = self._ravel_unravel(z_guess)
        
        soln = minimize(
            lambda z_vec: -self.logLike(x, unravel(z_vec), θ), 
            ravel(z_guess), 
            method = method,
            options = options
        )

        zMAP = unravel(soln.x)

        gradθ = self.logLike_and_gradzθ_logLike(x, zMAP, θ)[2]

        return ScoreAndMAP(gradθ, gradθ, zMAP, soln)

    def gradθ_and_hessθ_logPrior(self, θ, transformed_θ=None):
        g = jax.grad(self.logPrior)(θ)
        H = jax.hessian(self.logPrior)(θ)
        return (g, H)

    def _ravel_unravel(self, x):
        ravel = lambda x_tree: ravel_pytree(x_tree)[0]
        unravel = ravel_pytree(x)[1]
        return (ravel, unravel)

    def _split_rng(self, key, N):
        return jax.random.split(key, N)


class JittableJaxMuseProblem(JaxMuseProblem):

    @partial(jax.jit, static_argnames=("self",))
    def logLike_and_gradzθ_logLike(self, x, z, θ, transformed_θ=None):
        return super().logLike_and_gradzθ_logLike(x, z, θ, transformed_θ=transformed_θ)

    @partial(jax.jit, static_argnames=("self",))
    def gradθ_and_hessθ_logPrior(self, θ, transformed_θ=None):
        return super().gradθ_and_hessθ_logPrior(θ, transformed_θ=transformed_θ)

    @partial(jax.jit, static_argnames=("self","method"))
    def gradθ_logLike_at_zMAP(
        self, 
        x, 
        z_guess, 
        θ, 
        method = None,
        options = dict(),
        z_tol = None,
        θ_tol = None,
    ):
        return super().gradθ_logLike_at_zMAP(x, z_guess, θ, method, options, z_tol, θ_tol)
