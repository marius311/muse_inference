
from datetime import datetime, timedelta
from functools import partial

from numpy.random import SeedSequence

import jax
from jax.flatten_util import ravel_pytree
from jax.numpy import array, atleast_1d, atleast_2d, concatenate, mean
from jax.numpy.linalg import inv
from jax.scipy.optimize import minimize
from jax.scipy.sparse.linalg import cg

from .muse_inference import MuseProblem, MuseResult, ScoreAndMAP


class JaxMuseProblem(MuseProblem):

    def __init__(self):
        super().__init__()
        self.np = jax.numpy

    def logLike(self, x, z, θ):
        raise NotImplementedError()

    def logPrior(self, θ):
        raise NotImplementedError()

    def val_gradz_gradθ_logLike(self, x, z, θ, transformed_θ=None):
        logLike, (gradz_logLike, gradθ_logLike) = jax.value_and_grad(self.logLike, argnums=(1, 2))(x, z, θ)
        return (logLike, gradz_logLike, gradθ_logLike)

    def z_MAP_and_score(
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

        gradθ = self.val_gradz_gradθ_logLike(x, zMAP, θ)[2]

        return ScoreAndMAP(gradθ, gradθ, zMAP, soln)

    def gradθ_hessθ_logPrior(self, θ, transformed_θ=None):
        g = jax.grad(self.logPrior)(θ)
        H = jax.hessian(self.logPrior)(θ)
        return (g, H)

    def _ravel_unravel(self, x):
        ravel = lambda x_tree: ravel_pytree(x_tree)[0]
        unravel = ravel_pytree(x)[1]
        return (ravel, unravel)

    def _split_rng(self, key, N):
        return jax.random.split(key, N)

    def _default_rng(self):
        return jax.random.PRNGKey(SeedSequence().generate_state(1)[0])

    def get_H(self, *args, use_implicit_diff=True, **kwargs):
        if use_implicit_diff:
            return self._get_H_implicit_diff(*args, **kwargs)
        else:
            return super().get_H(*args, **kwargs)

    def _get_H_implicit_diff(
        self, 
        result = None,
        θ = None,
        method = None,
        θ_tol = None,
        z_tol = None,
        rng = None,
        nsims = 10, 
        pmap = map,
        progress = False, 
    ):

        if result is None:
            result = MuseResult()
        if rng is None:
            if result.rng is None:
                rng = self._default_rng()
            else:
                rng = result.rng
        if θ is None:
            if result.θ is None:
                raise Exception("θ or result.θ must be given.")
            else:
                θ = result.θ
        θfid = θ

        if result.ravel is None:
            (result.ravel, result.unravel) = self.ravel_θ, self.unravel_θ

        nsims_remaining = nsims - len(result.Hs)

        if nsims_remaining > 0:

            pbar = tqdm(total=nsims_remaining, desc="get_H") if progress else None
            t0 = datetime.now()
            rngs = self._split_rng(rng, nsims_remaining)
            result.Hs.extend(H for H in map(partial(self.get_Hi, θ=θfid, method=method, θ_tol=θ_tol, z_tol=z_tol), rngs) if H is not None)
            result.time += datetime.now() - t0

        result.H = mean(array(result.Hs), axis=0)
        result.finalize(self)
        return result


    def get_Hi(self, rng, *, θ, method=None, θ_tol=None, z_tol=None, progress=None):

        (x, z) = self.sample_x_z(rng, θ)
        z_MAP_guess = self.z_MAP_guess_from_truth(x, z, θ)
        z_MAP = self.z_MAP_and_score(x, z_MAP_guess, θ, method=method, θ_tol=θ_tol, z_tol=z_tol).z

        # non-implicit-diff term
        H1 = jax.grad(lambda θ1: jax.grad(lambda θ2: self.logLike(self.sample_x_z(rng, θ1)[0], z_MAP, θ2))(θ))(θ)

        # term involving dzMAP/dθ via implicit-diff (w/ conjugate-gradient linear solve)
        dFdθ = jax.jacfwd(lambda θ1: jax.grad(lambda z: self.logLike(self.sample_x_z(rng, θ1)[0], z, θ))(z_MAP))(θ)
        inv_dFdz_dFdθ = cg(lambda x: jax.jvp(lambda z: jax.grad(lambda z: self.logLike(x, z, θ))(z), (z_MAP,), (x,))[1], dFdθ, tol=(z_tol or 1e-3))[0]
        H2 = -dFdθ.T @ inv_dFdz_dFdθ

        if progress: pbar.update()
        return H1 + H2





        


class JittableJaxMuseProblem(JaxMuseProblem):

    @partial(jax.jit, static_argnames=("self",))
    def val_gradz_gradθ_logLike(self, x, z, θ, transformed_θ=None):
        return super().val_gradz_gradθ_logLike(x, z, θ, transformed_θ=transformed_θ)

    @partial(jax.jit, static_argnames=("self",))
    def gradθ_hessθ_logPrior(self, θ, transformed_θ=None):
        return super().gradθ_hessθ_logPrior(θ, transformed_θ=transformed_θ)

    @partial(jax.jit, static_argnames=("self","method"))
    def z_MAP_and_score(
        self, 
        x, 
        z_guess, 
        θ, 
        method = None,
        options = dict(),
        z_tol = None,
        θ_tol = None,
    ):
        return super().z_MAP_and_score(x, z_guess, θ, method, options, z_tol, θ_tol)

    @partial(jax.jit, static_argnames=("self", "method", "progress"))
    def get_Hi(self, rng, *, θ, method=None, θ_tol=None, z_tol=None, progress=None):
        return super().get_Hi(rng, θ=θ, method=method, θ_tol=θ_tol, z_tol=z_tol, progress=progress)
