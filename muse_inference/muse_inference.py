
__all__ = ["MuseProblem", "MuseResult", "ScoreAndMAP"]

from collections import namedtuple
from copy import copy
from datetime import datetime, timedelta
from functools import partial
from numbers import Number
from time import thread_time

import numpy as np
import scipy as sp
from numpy.random import SeedSequence, default_rng
from scipy.optimize import minimize
from tqdm import tqdm

from .util import pjacobian


class MuseResult():

    def __init__(self):
        self.θ = None
        self.H = None
        self.J = None
        self.Σ_inv = None
        self.Σ = None
        self.H_prior = None
        self.dist = None
        self.history = []
        self.s_MAP_sims = []
        self.z_MAP_sims = []
        self.Hs = []
        self.rng = None
        self.ravel = None
        self.unravel = None
        self.time = timedelta(0)

    def finalize(self, prob):
        if self.J is not None and self.H is not None and self.θ is not None:

            Nθ = len(self.ravel(self.θ))

            self.H_prior = -self.ravel(prob.gradθ_hessθ_logPrior(self.θ, transformed_θ=False)[1]).reshape(Nθ,Nθ)
            self.Σ_inv = self.H.T @ np.linalg.pinv(self.J) @ self.H + self.H_prior
            self.Σ = np.linalg.pinv(self.Σ_inv)
            if self.θ is not None:
                if Nθ == 1:
                    self.dist = sp.stats.norm(self.ravel(self.θ), np.sqrt(self.Σ[0,0]))
                else:
                    self.dist = sp.stats.multivariate_normal(self.ravel(self.θ), self.Σ)

    def __repr__(self):

        if self.θ is None:
            return "MuseResult()"

        σ = self.unravel(np.sqrt(np.diag(self.Σ))) if self.Σ is not None else None
        
        def _repr(θ, σ):
            if isinstance(θ, dict):
                return "{" + ', '.join(f'{k}={_repr(θ[k], σ[k] if σ is not None else None)}' for k in θ.keys()) + "}"
            elif isinstance(θ, (list, tuple)) or (isinstance(θ, np.ndarray) and θ.shape != ()):
                return "[" + ', '.join(f'{_repr(θ[i], σ[i] if σ is not None else None)}' for i in range(len(θ))) + "]"
            elif isinstance(θ, Number) or (hasattr(θ, "shape") and θ.shape == ()):
                return f"{θ:.3g}" if σ is None else f"{θ:.3g}±{σ:.3g}"
            else:
                return repr(θ)
                
        return f"MuseResult({_repr(self.θ, σ)})"



ScoreAndMAP = namedtuple("ScoreAndMAP", "s s̃ z history")

class MuseProblem():

    def __init__(self):
        self.x = None
        self._ravel_θ = None
        self._ravel_z = None
        self._unravel_θ = None
        self._unravel_z = None
        self.np = np

    def sample_x_z(self, θ):
        raise NotImplementedError()

    def val_gradz_gradθ_logLike(self, x, z, θ, transformed_θ=None):
        raise NotImplementedError()

    def set_x(self, x):
        self.x = x

    def standardize_θ(self, θ):
        return θ

    def transform_θ(self, θ):
        return θ

    def inv_transform_θ(self, θ):
        return θ

    def has_θ_transform(self):
        return False

    def z_MAP_guess_from_truth(self, x, z, θ):
        return self.unravel_z(0 * self.ravel_z(z))

    def gradθ_hessθ_logPrior(self, θ, transformed_θ=None):
        return (0,0)

    def _split_rng(self, rng, N):
        return [default_rng(s) for s in copy(rng).spawn(N)]

    def _default_rng(self):
        return SeedSequence()

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

        # this function iteratively maximizes over z given fixed θ
        # until the θ-gradient at the z-solution converges

        if z_tol is not None:
            options = dict(gtol=z_tol, **options)
        if method is None:
            method = 'L-BFGS-B'

        np = self.np
                
        last_gradθ_logLike = None
        gradθ_logLikes = []
        terminate = False

        θ̃ = self.transform_θ(θ)

        # compute joint (z-θ)-gradient, save θ part and return z part
        # to solver. we also use a hacky way to terminate early because
        # not all minimizers support early termination via callback
        def objective(z_vec):
            nonlocal last_gradθ_logLike
            if terminate:
                return (0, 0*z_guess)
            else:
                logLike, gradz_logLike, last_gradθ_logLike = self.val_gradz_gradθ_logLike(x, self.unravel_z(z_vec), θ̃, transformed_θ=True)
                return (-logLike, -self.ravel_z(gradz_logLike))
        
        # check if θ-gradient is converged
        def callback(z_vec, *args):
            nonlocal terminate
            gradθ_logLikes.append(self.ravel_θ(last_gradθ_logLike))
            if θ_tol is not None and len(gradθ_logLikes) >= 2:
                Δgradθ = gradθ_logLikes[-1] - gradθ_logLikes[-2]
                if all(np.abs(Δgradθ) < θ_tol):
                    terminate = True
        
        # run optimization
        soln = minimize(objective, self.ravel_z(z_guess), method=method, jac=True, callback=callback, options=options)

        # save some debug info
        soln.gradθ_logLikes = gradθ_logLikes
        soln.success_gradθ = terminate
        soln.convergence_type = "gradθ stable" if soln.success_gradθ else "gradz tolerance" if soln.success else "not converged"
        soln.θ_tol = θ_tol

        z = self.unravel_z(soln.x)
        s̃ = self.val_gradz_gradθ_logLike(x, z, θ̃, transformed_θ=True)[2]
        s = self.val_gradz_gradθ_logLike(x, z, θ, transformed_θ=False)[2] if self.has_θ_transform() else s̃
        history = soln
        return ScoreAndMAP(s, s̃, z, history)

    def ravel_θ(self, θ):
        if self._ravel_θ is None:
            self._ravel_θ, self._unravel_θ = self._ravel_unravel(θ)
        return self._ravel_θ(θ)

    def unravel_θ(self, θ):
        return self._unravel_θ(θ)

    def ravel_z(self, z):
        if self._ravel_z is None:
            self._ravel_z, self._unravel_z = self._ravel_unravel(z)
        return self._ravel_z(z)

    def unravel_z(self, z):
        return self._unravel_z(z)

    def _ravel_unravel(self, x):
        np = self.np
        if isinstance(x, (tuple,list)):
            i = 0
            slices_shapes = []
            for elem in x:
                if isinstance(elem, Number):
                    slices_shapes.append((i, None))
                    i += 1
                else:
                    slices_shapes.append((slice(i,i+elem.size), elem.shape))
                    i += elem.size
            ravel = lambda tup: np.concatenate(tup, axis=None)
            unravel = lambda vec: tuple(vec[sl] if shape is None else vec[sl].reshape(shape) for (sl,shape) in slices_shapes)
        elif isinstance(x, Number):
            ravel = lambda val: np.array([val])
            unravel = lambda vec: vec.item()
        elif isinstance(x, dict):
            keys = x.keys()
            ravel_to_tup = lambda dct: tuple(dct[k] for k in keys) if isinstance(dct, dict) else dct
            unravel_tup = lambda tup: {k: v for (k, v) in zip(keys, tup)}
            ravel_tup, unravel_to_tup = self._ravel_unravel(ravel_to_tup(x))
            ravel = lambda dct: ravel_tup(ravel_to_tup(dct))
            unravel = lambda vec: unravel_tup(unravel_to_tup(vec))
        else:
            ravel = unravel = lambda z: z
        return (ravel, unravel)

    def solve(
        self,
        θ_start = None,
        result = None,
        rng = None,
        z0 = 0,
        maxsteps = 50,
        θ_rtol = 1e-2,
        z_tol = None,
        θ_tol_initial = None,
        method = None,
        nsims = 100,
        α = 0.7,
        progress = False,
        pmap = map,
        regularize = lambda θ: θ,
        H_inv_like = None,
        H_inv_update = "sims",
        broyden_memory = -1,
        checkpoint_filename = None,
        get_covariance = True,
        save_MAP_history = False
    ):

        np = self.np

        if result is None:
            result = MuseResult()
        if rng is None:
            if result.rng is None:
                rng = self._default_rng()
            else:
                rng = result.rng
        result.rng = rng

        θ_tol = θ_tol_initial

        if result.ravel is None:
            (result.ravel, result.unravel) = self.ravel_θ, self.unravel_θ

        MAP_history_dat = MAP_history_sims = None
        θunreg = θ = self.standardize_θ(result.θ if result.θ is not None else θ_start)
        θ̃unreg = θ̃ = self.transform_θ(θ)

        if z0 in ["prior", 0]:
            (_, z) = self.sample_x_z(self._split_rng(rng,1)[0], θ)
            if z0 == "prior":
                z0 = z
            else:
                z0 = self.unravel_z(0 * self.ravel_z(z))

        Nθ = len(self.ravel_θ(θ̃))
        
        xz_sims = [self.sample_x_z(_rng, θ) for _rng in self._split_rng(rng, nsims)]
        xs = [self.x] + [x for (x,_) in xz_sims]
        ẑs = [z0]     + [self.z_MAP_guess_from_truth(x, z, θ) for (x, z) in xz_sims]

        pbar = tqdm(total=(maxsteps-len(result.history))*(nsims+1), desc="MUSE") if progress else None

        try:
            
            for i in range((len(result.history)+1), maxsteps+1):
                
                t0 = datetime.now()

                if i > 1:
                    xs = [self.x] + [self.sample_x_z(_rng, θ)[0] for _rng in self._split_rng(rng,nsims)]
                    θ_tol = np.sqrt(-np.diag(result.history[-1]["H̃_inv_post"])) * θ_rtol

                if i > 2:
                    Δθ̃ = self.ravel_θ(result.history[-1]["θ̃"]) - self.ravel_θ(result.history[-2]["θ̃"])
                    if np.sqrt(-np.inner(Δθ̃, np.inner(np.linalg.pinv(result.history[-1]["H̃_inv_post"]), Δθ̃))) < θ_rtol:
                        break

                # MUSE gradient
                def get_MAPs(x_z):
                    x, ẑ_prev = x_z
                    result = self.z_MAP_and_score(x, ẑ_prev, θ, method=method, z_tol=z_tol, θ_tol=θ_tol)
                    if progress: pbar.update()
                    return result

                MAPs = list(pmap(get_MAPs, zip(xs, ẑs)))

                ẑs = [MAP.z for MAP in MAPs]
                if save_MAP_history:
                    MAP_history_dat, *MAP_history_sims = [MAP.history for MAP in MAPs]
                s_MAP_dat, *s_MAP_sims = [MAP.s for MAP in MAPs]
                s̃_MAP_dat, *s̃_MAP_sims = [MAP.s̃ for MAP in MAPs]
                s̃_MUSE = self.unravel_θ(self.ravel_θ(s̃_MAP_dat) - np.nanmean(np.stack(list(map(self.ravel_θ, s̃_MAP_sims))), axis=0))
                s̃_prior, H̃_prior = self.gradθ_hessθ_logPrior(θ̃, transformed_θ=True)
                s̃_post = self.unravel_θ(self.ravel_θ(s̃_MUSE) + self.ravel_θ(s̃_prior))

                H̃_inv_like_sims = np.diag(-1 / np.nanvar(np.stack(list(map(self.ravel_θ, s̃_MAP_sims))), axis=0))
                H̃_inv_post = np.linalg.pinv(np.linalg.pinv(H̃_inv_like_sims) + self.ravel_θ(H̃_prior).reshape(Nθ,Nθ))
                
                t = datetime.now() - t0
                result.time += t

                result.history.append({
                    "t":t, "θ̃":θ̃, "θ̃unreg":θ̃unreg, "θ":θ, "θunreg":θunreg,
                    "s_MAP_dat": s_MAP_dat, "s_MAP_sims": s_MAP_sims,
                    "s̃_MAP_dat": s̃_MAP_dat, "s̃_MAP_sims": s̃_MAP_sims, 
                    "s̃_MUSE": s̃_MUSE,
                    "s̃_prior": s̃_prior, "s̃_post": s̃_post, 
                    "H̃_inv_post": H̃_inv_post, "H̃_prior": H̃_prior, 
                    "H̃_inv_like_sims": H̃_inv_like_sims,
                    "θ_tol": θ_tol,
                    "MAP_history_dat": MAP_history_dat, 
                    "MAP_history_sims": MAP_history_sims,
                })

                θ̃unreg = self.unravel_θ(self.ravel_θ(θ̃) - α * (np.inner(H̃_inv_post, self.ravel_θ(s̃_post))))
                θunreg = self.inv_transform_θ(θ̃unreg)
                θ̃ = regularize(θ̃unreg)
                θ = self.inv_transform_θ(θ̃)

            if progress: pbar.update(pbar.total - pbar.n)
        
        finally:
            if progress: pbar.close()

        result.θ = θunreg
        result.s_MAP_sims = result.history[-1]["s_MAP_sims"]
        _, *result.z_MAP_sims = ẑs

        if get_covariance:
            self.get_J(
                result=result, nsims=nsims, 
                rng=rng, θ_tol=θ_tol, z_tol=z_tol,
                pmap=pmap, progress=progress,
                method=method,
            )
            self.get_H(
                result=result, nsims=max(1,nsims//10), 
                rng=rng, θ_tol=θ_tol, z_tol=z_tol,
                pmap=pmap, progress=progress,
                method=method,
            )

        return result


    def get_J(
        self,
        result = None,
        θ = None,
        θ_tol = None,
        z_tol = None,
        method = None,
        rng = None,
        nsims = 100, 
        pmap = map,
        progress = False, 
        skip_errors = False,
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
                raise Exception("θ0 or result.θ must be given.")
            else:
                θ = result.θ

        if result.ravel is None:
            (result.ravel, result.unravel) = self.ravel_θ, self.unravel_θ
        
        nsims_remaining = nsims - len(result.s_MAP_sims)

        if nsims_remaining > 0:

            pbar = tqdm(total=nsims_remaining, desc="get_J") if progress else None
            t0 = datetime.now()

            def get_s_MAP(rng):
                try:
                    (x, z) = self.sample_x_z(rng, θ)
                    z_MAP_guess = self.z_MAP_guess_from_truth(x, z, θ)
                    return self.z_MAP_and_score(x, z_MAP_guess, θ, method=method, θ_tol=θ_tol, z_tol=z_tol).s
                except Exception:
                    if skip_errors:
                        return None
                    else:
                        raise
                finally:
                    if progress: 
                        pbar.update()

            rngs = self._split_rng(rng, nsims_remaining)
            result.s_MAP_sims.extend(s for s in pmap(get_s_MAP, rngs) if s is not None)

            result.time += datetime.now() - t0

        result.J = np.atleast_2d(np.cov(np.stack(list(map(self.ravel_θ, result.s_MAP_sims))), rowvar=False))
        result.finalize(self)
        return result


    def _get_H_i(self, rng, z_MAP_guess_fid, *, θ, method=None, θ_tol=None, z_tol=None, step=None, skip_errors=False):

        # for each sim, do one fit at fiducial which we'll
        # reuse as a starting point when fudging θ by +/-ϵ
        θfid = θ
        (x, z) = self.sample_x_z(rng, θfid)
        if z_MAP_guess_fid is None:
            z_MAP_guess = self.z_MAP_guess_from_truth(x, z, θfid)
            z_MAP_guess_fid = self.z_MAP_and_score(x, z_MAP_guess, θfid, method=method, θ_tol=θ_tol, z_tol=z_tol).z

        def get_s_MAP(θvec):
            θ = self.unravel_θ(θvec)
            (x, _) = self.sample_x_z(copy(rng), θ)
            return self.ravel_θ(self.z_MAP_and_score(x, z_MAP_guess_fid, θfid, method=method, θ_tol=θ_tol, z_tol=z_tol).s)

        try:
            return pjacobian(get_s_MAP, self.ravel_θ(θfid), step)
        except Exception:
            if skip_errors:
                return None
            else:
                raise


    def get_H(
        self,
        result = None,
        θ = None,
        step = None,
        method = None,
        θ_tol = None,
        z_tol = None,
        rng = None,
        nsims = 10, 
        pmap = map,
        progress = False, 
        skip_errors = False,
        use_median = False
    ):

        np = self.np

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

        if result.ravel is None:
            (result.ravel, result.unravel) = self.ravel_θ, self.unravel_θ

        nsims_remaining = nsims - len(result.Hs)

        if nsims_remaining > 0:

            # default to finite difference step size of 0.1σ with σ roughly
            # estimated from s_MAP_sims sims, if we have them
            if step is None:
                if len(result.s_MAP_sims) > 0:
                    step = 0.1 / np.std(np.stack(list(map(self.ravel_θ, result.s_MAP_sims))), axis=0)
                else:
                    step = 1e-5

            Nθ = len(self.ravel_θ(θ))
            pbar = partial(tqdm, total=nsims_remaining, desc="get_H", disable=(not progress))

            t0 = datetime.now()
            rngs = self._split_rng(rng, nsims)[-nsims_remaining:]
            z_MAP_sims = (result.z_MAP_sims + [None]*(max(0, nsims - len(result.z_MAP_sims))))[-nsims_remaining:]
            _get_H_i = partial(self._get_H_i, θ=θ, method=method, θ_tol=θ_tol, z_tol=z_tol, step=step, skip_errors=skip_errors)
            result.Hs.extend(H for H in pbar(map(lambda args: _get_H_i(*args), zip(rngs, z_MAP_sims)) )if H is not None)
            result.time += datetime.now() - t0

        avg = np.median if use_median else np.mean
        result.H = avg(np.array(result.Hs), axis=0)
        result.finalize(self)
        return result
