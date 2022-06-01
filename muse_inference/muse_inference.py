
__all__ = ["MuseProblem", "MuseResult"]

from collections import namedtuple
from copy import copy
from datetime import datetime, timedelta
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
        self.dist = None
        self.history = []
        self.s_MAP_sims = []
        self.Hs = []
        self.rng = None
        self.time = timedelta(0)

    def finalize(self, prob):
        if self.J is not None and self.H is not None and self.θ is not None:

            is_scalar_θ = isinstance(self.θ, Number)
            ravel, unravel = prob._ravel_unravel(self.θ)
            Nθ = 1 if is_scalar_θ else len(ravel(self.θ))

            H_prior = ravel(prob.gradθ_and_hessθ_logPrior(self.θ, transformed_θ=False)[1]).reshape(Nθ,Nθ)
            self.Σ_inv = self.H.T @ np.linalg.inv(self.J) @ self.H - H_prior
            self.Σ = np.linalg.inv(self.Σ_inv)
            if self.θ is not None:
                if isinstance(self.θ, Number):
                    self.dist = sp.stats.norm(self.θ, np.sqrt(self.Σ[0,0]))
                else:
                    self.dist = sp.stats.multivariate_normal(self.θ, self.Σ)



XZSample = namedtuple("XZSample", "x z")
ScoreAndMAP = namedtuple("ScoreAndMAP", "s s̃ z history")

class MuseProblem():

    def __init__(self):
        self.x = None
        self.np = np

    def standardizeθ(self, θ):
        return θ

    def transform_θ(self, θ):
        return θ

    def inv_transform_θ(self, θ):
        return θ

    def has_θ_transform(self):
        return False

    def sample_x_z(self, θ):
        raise NotImplementedError()

    def gradθ_and_hessθ_logPrior(self, θ, transformed_θ=False):
        return (0,0)

    def logLike_and_gradzθ_logLike(self, x, z, θ):
        raise NotImplementedError()

    def gradθ_logLike_at_zMAP(
        self, 
        x, 
        z_guess, 
        θ, 
        method = 'L-BFGS-B', 
        options = dict(),
        z_tol = 1e-5,
        θ_tol = 1e-2,
    ):

        np = self.np

        # this function iteratively maximizes over z given fixed θ
        # until the θ-gradient at the z-solution converges
        
        last_ztol = last_gradθ_logLike = None
        gradθ_logLikes = []
        ztol_history = []
        terminate = False

        θ̃ = self.transform_θ(θ)

        # compute joint (z-θ)-gradient, save θ part and return z part
        # to solver. we also use a hacky way to terminate early because
        # not all minimizers support early termination via callback
        def objective(z_vec):
            nonlocal last_gradθ_logLike, last_ztol
            if terminate:
                return (0, 0*z_guess)
            else:
                logLike, gradz_logLike, last_gradθ_logLike = self.logLike_and_gradzθ_logLike(x, z_vec, θ̃, transformed_θ=True)
                last_ztol = np.max(np.abs(gradz_logLike))
                return (-logLike, -gradz_logLike)
        
        # check if θ-gradient is converged
        def callback(z_vec, *args):
            nonlocal terminate
            gradθ_logLikes.append(last_gradθ_logLike)
            ztol_history.append(last_ztol)
            if θ_tol is not None and len(gradθ_logLikes) >= 2:
                Δgradθ = gradθ_logLikes[-1] - gradθ_logLikes[-2]
                if all(np.abs(Δgradθ) < θ_tol):
                    terminate = True
        
        # run optimization
        soln = minimize(objective, z_guess, method=method, jac=True, callback=callback, options=dict(gtol=z_tol, **options))

        # save some debug info
        soln.gradθ_logLikes = gradθ_logLikes
        soln.ztols = ztol_history
        soln.success_gradθ = terminate
        soln.convergence_type = "gradθ stable" if soln.success_gradθ else "gradz tolerance" if soln.success else "not converged"
        soln.z_tol = z_tol
        soln.θ_tol = θ_tol

        s̃ = self.logLike_and_gradzθ_logLike(x, soln.x, θ̃, transformed_θ=True)[2]
        s = self.logLike_and_gradzθ_logLike(x, soln.x, θ, transformed_θ=False)[2] if self.has_θ_transform() else s̃
        z = soln.x
        history = soln
        return ScoreAndMAP(s, s̃, z, history)


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
        else:
            ravel = unravel = lambda z: z
        return (ravel, unravel)

    def _split_rng(self, rng: SeedSequence, N):
        return [default_rng(s) for s in copy(rng).spawn(N)]

    def solve(
        self,
        θ_start = None,
        result = None,
        rng = None,
        z0 = None,
        maxsteps = 50,
        θ_rtol = 1e-2,
        z_tol = None,
        s_MAP_tol_initial = None,
        method = "L-BFGS-B",
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
            rng = SeedSequence()
        if z0 is None:
            z0 = self.sample_x_z(self._split_rng(rng,1)[0], θ_start)[1]

        s_MAP_tol = s_MAP_tol_initial

        MAP_history_dat = MAP_history_sims = None
        θunreg = θ = result.θ if result.θ is not None else θ_start
        θ̃unreg = θ̃ = self.transform_θ(θ)

        is_scalar_θ = isinstance(θ̃, Number)
        ravel, unravel = self._ravel_unravel(θ̃)
        Nθ = 1 if is_scalar_θ else len(ravel(θ̃))
        
        xz_sims = [self.sample_x_z(_rng, θ) for _rng in self._split_rng(rng, nsims)]
        xs = [self.x] + [sim.x for sim in xz_sims]
        ẑs = [z0]     + [z0 if z0 is not None else sim.z for sim in xz_sims]

        pbar = tqdm(total=(maxsteps-len(result.history))*(nsims+1), desc="MUSE") if progress else None

        try:
            
            for i in range((len(result.history)+1), maxsteps+1):
                
                t0 = datetime.now()

                if i > 1:
                    xs = [self.x] + [self.sample_x_z(_rng, θ)[0] for _rng in self._split_rng(rng,nsims)]
                    s_MAP_tol = np.sqrt(-H̃_inv_post) * θ_rtol

                if i > 2:
                    Δθ̃ = ravel(result.history[-1]["θ̃"]) - ravel(result.history[-2]["θ̃"])
                    if np.sqrt(-np.inner(Δθ̃, np.inner(np.linalg.pinv(result.history[-1]["H̃_inv_post"]), Δθ̃))) < θ_rtol:
                        break

                # MUSE gradient
                def get_MAPs(x_z):
                    x, ẑ_prev = x_z
                    result = self.gradθ_logLike_at_zMAP(x, ẑ_prev, θ, method=method, z_tol=z_tol, θ_tol=s_MAP_tol)
                    if progress: pbar.update()
                    return result

                MAPs = list(pmap(get_MAPs, zip(xs, ẑs)))

                ẑs = [MAP.z for MAP in MAPs]
                if save_MAP_history:
                    MAP_history_dat, *MAP_history_sims = [MAP.history for MAP in MAPs]
                s_MAP_dat, *s_MAP_sims = [MAP.s for MAP in MAPs]
                s̃_MAP_dat, *s̃_MAP_sims = [MAP.s̃ for MAP in MAPs]
                s̃_MUSE = s̃_MAP_dat - np.mean(np.stack(s̃_MAP_sims), axis=0)
                s̃_prior, H̃_prior = self.gradθ_and_hessθ_logPrior(θ̃, transformed_θ=True)
                s̃_post = s̃_MUSE + ravel(s̃_prior)

                h̃_inv_like_sims = -1 / np.var(np.stack(s̃_MAP_sims), axis=0)
                if is_scalar_θ:
                    H̃_inv_post = 1 / (1 / h̃_inv_like_sims + H̃_prior)
                else:
                    H̃_inv_post = np.linalg.pinv(np.linalg.pinv(np.diag(h̃_inv_like_sims)) + ravel(H̃_prior).reshape(Nθ,Nθ))
                
                t = datetime.now() - t0

                result.history.append({
                    "t":t, "θ̃":θ̃, "θ̃unreg":θ̃unreg, "θ":θ, "θunreg":θunreg,
                    "s_MAP_dat": s_MAP_dat, "s_MAP_sims": s_MAP_sims,
                    "s̃_MAP_dat": s̃_MAP_dat, "s̃_MAP_sims": s̃_MAP_sims, 
                    "s̃_MUSE": s̃_MUSE,
                    "s̃_prior": s̃_prior, "s̃_post": s̃_post, 
                    "H̃_inv_post": H̃_inv_post, "H̃_prior": H̃_prior, 
                    "h̃_inv_like_sims": h̃_inv_like_sims,
                    "s_MAP_tol": s_MAP_tol,
                    "MAP_history_dat": MAP_history_dat, 
                    "MAP_history_sims": MAP_history_sims,
                })


                θ̃unreg = unravel(ravel(θ̃) - α * (np.inner(H̃_inv_post, s̃_post)))
                θunreg = self.inv_transform_θ(θ̃unreg)
                θ̃ = regularize(θ̃unreg)
                θ = self.inv_transform_θ(θ̃)

            if progress: pbar.update(pbar.total - pbar.n)
        
        finally:
            if progress: pbar.close()

        result.θ = θunreg
        result.s_MAP_sims = result.history[-1]["s_MAP_sims"]
        result.time = sum((h["t"] for h in result.history), start=result.time)

        if get_covariance:
            self.get_J(
                result=result, nsims=nsims, 
                rng=rng, s_MAP_tol=s_MAP_tol, 
                pmap=pmap, progress=progress,
                method=method,
            )
            self.get_H(
                result=result, nsims=max(1,nsims//10), 
                rng=rng, s_MAP_tol=s_MAP_tol, 
                pmap=pmap, progress=progress,
                method=method,
            )

        return result


    def get_J(
        self,
        result = None,
        θ0 = None,
        s_MAP_tol = None,
        method = "L-BFGS-B",
        rng = None,
        nsims = 100, 
        pmap = map,
        progress = False, 
        skip_errors = False,
    ):

        if result is None:
            result = MuseResult()
        if rng is None:
            rng = SeedSequence()
        if θ0 is None:
            if result.θ is None:
                raise Exception("θ0 or result.θ must be given.")
            else:
                θ0 = result.θ

        nsims_remaining = nsims - len(result.s_MAP_sims)

        if nsims_remaining > 0:

            pbar = tqdm(total=nsims_remaining, desc="get_J") if progress else None
            t0 = datetime.now()

            ravel, unravel = self._ravel_unravel(θ0)

            xz_sims = [self.sample_x_z(_rng, θ0) for _rng in self._split_rng(rng, nsims_remaining)]

            def get_g(x_z):
                try:
                    x, z = x_z
                    result = self.gradθ_logLike_at_zMAP(x, ẑ_prev, θ, method=method, θ_tol=s_MAP_tol)
                    if progress: pbar.update()
                    return result.s
                except Exception:
                    if skip_errors:
                        return None
                    else:
                        raise

            result.s_MAP_sims.extend(g for g in pmap(get_g, xz_sims) if g is not None)

            result.time += datetime.now() - t0

        result.J = np.atleast_2d(np.cov(result.s_MAP_sims, rowvar=False))
        result.finalize(self)
        return result


    def get_H(
        self,
        result = None,
        θ0 = None,
        step = None,
        method = "L-BFGS-B",
        s_MAP_tol = None,
        rng = None,
        nsims = 10, 
        pmap = map,
        progress = False, 
        skip_errors = False,
    ):

        np = self.np

        if result is None:
            result = MuseResult()
        if rng is None:
            rng = SeedSequence()
        if θ0 is None:
            if result.θ is None:
                raise Exception("θ0 or result.θ must be given.")
            else:
                θ0 = result.θ


        nsims_remaining = nsims - len(result.Hs)

        if nsims_remaining > 0:

            # default to finite difference step size of 0.1σ with σ roughly
            # estimated from s_MAP_sims sims, if we have them
            if step is None and len(result.s_MAP_sims) > 0:
                step = 0.1 / np.std(result.s_MAP_sims)
                
            is_scalar_θ = isinstance(θ0, Number)
            ravel, unravel = self._ravel_unravel(θ0)
            Nθ = 1 if is_scalar_θ else len(ravel(θ0))

            pbar = tqdm(total=nsims_remaining*(2*Nθ+1), desc="get_H") if progress else None
            t0 = datetime.now()

            # generate simulations
            xz_sims = [self.sample_x_z(_rng, θ0) for _rng in self._split_rng(rng, nsims_remaining)]

            # initial fit at fiducial, used at starting points for finite difference below
            def get_zMAP(x_z):
                x, z = x_z
                result = self.gradθ_logLike_at_zMAP(x, z, θ0, method=method, θ_tol=s_MAP_tol)
                if progress: pbar.update()
                return result.z
            zMAPs_fid = pmap(get_zMAP, xz_sims)

            # finite difference Jacobian
            # pmap_sims, pmap_jac = (pmap_over == :jac || (pmap_over == :auto && length(θ₀) > nsims_remaining)) ? (_map, pmap) : (pmap, _map)
            def _get_H(x_zMAPfid_rng):
                x, zMAPfid, rng = x_zMAPfid_rng
                
                def get_sMAP(θvec):
                    θ = unravel(θvec)
                    x = self.sample_x_z(copy(rng), θ)[0]
                    return ravel(self.gradθ_logLike_at_zMAP(x, zMAPfid, θ0, method=method, θ_tol=s_MAP_tol).s)

                try:
                    return pjacobian(get_sMAP, ravel(θ0), step, pbar=pbar)
                except Exception:
                    if skip_errors:
                        return None
                    else:
                        raise

            x_zMAPfid_rngs = zip([sim.x for sim in xz_sims], zMAPs_fid, self._split_rng(rng, nsims))
            result.Hs.extend(H for H in map(_get_H, x_zMAPfid_rngs) if H is not None)
            
            result.time += datetime.now() - t0

        result.H = np.mean(result.Hs, axis=0)
        result.finalize(self)
        return result
