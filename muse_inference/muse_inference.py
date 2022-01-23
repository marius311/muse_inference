
__all__ = ["MuseProblem", "MuseResult"]

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
        self.gs = []
        self.Hs = []
        self.rng = None
        self.time = timedelta(0)

    def finalize(self, prob):
        if self.J is not None and self.H is not None:
            H_prior = prob.gradθ_and_hessθ_logPrior(self.θ)[1]
            self.Σ_inv = self.H.T @ np.linalg.inv(self.J) @ self.H + H_prior
            self.Σ = np.linalg.inv(self.Σ_inv)
            if self.θ is not None:
                if isinstance(self.θ, Number):
                    self.dist = sp.stats.norm(self.θ, np.sqrt(self.Σ[0,0]))
                else:
                    self.dist = sp.stats.multivariate_normal(self.θ, self.Σ)



class MuseProblem():

    def __init__(self):
        self.x = None
        self.np = np

    def sample_x_z(self, θ):
        raise NotImplementedError()

    def gradθ_logLike(self, x, z, θ):
        raise NotImplementedError()

    def logLike_and_gradz_logLike(self, x, z, θ):
        raise NotImplementedError()

    def gradθ_and_hessθ_logPrior(self, θ):
        return (0,0)

    def zMAP_at_θ(self, x, z0, θ, gradz_logLike_atol=None):
        ravel, unravel = self._ravel_unravel(z0)
        def objective(z_vec):
            logLike, gradz_logLike = self.logLike_and_gradz_logLike(x, unravel(z_vec), θ)
            return (-logLike, -ravel(gradz_logLike))
        soln = minimize(objective, ravel(z0), method='L-BFGS-B', jac=True, options=dict(gtol=gradz_logLike_atol))
        return (unravel(soln.x), soln)

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
        θ_rtol = 1e-5,
        gradz_logLike_atol = 1e-2,
        nsims = 100,
        α = 0.7,
        progress = False,
        pmap = map,
        regularize = lambda θ: θ,
        H_inv_like = None,
        H_inv_update = "sims",
        broyden_memory = -1,
        checkpoint_filename = None,
        get_covariance = False,
        save_zMAP_history = False
    ):

        np = self.np

        if result is None:
            result = MuseResult()
        if rng is None:
            rng = SeedSequence()
        if z0 is None:
            z0 = self.sample_x_z(self._split_rng(rng,1)[0], θ_start)[1]

        zMAP_history_dat = zMAP_history_sims = None
        θunreg = θ = θ_start

        is_scalar_θ = isinstance(θ, Number)
        ravel, unravel = self._ravel_unravel(θ)
        Nθ = 1 if is_scalar_θ else len(ravel(θ))
        
        xz_sims = [self.sample_x_z(_rng, θ) for _rng in self._split_rng(rng, nsims)]
        xs    = [self.x] + [x for (x,_) in xz_sims]
        zMAPs = [z0]     + [z for (_,z) in xz_sims]

        pbar = tqdm(total=(maxsteps-len(result.history))*(nsims+1), desc="MUSE") if progress else None

        try:
            
            for i in range((len(result.history)+1), maxsteps+1):
                
                t0 = datetime.now()

                if i > 1:
                    xs = [self.x] + [self.sample_x_z(_rng, θ)[0] for _rng in self._split_rng(rng,nsims)]

                if i > 2:
                    Δθ = ravel(result.history[-1]["θ"]) - ravel(result.history[-2]["θ"])
                    if np.sqrt(-np.inner(Δθ, np.inner(H_inv_post, Δθ))) < θ_rtol:
                        break

                # MUSE gradient
                def get_MAPs(args):
                    x, zMAP_prev = args
                    (zMAP, history) = self.zMAP_at_θ(x, zMAP_prev, θ, gradz_logLike_atol=gradz_logLike_atol)
                    g = ravel(self.gradθ_logLike(x, zMAP, θ))
                    if progress: pbar.update()
                    return (g, zMAP, history)

                g_zMAPs = list(pmap(get_MAPs, zip(xs, zMAPs)))

                zMAPs = [zMAP for (_,zMAP,_) in g_zMAPs]
                if save_zMAP_history: zMAP_history_dat, *zMAP_history_sims = [history for (_,_,history) in g_zMAPs]
                g_like_dat, *g_like_sims = [g for (g,_,_) in g_zMAPs]
                g_like = g_like_dat - np.mean(np.stack(g_like_sims), axis=0)
                g_prior, H_prior = self.gradθ_and_hessθ_logPrior(θ)
                g_post = g_like + ravel(g_prior)

                h_inv_like_sims = -1 / np.var(np.stack(g_like_sims), axis=0)
                if is_scalar_θ:
                    H_inv_post = 1 / (1 / h_inv_like_sims + H_prior)
                else:
                    H_inv_post = np.linalg.inv(np.linalg.inv(np.diag(h_inv_like_sims)) + ravel(H_prior).reshape(Nθ,Nθ))
                
                t = datetime.now() - t0

                result.history.append({
                    "θ":θ, "θunreg":θunreg, "t":t, "g_like_dat": g_like_dat,
                    "g_like_sims": g_like_sims, "g_like": g_like, "g_prior": g_prior,
                    "g_post": g_post, "H_inv_post": H_inv_post, "H_prior": H_prior,
                    "zMAP_history_dat": zMAP_history_dat, "zMAP_history_sims": zMAP_history_sims,
                })


                θunreg = unravel(ravel(θ) - α * (np.inner(H_inv_post, g_post)))
                θ = regularize(θunreg)

            if progress: pbar.update(pbar.total - pbar.n)
        
        finally:
            if progress: pbar.close()

        result.θ = θunreg
        result.gs = g_like_sims
        result.time = sum((h["t"] for h in result.history), start=result.time)

        if get_covariance:
            self.get_J(result=result, gradz_logLike_atol=gradz_logLike_atol, pmap=pmap, progress=progress)
            self.get_H(result=result, gradz_logLike_atol=gradz_logLike_atol, pmap=pmap, progress=progress)

        return result


    def get_J(
        self,
        result = None,
        θ = None,
        gradz_logLike_atol = 1e-2,
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
        if θ is None:
            θ = result.θ

        nsims_remaining = nsims - len(result.gs)

        if nsims_remaining > 0:

            pbar = tqdm(total=nsims_remaining, desc="get_J") if progress else None
            t0 = datetime.now()

            ravel, unravel = self._ravel_unravel(θ)

            xz_sims = [self.sample_x_z(_rng, θ) for _rng in self._split_rng(rng, nsims)]

            def get_g(x_z):
                try:
                    x, z = x_z
                    zMAP = self.zMAP_at_θ(x, z, θ, gradz_logLike_atol=gradz_logLike_atol)[0]
                    g = ravel(self.gradθ_logLike(x, zMAP, θ))
                    if progress: pbar.update()
                    return g
                except Exception:
                    if skip_errors:
                        return None
                    else:
                        raise

            result.gs.extend(g for g in pmap(get_g, xz_sims) if g is not None)

            result.time += datetime.now() - t0

        result.J = np.atleast_2d(np.cov(result.gs, rowvar=False))
        result.finalize(self)
        return result


    def get_H(
        self,
        result = None,
        θ = None,
        step = 0.01,
        gradz_logLike_atol = 1e-2,
        rng = None,
        nsims = 10, 
        pmap = map,
        progress = False, 
        skip_errors = False,
    ):

        if result is None:
            result = MuseResult()
        if rng is None:
            rng = SeedSequence()
        if θ is None:
            θ = result.θ

        is_scalar_θ = isinstance(θ, Number)
        ravel, unravel = self._ravel_unravel(θ)
        Nθ = 1 if is_scalar_θ else len(ravel(θ))
        
        nsims_remaining = nsims - len(result.Hs)

        if (nsims_remaining > 0):
                
            pbar = tqdm(total=nsims_remaining*(2*Nθ+1), desc="get_H") if progress else None
            t0 = datetime.now()

            ravel, unravel = self._ravel_unravel(θ)

            # generate simulations
            xz_sims = [self.sample_x_z(_rng, θ) for _rng in self._split_rng(rng, nsims)]

            # initial fit at fiducial, used at starting points for finite difference below
            def get_zMAP(x_z):
                x, z = x_z
                zMAP = self.zMAP_at_θ(x, z, θ, gradz_logLike_atol=gradz_logLike_atol)[0]
                if progress: pbar.update()
                return zMAP
            zMAPs_fid = pmap(get_zMAP, xz_sims)

            # finite difference Jacobian
            # pmap_sims, pmap_jac = (pmap_over == :jac || (pmap_over == :auto && length(θ₀) > nsims_remaining)) ? (_map, pmap) : (pmap, _map)
            def getH(x_zMAPfid_rng):
                x, zMAPfid, rng = x_zMAPfid_rng
                def get_sMAP(θ_vec):
                    θ = unravel(θ_vec)
                    x = self.sample_x_z(copy(rng), θ)[0]
                    zMAP = self.zMAP_at_θ(x, zMAPfid, θ, gradz_logLike_atol=gradz_logLike_atol)[0]
                    return ravel(self.gradθ_logLike(x, zMAP, θ))
                try:
                    return pjacobian(get_sMAP, ravel(θ), step, pbar=pbar)
                except Exception:
                    if skip_errors:
                        return None
                    else:
                        raise

            x_zMAPfid_rngs = zip([x for (x,_) in xz_sims], zMAPs_fid, self._split_rng(rng, nsims))
            result.Hs.extend(H for H in map(getH, x_zMAPfid_rngs) if H is not None)
            
            result.time += datetime.now() - t0

        result.H = np.mean(result.Hs, axis=0)
        result.finalize(self)
        return result
