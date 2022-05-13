
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
        if self.J is not None and self.H is not None and self.θ is not None:

            is_scalar_θ = isinstance(self.θ, Number)
            ravel, unravel = prob._ravel_unravel(self.θ)
            Nθ = 1 if is_scalar_θ else len(ravel(self.θ))

            H_prior = ravel(prob.gradθ_and_hessθ_logPrior(self.θ, transformed_θ=False)[1]).reshape(Nθ,Nθ)
            self.Σ_inv = self.H.T @ np.linalg.inv(self.J) @ self.H #- H_prior
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
        self.has_θ_transform = False

    def standardizeθ(self, θ):
        return θ

    def transform_θ(self, θ):
        return θ

    def inv_transform_θ(self, θ):
        return θ

    def has_θ_transform(self, θ):
        return False

    def sample_x_z(self, θ):
        raise NotImplementedError()

    def gradθ_logLike(self, x, z, θ, transformed_θ=False):
        raise NotImplementedError()

    def logLike_and_gradz_logLike(self, x, z, θ):
        raise NotImplementedError()

    def gradθ_and_hessθ_logPrior(self, θ, transformed_θ=False):
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
        θunreg  = θ  = result.θ if result.θ is not None else θ_start
        θunregʼ = θʼ = self.transform_θ(θ)

        is_scalar_θ = isinstance(θʼ, Number)
        ravel, unravel = self._ravel_unravel(θʼ)
        Nθ = 1 if is_scalar_θ else len(ravel(θʼ))
        
        xz_sims = [self.sample_x_z(_rng, θ) for _rng in self._split_rng(rng, nsims)]
        xs = [self.x] + [x for (x,_) in xz_sims]
        ẑs = [z0]     + [z for (_,z) in xz_sims]

        pbar = tqdm(total=(maxsteps-len(result.history))*(nsims+1), desc="MUSE") if progress else None

        try:
            
            for i in range((len(result.history)+1), maxsteps+1):
                
                t0 = datetime.now()

                if i > 1:
                    xs = [self.x] + [self.sample_x_z(_rng, θ)[0] for _rng in self._split_rng(rng,nsims)]

                if i > 2:
                    Δθʼ = ravel(result.history[-1]["θʼ"]) - ravel(result.history[-2]["θʼ"])
                    if np.sqrt(-np.inner(Δθʼ, np.inner(result.history[-1]["H_inv_postʼ"], Δθʼ))) < θ_rtol:
                        break

                # MUSE gradient
                def get_MAPs(args):
                    x, ẑ_prev = args
                    (ẑ, history) = self.zMAP_at_θ(x, ẑ_prev, θ, gradz_logLike_atol=gradz_logLike_atol)
                    gʼ = ravel(self.gradθ_logLike(x, ẑ, θʼ, transformed_θ=True))
                    g  = ravel(self.gradθ_logLike(x, ẑ, θ,  transformed_θ=False)) if self.has_θ_transform else gʼ
                    if progress: pbar.update()
                    return (gʼ, g, ẑ, history)

                gẑs = list(pmap(get_MAPs, zip(xs, ẑs)))

                ẑs = [zMAP for (_,_,zMAP,_) in gẑs]
                if save_zMAP_history: 
                    zMAP_history_dat, *zMAP_history_sims = [history for (_,_,history) in gẑs]
                g_like_dat,  *g_like_sims  = [g  for (g, _,  *_) in gẑs]
                g_like_datʼ, *g_like_simsʼ = [gʼ for (_, gʼ, *_) in gẑs]
                g_likeʼ = g_like_datʼ - np.mean(np.stack(g_like_simsʼ), axis=0)
                g_priorʼ, H_priorʼ = self.gradθ_and_hessθ_logPrior(θʼ, transformed_θ=True)
                g_postʼ = g_likeʼ + ravel(g_priorʼ)

                h_inv_like_simsʼ = -1 / np.var(np.stack(g_like_simsʼ), axis=0)
                if is_scalar_θ:
                    H_inv_postʼ = 1 / (1 / h_inv_like_simsʼ + H_priorʼ)
                else:
                    H_inv_postʼ = np.linalg.inv(np.linalg.inv(np.diag(h_inv_like_simsʼ)) + ravel(H_priorʼ).reshape(Nθ,Nθ))
                
                t = datetime.now() - t0

                result.history.append({
                    "θʼ":θʼ, "θunregʼ":θunregʼ, "θ":θ, "θunreg":θunreg,
                    "t":t, "g_like_datʼ": g_like_datʼ, "g_like_sims": g_like_sims, 
                    "g_like_simsʼ": g_like_simsʼ, "g_likeʼ": g_likeʼ, "g_priorʼ": g_priorʼ,
                    "g_postʼ": g_postʼ, "H_inv_postʼ": H_inv_postʼ, "H_priorʼ": H_priorʼ, 
                    "h_inv_like_simsʼ": h_inv_like_simsʼ,
                    "zMAP_history_dat": zMAP_history_dat, "zMAP_history_sims": zMAP_history_sims,
                })


                θunregʼ = unravel(ravel(θʼ) - α * (np.inner(H_inv_postʼ, g_postʼ)))
                θunreg = self.inv_transform_θ(θunregʼ)
                θʼ = regularize(θunregʼ)
                θ = self.inv_transform_θ(θʼ)

            if progress: pbar.update(pbar.total - pbar.n)
        
        finally:
            if progress: pbar.close()

        result.θ = θunreg
        result.gs = result.history[-1]["g_like_sims"]
        result.time = sum((h["t"] for h in result.history), start=result.time)

        if get_covariance:
            self.get_J(
                result=result, nsims=nsims, 
                rng=rng, gradz_logLike_atol=gradz_logLike_atol, 
                pmap=pmap, progress=progress
            )
            self.get_H(
                result=result, nsims=max(1,nsims//10), 
                rng=rng, gradz_logLike_atol=gradz_logLike_atol, 
                pmap=pmap, progress=progress
            )

        return result


    def get_sMUSE(
        self,
        θ,
        rng = None,
        z0 = None,
        nsims = 100,
        progress = False,
        gradz_logLike_atol = 1e-2,
        pmap = map
    ):

        if rng is None:
            rng = SeedSequence()
        if z0 is None:
            z0 = self.sample_x_z(self._split_rng(rng,1)[0], θ)[1]

        zMAP_history_dat = zMAP_history_sims = None

        is_scalar_θ = isinstance(θ, Number)
        ravel, unravel = self._ravel_unravel(θ)
        Nθ = 1 if is_scalar_θ else len(ravel(θ))
        
        xz_sims = [self.sample_x_z(_rng, θ) for _rng in self._split_rng(rng, nsims)]
        xs    = [self.x] + [x for (x,_) in xz_sims]
        zMAPs = [z0]     + [z for (_,z) in xz_sims]

        pbar = tqdm(total=(nsims+1), desc="get_sMUSE") if progress else None


        # MUSE gradient
        def get_MAPs(args):
            x, zMAP_prev = args
            (zMAP, history) = self.zMAP_at_θ(x, zMAP_prev, θ, gradz_logLike_atol=gradz_logLike_atol)
            g = ravel(self.gradθ_logLike(x, zMAP, θ))
            if progress: pbar.update()
            return (g, zMAP, history)

        g_zMAPs = list(pmap(get_MAPs, zip(xs, zMAPs)))

        zMAPs = [zMAP for (_,zMAP,_) in g_zMAPs]
        # if save_zMAP_history: zMAP_history_dat, *zMAP_history_sims = [history for (_,_,history) in g_zMAPs]
        g_like_dat, *g_like_sims = [g for (g,_,_) in g_zMAPs]
        g_like = g_like_dat - np.mean(np.stack(g_like_sims), axis=0)
        return unravel(g_like)



    def get_J(
        self,
        result = None,
        θ0 = None,
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
        if θ0 is None:
            if result.θ is None:
                raise Exception("θ0 or result.θ must be given.")
            else:
                θ0 = result.θ

        nsims_remaining = nsims - len(result.gs)

        if nsims_remaining > 0:

            pbar = tqdm(total=nsims_remaining, desc="get_J") if progress else None
            t0 = datetime.now()

            ravel, unravel = self._ravel_unravel(θ0)

            xz_sims = [self.sample_x_z(_rng, θ0) for _rng in self._split_rng(rng, nsims_remaining)]

            def get_g(x_z):
                try:
                    x, z = x_z
                    zMAP = self.zMAP_at_θ(x, z, θ0, gradz_logLike_atol=gradz_logLike_atol)[0]
                    g = ravel(self.gradθ_logLike(x, zMAP, θ0, transformed_θ=False))
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
        θ0 = None,
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
        if θ0 is None:
            if result.θ is None:
                raise Exception("θ0 or result.θ must be given.")
            else:
                θ0 = result.θ


        nsims_remaining = nsims - len(result.Hs)

        if nsims_remaining > 0:
                
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
                zMAP = self.zMAP_at_θ(x, z, θ0, gradz_logLike_atol=gradz_logLike_atol)[0]
                if progress: pbar.update()
                return zMAP
            zMAPs_fid = pmap(get_zMAP, xz_sims)

            # finite difference Jacobian
            # pmap_sims, pmap_jac = (pmap_over == :jac || (pmap_over == :auto && length(θ₀) > nsims_remaining)) ? (_map, pmap) : (pmap, _map)
            def getH(x_zMAPfid_rng):
                x, zMAPfid, rng = x_zMAPfid_rng
                def get_sMAP(θvec):
                    θ = unravel(θvec)
                    x = self.sample_x_z(copy(rng), θ)[0]
                    zMAP = self.zMAP_at_θ(x, zMAPfid, θ0, gradz_logLike_atol=gradz_logLike_atol)[0]
                    return ravel(self.gradθ_logLike(x, zMAP, θ0, transformed_θ=False))
                try:
                    return pjacobian(get_sMAP, ravel(θ0), step, pbar=pbar)
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