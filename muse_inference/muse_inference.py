
__all__ = ["MuseProblem", "MuseResult"]

from copy import copy
from datetime import datetime
from numbers import Number
from time import thread_time

import numpy as np
from numpy.random import RandomState
from scipy.optimize import minimize
from tqdm import tqdm


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
        ravel, unravel = self.ravel_unravel(z0)
        def objective(z_vec):
            logLike, gradz_logLike = self.logLike_and_gradz_logLike(x, unravel(z_vec), θ)
            return (-logLike, -ravel(gradz_logLike))
        soln = minimize(objective, ravel(z0), method='L-BFGS-B', jac=True, tol=gradz_logLike_atol)
        return (unravel(soln.x), soln)

    def ravel_unravel(self, x):
        if isinstance(x, (tuple,list)):
            np = self.np
            i = 0
            slices_shapes = []
            for elem in x:
                if isinstance(elem, Number):
                    slices_shapes.append((i, None))
                    i += 1
                else:
                    slices_shapes.append((slice(i,i+elem.size), elem.shape))
                    i += elem.size
            unravel = lambda vec: tuple(vec[sl] if shape is None else vec[sl].reshape(shape) for (sl,shape) in slices_shapes)
            ravel = lambda tup: np.concatenate(tup, axis=None)
        else:
            ravel = unravel = lambda x: x
        return (ravel, unravel)

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
        get_covariance = False
    ):

        np = self.np

        if result is None:
            result = MuseResult()
        if rng is None:
            rng = RandomState()
        if z0 is None:
            z0 = self.sample_x_z(copy(rng), θ_start)[1]

        θunreg = θ = θ_start

        is_scalar_θ = isinstance(θ, Number)
        ravel, unravel = self.ravel_unravel(θ)
        Nθ = 1 if is_scalar_θ else len(ravel(θ))

        result.rng = _rng = copy(rng)
        xz_sims = [self.sample_x_z(_rng, θ) for i in range(nsims)]
        xs    = [self.x] + [x for (x,_) in xz_sims]
        zMAPs = [z0]     + [z for (_,z) in xz_sims]

        if progress: 
            pbar = tqdm(total=(maxsteps-len(result.history))*(nsims+1))

        try:
            
            for i in range((len(result.history)+1), maxsteps+1):
                
                t0 = datetime.now()

                if i > 1:
                    _rng = copy(rng)
                    xs = [self.x] + [self.sample_x_z(_rng, θ)[0] for i in range(nsims)]

                if i > 2:
                    Δθ = ravel(result.history[-1]["θ"]) - ravel(result.history[-2]["θ"])
                    if np.sqrt(-np.inner(Δθ, np.inner(H_inv_post, Δθ))) < θ_rtol:
                        break

                # MUSE gradient
                def get_MAPs(x, zMAP_prev, θ):
                    (zMAP, history) = self.zMAP_at_θ(x, zMAP_prev, θ, gradz_logLike_atol=gradz_logLike_atol)
                    g = ravel(self.gradθ_logLike(x, zMAP, θ))
                    if progress: pbar.update()
                    return (g, zMAP, history)
                    
                g_zMAPs = list(pmap(get_MAPs, xs, zMAPs, [θ]*(nsims+1)))

                zMAPs = [zMAP for (_,zMAP,_) in g_zMAPs]
                zMAP_history_dat, *zMAP_history_sims = [history for (_,_,history) in g_zMAPs]
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

        result.θ = θ    

        return result



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
        self.time = 0

