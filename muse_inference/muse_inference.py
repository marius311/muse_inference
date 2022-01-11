
__all__ = ["MuseProblem", "MuseResult", "muse"]

from copy import copy
from datetime import datetime
from time import thread_time

from numpy import inner, mean, sqrt, squeeze, var
from numpy.linalg import inv, norm
from numpy.random import RandomState
from scipy.optimize import minimize
from tqdm import tqdm


class MuseProblem():

    def __init__(self):
        self.x = None

    def sample_x_z(self, θ):
        raise NotImplementedError()

    def gradθ_logLike(self, x, z, θ):
        raise NotImplementedError()

    def logLike_and_gradz_logLike(self, x, z, θ):
        raise NotImplementedError()

    def grad_hess_θ_logPrior(self, θ):
        return (0,0)

    def zMAP_at_θ(self, x, z0, θ, gradz_logLike_atol=None):
        def objective(z):
            logLike, gradz_logLike = self.logLike_and_gradz_logLike(x, z, θ)
            return -logLike, -gradz_logLike
        soln = minimize(objective, z0, method='L-BFGS-B', jac=True, tol=gradz_logLike_atol)
        return (soln.x, soln)



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




def muse(
    result: MuseResult,
    prob: MuseProblem,
    θ0 = None,
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

    if rng is None:
        rng = RandomState()
    if z0 is None:
        z0 = 0 * prob.sample_x_z(copy(rng), θ0)[1]

    θunreg = θ = θ0

    result.rng = _rng = copy(rng)
    xz_sims = [prob.sample_x_z(_rng, θ) for i in range(nsims)]
    xs    = [prob.x] + [x for (x,_) in xz_sims]
    zMAPs = [z0]     + [z for (_,z) in xz_sims]

    if progress: 
        pbar = tqdm(total=(maxsteps-len(result.history))*(nsims+1))

    try:
        
        for i in range((len(result.history)+1), maxsteps+1):
            
            t0 = datetime.now()

            if i > 1:
                _rng = copy(rng)
                xs = [prob.x] + [prob.sample_x_z(_rng, θ)[0] for i in range(nsims)]

            if i > 2:
                Δθ = result.history[-1]["θ"] - result.history[-2]["θ"]
                if sqrt(-inner(Δθ, inner(H_inv_post, Δθ))) < θ_rtol:
                    break

            # MUSE gradient
            def get_MAPs(x, zMAP_prev, θ):
                (zMAP, history) = prob.zMAP_at_θ(x, zMAP_prev, θ, gradz_logLike_atol=gradz_logLike_atol)
                g = prob.gradθ_logLike(x, zMAP, θ)
                if progress: pbar.update()
                return (g, zMAP, history)
                
            g_zMAPs = list(pmap(get_MAPs, xs, zMAPs, [θ]*(nsims+1)))

            zMAPs = [zMAP for (_,zMAP,_) in g_zMAPs]
            zMAP_history_dat, *zMAP_history_sims = [history for (_,_,history) in g_zMAPs]
            g_like_dat, *g_like_sims = [g for (g,_,_) in g_zMAPs]
            g_like = g_like_dat - mean(g_like_sims)
            g_prior, H_prior = prob.grad_hess_θ_logPrior(θ)
            g_post = g_like + g_prior

            h_inv_like_sims = -1 / var(g_like_sims)
            H_inv_like_sims = [[h_inv_like_sims]] if isinstance(h_inv_like_sims, float) else None # Diagonal
            H_inv_like = H_inv_like_sims

            H_inv_post = inv(inv(H_inv_like) + H_prior)

            t = datetime.now() - t0

            result.history.append({
                "θ":θ, "θunreg":θunreg, "t":t, "g_like_dat": g_like_dat,
                "g_like_sims": g_like_sims, "g_like": g_like, "g_prior": g_prior,
                "g_post": g_post, "H_inv_post": H_inv_post, "H_prior": H_prior,
                "H_inv_like": H_inv_like, "H_inv_like_sims": H_inv_like_sims,
                "zMAP_history_dat": zMAP_history_dat, "zMAP_history_sims": zMAP_history_sims,
            })

            θunreg = squeeze(θ - α * (inner(H_inv_post, g_post)))
            θ = regularize(θunreg)

        if progress: pbar.update(pbar.total - pbar.n)
    
    finally:
        if progress: pbar.close()

    return result
