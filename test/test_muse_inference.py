
import sys

sys.path.append("..")

from functools import partial
from numbers import Number

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest
from multiprocess import Pool
from muse_inference import MuseProblem, MuseResult
from muse_inference.jax import JaxMuseProblem, JittableJaxMuseProblem
from muse_inference.pymc import PyMCMuseProblem


@pytest.mark.parametrize("pmap", [map, Pool(4).map])
def test_scalar_numpy(pmap):

    class NumpyFunnelMuseProblem(MuseProblem):
        
        def __init__(self, N):
            super().__init__()
            self.N = N
        
        def sample_x_z(self, rng, θ):
            z = rng.normal(size=self.N) * np.exp(θ/2)
            x = z + rng.normal(size=self.N)
            return (x, z)
        
        def logLike_and_gradzθ_logLike(self, x, z, θ, transformed_θ=None):
            logLike = -(np.sum((x - z)**2) + np.sum(z**2) / np.exp(θ) + 512*θ) / 2
            gradz_logLike = x - z * (1 + np.exp(-θ))
            gradθ_logLike = np.sum(z**2)/(2*np.exp(θ)) - self.N/2
            return (logLike, gradz_logLike, gradθ_logLike)
        
        def gradθ_and_hessθ_logPrior(self, θ, transformed_θ=None):
            return (-θ/(3**2), -1/3**2)

    θ_true = 1.
    θ_start = 0.

    prob = NumpyFunnelMuseProblem(512)
    rng = np.random.RandomState(0)
    prob.x = prob.sample_x_z(rng, θ_true)[0]

    result = prob.solve(θ_start=θ_start, rng=np.random.SeedSequence(1), pmap=pmap)
    prob.get_J(result, nsims=len(result.s_MAP_sims)+10, rng=np.random.SeedSequence(2), pmap=pmap)

    assert isinstance(result.θ, Number)
    assert result.Σ.shape == (1,1)
    assert result.dist.cdf(θ_true) > 0.01



@pytest.mark.parametrize("pmap", [map, Pool(4).map])
def test_ravel_numpy(pmap):

    class NumpyFunnelMuseProblem(MuseProblem):
    
        def __init__(self, N):
            super().__init__()
            self.N = N
        
        def sample_x_z(self, rng, θ):
            (θ1, θ2) = θ
            z1 = rng.normal(size=self.N) * np.exp(θ1/2)
            z2 = rng.normal(size=self.N) * np.exp(θ2/2)        
            x1 = z1 + rng.normal(size=self.N)
            x2 = z2 + rng.normal(size=self.N)        
            return ((x1,x2), (z1,z2))
        
        def logLike_and_gradzθ_logLike(self, x, z, θ, transformed_θ=None):
            (θ1, θ2) = θ
            (x1, x2) = x
            (z1, z2) = z
            logLike = -(np.sum((x1 - z1)**2) + np.sum(z1**2) / np.exp(θ1) + 512*θ1) / 2 -(np.sum((x2 - z2)**2) + np.sum(z2**2) / np.exp(θ2) + 512*θ2) / 2
            gradz_logLike = (x1 - z1 * (1 + np.exp(-θ1)), x2 - z2 * (1 + np.exp(-θ2)))
            gradθ_logLike = (np.sum(z1**2)/(2*np.exp(θ1)) - self.N/2, np.sum(z2**2)/(2*np.exp(θ2)) - self.N/2)
            return (logLike, gradz_logLike, gradθ_logLike)
        
        def gradθ_and_hessθ_logPrior(self, θ, transformed_θ=None):
            (θ1, θ2) = θ
            g = (-θ1/(3**2), -θ2/(3**2))
            H = (
                (-1/3**2, 0),
                (0,       -1/3**2)
            )
            return (g, H)

    θ_true = (-1., 2.)
    θ_start = (0., 0.)

    prob = NumpyFunnelMuseProblem(512)
    rng = np.random.RandomState(0)
    prob.x = prob.sample_x_z(rng, θ_true)[0]

    result = prob.solve(θ_start=θ_start, rng=np.random.SeedSequence(1), pmap=pmap)
    prob.get_J(result, nsims=len(result.s_MAP_sims)+10, rng=np.random.SeedSequence(2), pmap=pmap)


    assert isinstance(result.θ, tuple)
    assert result.Σ.shape == (2,2)
    assert result.dist.cdf(result.ravel(θ_true)) > 0.01



def test_scalar_jax():

    class JaxFunnelMuseProblem(JittableJaxMuseProblem):

        def __init__(self, N):
            super().__init__()
            self.N = N

        @partial(jax.jit, static_argnums=0)
        def sample_x_z(self, key, θ):
            keys = jax.random.split(key, 2)
            z = jax.random.normal(keys[0], (self.N,)) * jnp.exp(θ/2)
            x = z + jax.random.normal(keys[1], (self.N,))
            return (x, z)

        @partial(jax.jit, static_argnums=0)
        def logLike(self, x, z, θ):
            return -(jnp.sum((x - z)**2) + jnp.sum(z**2) / jnp.exp(θ) + 512*θ) / 2

        @partial(jax.jit, static_argnums=0)
        def logPrior(self, θ):
            return -θ**2 / (2*3**2)

    θ_true = 1.
    θ_start = 0.

    prob = JaxFunnelMuseProblem(512)
    keys = prob._split_rng(jax.random.PRNGKey(0), 3)
    (x, z) = prob.sample_x_z(keys[0], θ_true)
    prob.x = x

    result = prob.solve(θ_start=θ_start, rng=keys[1], method=None, maxsteps=10)
    prob.get_J(result, nsims=len(result.s_MAP_sims)+10, rng=keys[2], method=None)

    assert result.θ.shape == ()
    assert result.Σ.shape == (1,1)
    assert result.dist.cdf(result.ravel(θ_true)) > 0.01


def test_ravel_jax():

    class JaxFunnelMuseProblem(JittableJaxMuseProblem):

        def __init__(self, N):
            super().__init__()
            self.N = N

        @partial(jax.jit, static_argnums=0)
        def sample_x_z(self, key, θ):
            (θ1, θ2) = (θ["θ1"], θ["θ2"])
            keys = jax.random.split(key, 4)
            z1 = jax.random.normal(keys[0], (self.N,)) * jnp.exp(θ1/2)
            z2 = jax.random.normal(keys[1], (self.N,)) * jnp.exp(θ2/2)        
            x1 = z1 + jax.random.normal(keys[2], (self.N,))
            x2 = z2 + jax.random.normal(keys[3], (self.N,))        
            return ({"x1":x1, "x2":x2}, {"z1":z1, "z2":z2})

        @partial(jax.jit, static_argnums=0)
        def logLike(self, x, z, θ):
            return (
                -(jnp.sum((x["x1"] - z["z1"])**2) + jnp.sum(z["z1"]**2) / jnp.exp(θ["θ1"]) + 512*θ["θ1"]) / 2
                -(jnp.sum((x["x2"] - z["z2"])**2) + jnp.sum(z["z2"]**2) / jnp.exp(θ["θ2"]) + 512*θ["θ2"]) / 2
            )
        
        @partial(jax.jit, static_argnums=0)
        def logPrior(self, θ):
            return - θ["θ1"]**2 / (2*3**2) - θ["θ2"]**2 / (2*3**2)

    θ_true = {"θ1":-1., "θ2":2.}
    θ_start = {"θ1":0., "θ2":0.}

    prob = JaxFunnelMuseProblem(512)
    keys = prob._split_rng(jax.random.PRNGKey(0), 3)
    (x, z) = prob.sample_x_z(keys[0], θ_true)
    prob.x = x

    result = prob.solve(θ_start=θ_start, rng=keys[1], method=None, maxsteps=10)
    prob.get_J(result, nsims=len(result.s_MAP_sims)+10, rng=keys[2], method=None)

    assert isinstance(result.θ, dict) and result.θ["θ1"].shape == result.θ["θ2"].shape == ()
    assert result.Σ.shape == (2,2)
    assert result.dist.cdf(result.ravel(θ_true)) > 0.01


def test_scalar_pymc():

    def gen_funnel(x=None, θ=None):
        with pm.Model() as funnel:
            θ = pm.Normal("θ", mu=0, sigma=3) if θ is None else θ
            z = pm.Normal("z", mu=0, sigma=np.exp(θ/2), size=512)
            x = pm.Normal("x", mu=z, sigma=1, observed=x)
        return funnel

    θ_true = 1.
    θ_start = 0.

    with gen_funnel(θ=θ_true):
        x_obs = pm.sample_prior_predictive(1, random_seed=0).prior.x[0,0]
    funnel = gen_funnel(x_obs)
    prob = PyMCMuseProblem(funnel)

    result = prob.solve(θ_start=θ_start, rng=np.random.SeedSequence(1))
    prob.get_J(result, nsims=len(result.s_MAP_sims)+10, rng=np.random.SeedSequence(2))

    assert isinstance(result.θ, dict) and result.θ["θ"].shape == ()
    assert result.Σ.shape == (1,1)
    assert result.dist.cdf(θ_true) > 0.01


def test_ravel_pymc():

    def gen_funnel(x1=None, x2=None, θ1=None, θ2=None):
        with pm.Model() as funnel:
            θ1 = pm.Normal("θ1", mu=0, sigma=3) if θ1 is None else θ1
            θ2 = pm.Normal("θ2", mu=0, sigma=3) if θ2 is None else θ2
            z1 = pm.Normal("z1", mu=0, sigma=np.exp(θ1/2), size=256)
            z2 = pm.Normal("z2", mu=0, sigma=np.exp(θ2/2), size=(16,16))
            x1 = pm.Normal("x1", mu=z1, sigma=1, observed=x1)
            x2 = pm.Normal("x2", mu=z2, sigma=1, observed=x2)
        return funnel

    θ_true  = dict(θ1=-1, θ2=2)
    θ_start = dict(θ1=0,  θ2=0)

    with gen_funnel(rng_seeder=rng, **θ_true):
        truth = pm.sample_prior_predictive(1, random_seed=0).prior
    funnel = gen_funnel(x1=truth.x1[0,0], x2=truth.x2[0,0])
    prob = PyMCMuseProblem(funnel)

    result = prob.solve(θ_start=θ_start, rng=np.random.SeedSequence(1))
    prob.get_J(result, nsims=len(result.s_MAP_sims)+10, rng=np.random.SeedSequence(2))

    assert isinstance(result.θ, dict) and result.θ["θ1"].shape == result.θ["θ2"].shape == ()
    assert result.Σ.shape == (2,2)
    assert result.dist.cdf(result.ravel(θ_true)) > 0.01
