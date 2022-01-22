
import sys

sys.path.append("..")

from numbers import Number

import jax
import jax.numpy as jnp
import numpy as np
from muse_inference import MuseProblem, MuseResult
from muse_inference.jax import JaxMuseProblem, JittedJaxMuseProblem


def test_scalar_numpy():

    class NumpyFunnelMuseProblem(MuseProblem):
        
        def __init__(self, N):
            super().__init__()
            self.N = N
        
        def sample_x_z(self, rng, θ):
            z = rng.randn(self.N) * np.exp(θ/2)
            x = z + rng.randn(self.N)
            return (x, z)
        
        def gradθ_logLike(self, x, z, θ):
            return np.sum(z**2)/(2*np.exp(θ)) - self.N/2
        
        def logLike_and_gradz_logLike(self, x, z, θ):
            logLike = -(np.sum((x - z)**2) + np.sum(z**2) / np.exp(θ) + 512*θ) / 2
            gradz_logLike = x - z * (1 + np.exp(-θ))
            return (logLike, gradz_logLike)
        
        def grad_hess_θ_logPrior(self, θ):
            return (-θ/(3**2), -1/3**2)
        
    
    θ_true = 1.
    θ_start = 0.

    prob = NumpyFunnelMuseProblem(512)
    rng = np.random.RandomState(0)
    prob.x = prob.sample_x_z(rng, θ_true)[0]

    result = prob.solve(θ_start=θ_start, rng=rng, gradz_logLike_atol=1e-4, maxsteps=10)

    assert isinstance(result.θ, Number)
    assert np.linalg.norm(result.θ - θ_true) < 0.1




def test_ravel_numpy():

    class NumpyFunnelMuseProblem(MuseProblem):
    
        def __init__(self, N):
            super().__init__()
            self.N = N
        
        def sample_x_z(self, rng, θ):
            (θ1, θ2) = θ
            z1 = rng.randn(self.N) * np.exp(θ1/2)
            z2 = rng.randn(self.N) * np.exp(θ2/2)        
            x1 = z1 + rng.randn(self.N)
            x2 = z2 + rng.randn(self.N)        
            return ((x1,x2), (z1,z2))
        
        def gradθ_logLike(self, x, z, θ):
            (θ1, θ2) = θ
            (x1, x2) = x
            (z1, z2) = z
            return (np.sum(z1**2)/(2*np.exp(θ1)) - self.N/2, np.sum(z2**2)/(2*np.exp(θ2)) - self.N/2)
        
        def logLike_and_gradz_logLike(self, x, z, θ):
            (θ1, θ2) = θ
            (x1, x2) = x
            (z1, z2) = z
            logLike = -(np.sum((x1 - z1)**2) + np.sum(z1**2) / np.exp(θ1) + 512*θ1) / 2 -(np.sum((x2 - z2)**2) + np.sum(z2**2) / np.exp(θ2) + 512*θ2) / 2
            gradz_logLike = (x1 - z1 * (1 + np.exp(-θ1)), x2 - z2 * (1 + np.exp(-θ2)))
            return (logLike, gradz_logLike)
        
        def gradθ_and_hessθ_logPrior(self, θ):
            (θ1, θ2) = θ
            g = (-θ1/(3**2), -θ2/(3**2))
            H = ((-1/3**2, 0),
                (0,      -1/3**2))
            return g, H    


    θ_true = (-1., 3.)
    θ_start = (0., 0.)

    prob = NumpyFunnelMuseProblem(512)
    rng = np.random.RandomState(0)
    prob.x = prob.sample_x_z(rng, θ_true)[0]

    result = prob.solve(θ_start=θ_start, rng=rng, gradz_logLike_atol=1e-4, maxsteps=40)

    ravel = prob._ravel_unravel(θ_true)[0]

    assert isinstance(result.θ, tuple)
    assert np.linalg.norm(ravel(result.θ) - ravel(θ_true)) < 0.2



def test_scalar_jax():

    class JaxFunnelMuseProblem(JittedJaxMuseProblem):
        
        def __init__(self, N):
            super().__init__()
            self.N = N

        def sample_x_z(self, rng, θ):
            z = rng.randn(self.N) * np.exp(θ/2)
            x = z + rng.randn(self.N)
            return (jnp.array(x), jnp.array(z))

        def logLike(self, x, z, θ):
            return -(jnp.sum((x - z)**2) + jnp.sum(z**2) / jnp.exp(θ) + 512*θ) / 2
        
        def logPrior(self, θ):
            return -θ**2 / (2*3**2)


    θ_true = 1.
    θ_start = 0.

    prob = JaxFunnelMuseProblem(512)
    rng = np.random.RandomState(0)
    prob.x = prob.sample_x_z(rng, θ_true)[0]

    result = prob.solve(θ_start=θ_start, rng=rng, gradz_logLike_atol=1e-4, maxsteps=10)

    assert result.θ.size == 1
    assert np.linalg.norm(result.θ - θ_true) < 0.1



def test_ravel_jax():

    class JaxFunnelMuseProblem(JittedJaxMuseProblem):
    
        def __init__(self, N):
            super().__init__()
            self.N = N

        def sample_x_z(self, rng, θ):
            (θ1, θ2) = (θ["θ1"], θ["θ2"])
            z1 = rng.randn(self.N) * np.exp(θ1/2)
            z2 = rng.randn(self.N) * np.exp(θ2/2)        
            x1 = z1 + rng.randn(self.N)
            x2 = z2 + rng.randn(self.N)        
            return ({"x1":x1, "x2":x2}, {"z1":z1, "z2":z2})

        def logLike(self, x, z, θ):
            return (
                -(jnp.sum((x["x1"] - z["z1"])**2) + jnp.sum(z["z1"]**2) / jnp.exp(θ["θ1"]) + 512*θ["θ1"]) / 2
                -(jnp.sum((x["x2"] - z["z2"])**2) + jnp.sum(z["z2"]**2) / jnp.exp(θ["θ2"]) + 512*θ["θ2"]) / 2
            )
        
        def logPrior(self, θ):
            return - θ["θ1"]**2 / (2*3**2) - θ["θ2"]**2 / (2*3**2)


    θ_true = {"θ1":-1., "θ2":3.}
    θ_start = {"θ1":0., "θ2":0.}

    prob = JaxFunnelMuseProblem(512)
    rng = np.random.RandomState(0)
    prob.x = prob.sample_x_z(rng, θ_true)[0]

    result = prob.solve(θ_start=θ_start, rng=rng, gradz_logLike_atol=1e-4, maxsteps=10)

    ravel = prob._ravel_unravel(θ_true)[0]
    assert isinstance(result.θ, dict)
    assert np.linalg.norm(ravel(result.θ) - ravel(θ_true)) < 0.2
