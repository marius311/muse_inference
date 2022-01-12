---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3 (MuseInference)
    language: python
    name: python3-museinference
---

```python
%pylab inline
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 130
```

```python
%load_ext autoreload
```

```python
%autoreload 2
```

```python
import sys
sys.path.append("..")
```

```python
from muse_inference import muse, MuseProblem, MuseResult
import numpy as np
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
# Numpy
<!-- #endregion -->

```python
class NumpyFunnelMuseProblem(MuseProblem):
    
    def __init__(self, N):
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
```

```python
prob = NumpyFunnelMuseProblem(512)
rng = np.random.RandomState(0)
(x, z) = prob.sample_x_z(rng, 0)
prob.x = x
```

```python
result = MuseResult()
muse(result, prob, 0, α=0.7, rng=np.random.RandomState(3), gradz_logLike_atol=1e-4, progress=True);
```

```python
i = 2
hist([g for g in result.history[i]["g_like_sims"]])
axvline(result.history[i]["g_like_dat"], c="C1")
```

```python
plot([h["θ"] for h in result.history], ".-")
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
# Jax
<!-- #endregion -->

```python
from muse_inference.jax import JaxMuseProblem, JittedJaxMuseProblem
import jax
import jax.numpy as jnp
from functools import partial
```

```python
class JaxFunnelMuseProblem(JittedJaxMuseProblem):
    
    def __init__(self, N):
        self.N = N

    def sample_x_z(self, rng, θ):
        z = rng.randn(self.N) * np.exp(θ/2)
        x = z + rng.randn(self.N)
        return (x, z)

    def logLike(self, x, z, θ):
        return -(jnp.sum((x - z)**2) + jnp.sum(z**2) / jnp.exp(θ) + 512*θ) / 2
    
    def logPrior(self, θ):
        return -θ**2 / (2*3**2)
```

```python
prob = JaxFunnelMuseProblem(512)
rng = np.random.RandomState(0)
(x, z) = prob.sample_x_z(rng, 0)
prob.x = x
```

```python
%timeit prob.logLike_and_gradz_logLike(x, z, 0.)
```

```python
%timeit prob.zMAP_at_θ(x, z, 0.)
```

```python
result = MuseResult()
muse(result, prob, 0., α=0.7, rng=np.random.RandomState(3), gradz_logLike_atol=1e-4, progress=True);
```

```python
plot([h["θ"] for h in result.history], ".-")
```

# PyMC

```python
from muse_inference.pymc import PyMCMuseProblem
import pymc as pm
import aesara
import aesara.tensor as at
from pymc.distributions import logpt as joint_logpt
```

```python
def gen_funnel(x=None, θ=None):
    with pm.Model() as funnel:
        θ = θ if θ else pm.Normal("θ", 0, 3)
        z = pm.Normal("z", 0, at.exp(θ / 2), size=512)
        x = pm.Normal("x", z, 1, observed=x)
    return funnel
        
x_obs = pm.sample_prior_predictive(1, model=gen_funnel(θ=2)).prior.x[0,0]
funnel = gen_funnel(x_obs)
prob = PyMCMuseProblem(funnel)
prob.x = x_obs
```

```python
result = MuseResult()
muse(result, prob, 0., α=0.7, rng=np.random.RandomState(3), gradz_logLike_atol=1e-4, progress=True);
```

```python
plot([h["θ"] for h in result.history], ".-")
```
