---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Demo

```python tags=[] nbsphinx="hidden"
%matplotlib inline
%config InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 110}
%load_ext autoreload
%autoreload 2
```

Sample text. $\mathcal{P}(x\,|\,\theta)$

```python
import matplotlib.pyplot as plt
import muse_inference
from muse_inference import MuseProblem
import numpy as np
```

```python
θ_true = 1.
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] -->
## With numpy
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Scalar
<!-- #endregion -->

```python
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
    
```

```python
prob = NumpyFunnelMuseProblem(512)
rng = np.random.RandomState(0)
(x, z) = prob.sample_x_z(rng, θ_true)
prob.x = x
```

```python
result = prob.solve(0, α=0.7, rng=np.random.RandomState(3), gradz_logLike_atol=1e-4, progress=True, maxsteps=10);
```

```python
plt.plot([h["θ"] for h in result.history], ".-")
plt.xlabel("step")
plt.ylabel("θ");
```

### Tuple

```python
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
```

```python
prob = NumpyFunnelMuseProblem(2048)
θ = (-1., 5.)
x, z = prob.sample_x_z(np.random.RandomState(0), θ)
prob.x = x
```

```python
result = prob.solve(θ, α=0.7, rng=np.random.RandomState(3), gradz_logLike_atol=1e-4, progress=True, maxsteps=10);
```

```python
result.history[-1]["θ"]
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## With Jax
<!-- #endregion -->

```python
import jax
import jax.numpy as jnp
from muse_inference.jax import JittedJaxMuseProblem, JaxMuseProblem
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
### Scalar
<!-- #endregion -->

```python
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
```

```python
prob = JaxFunnelMuseProblem(512)
rng = np.random.RandomState(0)
(x, z) = prob.sample_x_z(rng, θ_true)
prob.x = x
```

```python
result = prob.solve(0., α=0.7, rng=np.random.RandomState(3), gradz_logLike_atol=1e-4, progress=True, maxsteps=10);
```

```python
[h["θ"] for h in result.history]
```

```python
plt.plot([h["θ"] for h in result.history], ".-")
plt.xlabel("step")
plt.ylabel("θ");
```

### Tuple

```python
class JaxFunnelMuseProblem(muse_inference.jax.JittedJaxMuseProblem):
    
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
        return -θ["θ1"]**2 / (2*3**2) - θ["θ2"]**2 / (2*3**2)
```

```python
prob = JaxFunnelMuseProblem(512)
θ = {"θ1":1., "θ2":2.}
x, z = prob.sample_x_z(np.random.RandomState(0), θ)
prob.x = x
```

```python
result = prob.solve(θ, rng=rng, gradz_logLike_atol=1e-4, progress=True, maxsteps=10);
```

```python
result.θ
```

```python
plt.plot([h["θ"]["θ1"] for h in result.history], ".-")
plt.plot([h["θ"]["θ2"] for h in result.history], ".-")

plt.xlabel("step")
plt.ylabel("θ");
```

## With PyMC

```python
import sys
import pymc as pm
from muse_inference.pymc import PyMCMuseProblem
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Scalar
<!-- #endregion -->

```python
# define 
def gen_funnel(x=None, θ=None, rng_seeder=None):
    with pm.Model(rng_seeder=rng_seeder) as funnel:
        θ = θ if θ else pm.Normal("θ", 0, 3)
        z = pm.Normal("z", 0, np.exp(θ / 2), size=512)
        x = pm.Normal("x", z, 1, observed=x)
    return funnel
        
# generated simulated data
rng = np.random.RandomState(0)
x_obs = pm.sample_prior_predictive(1, model=gen_funnel(θ=θ_true, rng_seeder=rng)).prior.x[0,0]

# set up problem
funnel = gen_funnel(x_obs)
prob = PyMCMuseProblem(funnel)
```

```python
result = prob.solve(0., rng=rng, gradz_logLike_atol=1e-4, progress=True, maxsteps=10);
```

```python
[h["θ"] for h in result.history]
```

```python
plt.plot([h["θ"] for h in result.history], ".-")
plt.xlabel("step")
plt.ylabel("θ");
```

### Tuple

```python
# define 
def gen_funnel(x=(None,None), θ=(None,None), rng_seeder=None, N=3):
    (α, β) = θ
    with pm.Model(rng_seeder=rng_seeder) as funnel:
        α = pm.Normal("α", 0, 3) if α is None else α
        β = pm.Normal("β", 0, 3) if β is None else β
        z1 = pm.Normal("z1", 0, np.exp(α / 2), size=N)
        z2 = pm.Normal("z2", 0, np.exp(β / 2), size=N)
        x1 = pm.Normal("x1", z1, 1, observed=x[0])
        x2 = pm.Normal("x2", z2, 1, observed=x[1])
    return funnel
        
# generated simulated data
rng = np.random.RandomState(0)
θ = (-1, 3)
prior = pm.sample_prior_predictive(1, model=gen_funnel(θ=θ, rng_seeder=rng)).prior
x_obs = (prior.x1, prior.x2)
```

```python
# set up problem
funnel = gen_funnel(x_obs)
prob = PyMCMuseProblem(funnel)
```

```python
result = prob.solve((0,0), α=0.3, rng=rng, gradz_logLike_atol=1e-4, progress=True, maxsteps=10);
```

```python
[h["θ"] for h in result.history]
```

```python
plt.plot([h["θ"][0] for h in result.history], ".-")
plt.plot([h["θ"][1] for h in result.history], ".-")

plt.xlabel("step")
plt.ylabel("θ");
```
