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
    display_name: Poetry
    language: python
    name: poetry-kernel
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
from multiprocess import Pool
from tqdm import tqdm
```

```python
θ_true = 1.
(θ1_true, θ2_true) = (-1., 2.)
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
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
        z = rng.normal(size=self.N) * np.exp(θ/2)
        x = z + rng.normal(size=self.N)
        return (x, z)
    
    def gradθ_logLike(self, x, z, θ):
        return np.sum(z**2)/(2*np.exp(θ)) - self.N/2
    
    def logLike_and_gradz_logLike(self, x, z, θ):
        logLike = -(np.sum((x - z)**2) + np.sum(z**2) / np.exp(θ) + 512*θ) / 2
        gradz_logLike = x - z * (1 + np.exp(-θ))
        return (logLike, gradz_logLike)
    
    def gradθ_and_hessθ_logPrior(self, θ):
        return (-θ/(3**2), -1/3**2)
```

```python
prob = NumpyFunnelMuseProblem(512)
rng = np.random.RandomState(0)
(x, z) = prob.sample_x_z(rng, θ_true)
prob.x = x
```

```python
result = prob.solve(0, rng=np.random.SeedSequence(0), gradz_logLike_atol=1e-4, progress=True, maxsteps=10, get_covariance=True);
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
        z1 = rng.normal(size=self.N) * np.exp(θ1/2)
        z2 = rng.normal(size=self.N) * np.exp(θ2/2)        
        x1 = z1 + rng.normal(size=self.N)
        x2 = z2 + rng.normal(size=self.N)        
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
rng = np.random.RandomState(0)
x, z = prob.sample_x_z(rng, (θ1_true, θ2_true))
prob.x = x
```

```python
result = prob.solve((0,0), α=0.7, rng=np.random.SeedSequence(0), gradz_logLike_atol=1e-4, progress=True, maxsteps=10, get_covariance=True);
```

```python
plt.plot([h["θ"][0] for h in result.history], ".-")
plt.plot([h["θ"][1] for h in result.history], ".-")

plt.xlabel("step")
plt.ylabel("θ");
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
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

    def sample_x_z(self, key, θ):
        keys = jax.random.split(key, 2)
        z = jax.random.normal(keys[0], (self.N,)) * np.exp(θ/2)
        x = z + jax.random.normal(keys[1], (self.N,))
        return (x, z)

    def logLike(self, x, z, θ):
        return -(jnp.sum((x - z)**2) + jnp.sum(z**2) / jnp.exp(θ) + 512*θ) / 2
    
    def logPrior(self, θ):
        return -θ**2 / (2*3**2)
```

```python
prob = JaxFunnelMuseProblem(512)
key = jax.random.PRNGKey(0)
(x, z) = prob.sample_x_z(key, θ_true)
prob.x = x
```

```python
result = prob.solve(0., α=0.7, rng=jax.random.PRNGKey(1), gradz_logLike_atol=1e-4, progress=True, maxsteps=10);
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

    def sample_x_z(self, key, θ):
        (θ1, θ2) = (θ["θ1"], θ["θ2"])
        keys = jax.random.split(key, 4)
        z1 = jax.random.normal(keys[0], (self.N,)) * np.exp(θ1/2)
        z2 = jax.random.normal(keys[1], (self.N,)) * np.exp(θ2/2)        
        x1 = z1 + jax.random.normal(keys[2], (self.N,))
        x2 = z2 + jax.random.normal(keys[3], (self.N,))        
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
key = jax.random.PRNGKey(0)
(x, z) = prob.sample_x_z(key, {"θ1":θ1_true, "θ2":θ2_true})
prob.x = x
```

```python
result = prob.solve(θ_start={"θ1":0., "θ2":0.}, rng=jax.random.PRNGKey(0), gradz_logLike_atol=1e-4, progress=True, maxsteps=10);
```

```python
plt.plot([h["θ"]["θ1"] for h in result.history], ".-")
plt.plot([h["θ"]["θ2"] for h in result.history], ".-")

plt.xlabel("step")
plt.ylabel("θ");
```

### Transform

```python
θ_true = 10
```

```python
class JaxFunnelMuseProblem(JittedJaxMuseProblem):
    
    def __init__(self, N):
        super().__init__()
        self.N = N

    def sample_x_z(self, key, logθ):
        θ = jnp.exp(logθ)
        keys = jax.random.split(key, 2)
        z = jax.random.normal(keys[0], (self.N,)) * θ
        x = z + jax.random.normal(keys[1], (self.N,))
        return (x, z)

    def logLike(self, x, z, logθ):
        θ = jnp.exp(logθ)
        return -(jnp.sum((x - z)**2) + jnp.sum(z**2) / θ**2) / 2
    
    def logPrior(self, logθ):
        θ = jnp.exp(logθ)
        return -θ**2 / (2*3**2)
```

```python
prob = JaxFunnelMuseProblem(512)
key = jax.random.PRNGKey(0)
(x, z) = prob.sample_x_z(key, np.log(θ_true))
prob.x = x
```

```python
result = prob.solve(np.log(θ_true), α=0.7, rng=jax.random.PRNGKey(1), θ_rtol=0, gradz_logLike_atol=1e-4, progress=False, maxsteps=10);
prob.get_J(result=result)
```

```python
plt.plot([h["θ"] for h in result.history], ".-")
plt.axhline(np.log(θ_true), c="k")
plt.errorbar(len(result.history)-1, result.θ, 1/np.sqrt(result.J), c="C0", capsize=4)
plt.xlabel("step")
plt.ylabel("θ");
```

```python
θs = np.linspace(np.log(8), np.log(12), 50)
Δθ = np.diff(θs)[0]
sMUSEs = np.array([prob.get_sMUSE(θ, rng=jax.random.PRNGKey(1), nsims=200) for θ in tqdm(θs)])
sPriors = np.array([prob.gradθ_and_hessθ_logPrior(θ)[0] for θ in tqdm(θs)])
```

```python
plt.plot(np.exp(θs), expnorm(np.cumsum(sMUSEs * Δθ)) / Δθ, "C1")
plt.plot(np.exp(θs), expnorm(np.cumsum(sPriors * Δθ)) / Δθ, "C2")
```

```python
prob_jax = prob
```

## With PyMC

```python
import sys
import pymc as pm
from muse_inference.pymc import PyMCMuseProblem
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true -->
### Scalar
<!-- #endregion -->

```python
# define 
def gen_funnel(x=None, θ=None, rng=None):
    with pm.Model(rng_seeder=rng) as funnel:
        θ = θ if θ else pm.Normal("θ", 0, 3)
        z = pm.Normal("z", 0, np.exp(θ / 2), size=512)
        x = pm.Normal("x", z, 1, observed=x)
    return funnel
        
# generated simulated data
rng = np.random.RandomState(0)
x_obs = pm.sample_prior_predictive(1, model=gen_funnel(θ=θ_true, rng=rng)).prior.x[0,0]

# set up problem
funnel = gen_funnel(x_obs)
prob = PyMCMuseProblem(funnel)
```

```python
result = prob.solve(0., rng=np.random.SeedSequence(0), gradz_logLike_atol=1e-4, progress=True, maxsteps=10);
```

```python
plt.plot([h["θ"] for h in result.history], ".-")
plt.xlabel("step")
plt.ylabel("θ");
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
### Tuple
<!-- #endregion -->

```python
# define 
def gen_funnel(x=(None,None), θ=(None,None), rng_seeder=None, N=3):
    (θ1, θ2) = θ
    with pm.Model(rng_seeder=rng_seeder) as funnel:
        θ1 = pm.Normal("θ1", 0, 3) if θ1 is None else θ1
        θ2 = pm.Normal("θ2", 0, 3) if θ2 is None else θ2
        z1 = pm.Normal("z1", 0, np.exp(θ1 / 2), size=N)
        z2 = pm.Normal("z2", 0, np.exp(θ2 / 2), size=N)
        x1 = pm.Normal("x1", z1, 1, observed=x[0])
        x2 = pm.Normal("x2", z2, 1, observed=x[1])
    return funnel
        
# generated simulated data
rng = np.random.RandomState(0)
prior = pm.sample_prior_predictive(1, model=gen_funnel(θ=(θ1_true,θ2_true), rng_seeder=rng)).prior
x_obs = (prior.x1, prior.x2)
```

```python
# set up problem
funnel = gen_funnel(x_obs)
prob = PyMCMuseProblem(funnel)
```

```python
result = prob.solve((0,0), α=0.3, rng=np.random.SeedSequence(0), gradz_logLike_atol=1e-4, progress=True, maxsteps=10);
```

```python
plt.plot([h["θ"][0] for h in result.history], ".-")
plt.plot([h["θ"][1] for h in result.history], ".-")

plt.xlabel("step")
plt.ylabel("θ");
```

### Transform

```python
import aesara.tensor as at
import arviz as az
import scipy.stats as st
from muse_inference import MuseResult
```

```python
# define 
def gen_funnel(x=None, θ=None, rng=None):
    with pm.Model(rng_seeder=rng) as funnel:
        θ = θ if θ else pm.HalfNormal("θ", sigma=3)
        z = pm.Normal("z", 0, θ, size=512)
        x = pm.Normal("x", z, 1, observed=x)
    return funnel

# generated simulated data
rng = np.random.RandomState(0)
x_obs = pm.sample_prior_predictive(1, model=gen_funnel(θ=θ_true, rng=rng)).prior.x[0,0]

# set up problem
funnel = gen_funnel(x_obs)
prob = PyMCMuseProblem(funnel)
```

```python
with funnel:
    chain = pm.sample(1000)
```

```python
# t = funnel.rvs_to_values[funnel.θ].tag.transform
# x = at.dscalar('x')
# aesara.function([x], t.backward(x))
```

```python
# prob.solve(np.log(9.), α=0.7, rng=np.random.SeedSequence(0), θ_rtol=0, gradz_logLike_atol=1e-4, progress=True, maxsteps=40)
```

```python
θs = np.linspace(np.log(9), np.log(11), 100)
Δθ = np.diff(θs)[0]
sMUSEs = np.array([prob.get_sMUSE(θ, rng=np.random.SeedSequence(0), nsims=200) for θ in tqdm(θs)])
sPriors = np.array([prob.gradθ_and_hessθ_logPrior(θ)[0] for θ in tqdm(θs)])
```

```python
def expnorm(Ps):
    Ps = np.exp(Ps - max(Ps))
    return Ps / sum(Ps)
```

```python
plt.semilogy(np.exp(θs), expnorm(np.cumsum(sMUSEs * Δθ)) / Δθ, "C1")
# plt.plot(np.exp(θs), expnorm(np.cumsum(sPriors * Δθ)) / Δθ, "C2")
```

```python
az.plot_posterior(chain, var_names=['θ'])
plt.twinx()
_xlim = plt.xlim()
plt.plot(np.exp(θs), expnorm(np.cumsum(sMUSEs * Δθ)) / Δθ, "C1")
plt.plot(np.exp(θs), expnorm(np.cumsum(sPriors * Δθ)) / Δθ, "C2")
plt.xlim(*_xlim)

# for sigma in [1]:
#     # pdf = st.halfnorm.pdf(np.exp(θs), loc=0, scale=sigma)
#     pdf = st.halfnorm.pdf(θs, loc=0, scale=sigma)
#     plt.plot(θs, pdf / max(pdf))
```

```python
import aesara
```

```python
?funnel.recompute_initial_point()
```

```python
aesara.function([funnel.θ], prob.rvs_to_values[funnel.θ].tag.transform.forward(funnel.θ))(3)
```

```python
aesara.function([prob.rvs_to_values[funnel.θ]], [prob.rvs_to_values[funnel.x]])
```

```python
plt.plot(prob.sample_x_z(np.random.RandomState(), np.exp(3))[1])
plt.plot(prob_jax.sample_x_z(jax.random.PRNGKey(1), 3)[1])
```

```python
result = MuseResult()
result = prob.solve(8, result=result, α=0.01, rng=np.random.SeedSequence(1), θ_rtol=0, gradz_logLike_atol=1e-6, progress=True, maxsteps=100)
```

```python
plt.hist([x[0,0] for x in result.history[1]["g_like_sims"]]);
```

```python
plt.semilogy([-1/np.squeeze(h["h_inv_like_sims"]) for h in result.history], label="like")
plt.semilogy([-np.squeeze(h["H_prior"]) for h in result.history], label="prior")
plt.semilogy([-1/np.squeeze(h["H_inv_post"]) for h in result.history], label="post")
plt.legend()
plt.axhline(0,c="k",ls="--")
plt.yscale("symlog", linthresh=1e-4)
```

```python
plt.plot([h["θ"] for h in result.history], "-")
```

```python
plt.plot([np.squeeze(h["g_post"]) for h in result.history], "C1--", label="post")
plt.plot([np.squeeze(h["g_prior"]) for h in result.history], "C2-", label="prior")
plt.plot([np.squeeze(h["g_like"]) for h in result.history], "C3-", label="like")
plt.yscale("symlog", linthresh=1e-2)
plt.legend(ncol=3)
plt.axhline(0,c="k",ls="--")
```

```python
az.plot_posterior(chain, var_names=['θ'])
# plt.axvline(np.exp(result.θ))
plt.axvline(result.θ)
```

```python
# with pm.Model() as model:
#     θ = pm.HalfNormal("θ", sigma=1000)
#     chain = pm.sample_prior_predictive(10000)

# plt.hist(chain.prior.θ[0], bins=30)
```

```python

```
