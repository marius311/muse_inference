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

```python tags=[] nbsphinx="hidden"
%pylab inline
%config InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 110}
%load_ext autoreload
%autoreload 2
```

```python
import timeit

import jax
import jax.numpy as jnp
import muse_inference
import numpy as np
import pymc as pm
from muse_inference import MuseProblem, MuseResult
from muse_inference.jax import JaxMuseProblem, JittedJaxMuseProblem
from muse_inference.pymc import PyMCMuseProblem
```

```python
from julia.api import Julia
jl = Julia(compiled_modules=False)
%load_ext julia.magic
%julia using MuseInference, Random, Turing, Zygote, BenchmarkTools
```

```python
θ_true = 1.
θ_start = 0.
```

# Numpy

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

prob = NumpyFunnelMuseProblem(512)

rng = np.random.RandomState(0)
(x, z) = prob.sample_x_z(rng, θ_true)
prob.x = x
t_numpy = timeit.timeit("prob.solve(θ_start=θ_start, rng=np.random.SeedSequence(0), θ_rtol=0, gradz_logLike_atol=1e-2, progress=False, maxsteps=10)", globals=globals(), number=15)
```

# Jax

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
    
prob = JaxFunnelMuseProblem(512)
key = jax.random.PRNGKey(0)
(x, z) = prob.sample_x_z(key, θ_true)
prob.x = x
t_jax = timeit.timeit("prob.solve(θ_start=θ_start, rng=key, θ_rtol=0, gradz_logLike_atol=1e-2, progress=False, maxsteps=10)", globals=globals(), number=15)
```

# PyMC

```python
def gen_funnel(x=None, θ=None, rng_seeder=None):
    with pm.Model(rng_seeder=rng_seeder) as funnel:
        θ = θ if θ else pm.Normal("θ", 3)
        z = pm.Normal("z", 0, np.exp(θ / 2), size=512)
        x = pm.Normal("x", z, 1, observed=x)
    return funnel
        
# generated simulated data
rng = np.random.RandomState(0)
x_obs = pm.sample_prior_predictive(1, model=gen_funnel(θ=0.1, rng_seeder=rng)).prior.x[0,0]

# set up problem
funnel = gen_funnel(x_obs)
prob = PyMCMuseProblem(funnel)

rng = np.random.RandomState(0)
(x, z) = prob.sample_x_z(rng, θ_true)
prob.x = x
t_pymc = timeit.timeit("prob.solve(θ_start=θ_start, rng=np.random.SeedSequence(0), θ_rtol=0, gradz_logLike_atol=1e-2, progress=False, maxsteps=10)", globals=globals(), number=15)
```

# Turing

```julia
@model function funnel()
    θ ~ Normal(0, 3)
    z ~ MvNormal(zeros(512), exp(θ/2))
    x ~ MvNormal(z, 1)
end

Random.seed!(0)
x = (funnel() | (θ=$θ_true,))() # draw sample of `x` to use as simulated data
model = funnel() | (;x);

Turing.setadbackend(:zygote)
```

```pythons
t_turing = %julia 10 * @belapsed muse(model, py"θ_start"; θ_rtol=0, maxsteps=10, get_covariance=false, ∇z_logLike_atol=1e-2)
```

# Zygote

```julia
prob = MuseProblem(
    x,
    function sample_x_z(rng, θ)
        z = rand(rng, MvNormal(zeros(512), exp(θ/2)))
        x = rand(rng, MvNormal(z, 1))
        (;x, z)
    end,
    function logLike(x, z, θ)
        -(1//2) * (sum((x .- z).^2) + sum(z.^2) / exp(θ) + 512*θ)
    end,
    function logPrior(θ)
        -θ^2/(2*3^2)
    end,
    MuseInference.ZygoteBackend()
);
```

```python
t_zygote = %julia 10 * @belapsed muse(prob, py"θ_start"; θ_rtol=0, maxsteps=10, get_covariance=false, ∇z_logLike_atol=1e-2)
```

# Make plot

```python
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
```

```python
bar(
    range(5), 
    100*array([t_zygote, t_numpy, t_jax, t_pymc, t_turing]), 
    tick_label=["Zygote", "Numpy", "Jax", "PyMC", "Turing"],
    color=["C0","C1","C0","C2","C2"]
)
ylabel("runtime [ms]")
legend_elements = [
    Patch(facecolor='C1', label='Manual posterior / manual gradients'),
    Patch(facecolor='C0', label='Manual posterior / AD gradients'),
    Patch(facecolor='C2', label='PPL posterior / AD gradients'),
]
legend(handles=legend_elements, loc='upper left')
```
