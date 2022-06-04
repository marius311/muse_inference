---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: 'Python 3.9.7 (''.venv'': poetry)'
    language: python
    name: python3
---

# User API

```python nbsphinx="hidden" tags=[]
%config InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 110}
%load_ext autoreload
%autoreload 2
import logging, warnings
logging.getLogger("pymc").setLevel(logging.FATAL)
warnings.filterwarnings("ignore")
```

## PyMC


The [Example](example.html) page introduces how to use *muse-inference* for a problem defined with PyMC. Here we consider a more complex problem to highlight additional features. In particular:

* We can estimate any number of parameters with any shapes. Here we have a 2-dimensional array $\mu$ and a scalar $\theta$. Note that by default, *muse-inference* considers any variables which do not depend on others as "parameters" (i.e. the "leaves" of the probabilistic graph). However, the algorithm is not limited to such parameters, and any choice can be selected by providing a list of `params` to the `PyMCMuseProblem` constructor.

* We can work with distributions with limited domain support. For example, below we use the $\rm Beta$ distribution with support on $(0,1)$ and the $\rm LogNormal$ distribution with support on $(0,\infty)$. All necessary transformations are handled internally.

* The data and latent space can include any number of variables, with any shapes. Below we demonstrate an $x$ and $z$ which are 2-dimensional arrays. 

First, load the relevant packages:

```python
%pylab inline
import pymc as pm
from muse_inference.pymc import PyMCMuseProblem
```

Then define the problem,

```python
def gen_funnel(x=None, θ=None, μ=None, rng=None):
    with pm.Model(rng_seeder=rng) as model:
        μ = pm.Beta("μ", 2, 5, size=2) if μ is None else μ
        θ = pm.Normal("θ", 0, 3) if θ is None else θ
        z = pm.LogNormal("z", μ, np.exp(θ/2), size=(100, 2))
        x = pm.Normal("x", z, 1, observed=x)
    return model
```

generate the model and some data, given some chosen true values of parameters,

```python
params_true = dict(μ=[0.3, 0.7], θ=1)
x_obs = pm.sample_prior_predictive(1, gen_funnel(rng=RandomState(0), **params_true)).prior.x[0,0]
model = gen_funnel(x=x_obs)
prob = PyMCMuseProblem(model)
```

and finally, run MUSE:

```python
params_start = dict(μ=[0.5, 0.5], θ=0)
result = prob.solve(params_start, progress=True)
```

When there are multiple parameters, the starting guess should be specified as as a dictionary, as above.

The solution is returned as a 1-dimensional vector of all parameters concatenated in the order they appear in the model:

```python
result.θ, result.Σ
```

## Jax


We can also use [Jax](https://jax.readthedocs.io/) to define the problem. In this case we will write out function to generate forward samples and to compute the posterior, and Jax will provide necessary gradients for free. To use Jax, load the necessary packages:

```python
from functools import partial
import jax
import jax.numpy as jnp
from muse_inference.jax import JittableJaxMuseProblem, JaxMuseProblem
from muse_inference import XZSample
```

Let's implement the noisy funnel problem from the [Example](example.html) page. To do so, extend either `JaxMuseProblem`, or, if your code is able to be JIT compiled by Jax, extend `JittableJaxMuseProblem` and decorate the functions with `jax.jit`:

```python
class JaxFunnelMuseProblem(JittableJaxMuseProblem):

    def __init__(self, N):
        super().__init__()
        self.N = N

    @partial(jax.jit, static_argnums=0)
    def sample_x_z(self, key, θ):
        keys = jax.random.split(key, 2)
        z = jax.random.normal(keys[0], (self.N,)) * jnp.exp(θ/2)
        x = z + jax.random.normal(keys[1], (self.N,))
        return XZSample(x, z)

    @partial(jax.jit, static_argnums=0)
    def logLike(self, x, z, θ):
        return -(jnp.sum((x - z)**2) + jnp.sum(z**2) / jnp.exp(θ) + 512*θ) / 2

    @partial(jax.jit, static_argnums=0)
    def logPrior(self, θ):
        return -θ**2 / (2*3**2)
```

Now generate some simulated data, which we set into `prob.x`. Note also the use of `PRNGKey` (rather than `RandomState` for PyMC/Numpy) for random number generation. 

```python
prob = JaxFunnelMuseProblem(10000)
key = jax.random.PRNGKey(0)
(x, z) = prob.sample_x_z(key, jnp.array([1.]))
prob.x = x
```

And finally, run MUSE:

```python
result = prob.solve(θ_start=0., rng=jax.random.PRNGKey(1), progress=True)
```

Note that the solution here is obtained around 10X faster that the PyMC version of this in the [Example](example.html) page. The Jax interface has much lower overhead, which will be noticeable for very fast posteriors like the one above. 


One powerful aspect of using Jax is that the parameters, `θ`, and latent space, `z`, can be any [pytree](https://jax.readthedocs.io/en/latest/pytrees.html), ie tuples, dictionaries, nested combinations of them, etc... (there is no requirement on the data format of the `x` variable). To demonstrate, consider a problem which is just two copies of the noisy funnel problem:

```python
class JaxPyTreeFunnelMuseProblem(JittableJaxMuseProblem):

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
        return XZSample(x={"x1":x1, "x2":x2}, z={"z1":z1, "z2":z2})

    @partial(jax.jit, static_argnums=0)
    def logLike(self, x, z, θ):
        return (
            -(jnp.sum((x["x1"] - z["z1"])**2) + jnp.sum(z["z1"]**2) / jnp.exp(θ["θ1"]) + 512*θ["θ1"]) / 2
            -(jnp.sum((x["x2"] - z["z2"])**2) + jnp.sum(z["z2"]**2) / jnp.exp(θ["θ2"]) + 512*θ["θ2"]) / 2
        )

    @partial(jax.jit, static_argnums=0)
    def logPrior(self, θ):
        return - θ["θ1"]**2 / (2*3**2) - θ["θ2"]**2 / (2*3**2)
```

Here, `x`, `θ`, and `z` are all dictionaries. We generate the problem as usual, passing in parameters as dictionaries,

```python
θ_true = dict(θ1=-1., θ2=2.)
θ_start = dict(θ1=0., θ2=0.)
```

```python
prob = JaxPyTreeFunnelMuseProblem(10000)
key = jax.random.PRNGKey(0)
(x, z) = prob.sample_x_z(key, θ_true)
prob.x = x
```

and run MUSE:

```python
result = prob.solve(θ_start=θ_start, rng=jax.random.PRNGKey(0), progress=True)
```

The result is returned as a dictionary:

```python
result.θ
```

and the covariance as a matrix:

```python
result.Σ
```
