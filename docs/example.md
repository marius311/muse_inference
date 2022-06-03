---
jupyter:
  jupytext:
    formats: md,ipynb
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

# Example


This package allows defining your inference problem with:

* the [PyMC](https://docs.pymc.io/) probabilistic programming language
* by coding up the posterior by-hand and using [Jax](https://jax.readthedocs.io/) to automatically compute the necessary posterior gradients
* by specifying the posterior and its gradients completely by-hand

We'll start withPyMC, since it is the easiest. First, load up the relevant packages:

```python
%pylab inline
from scipy import stats
import pymc as pm
from ttictoc import tic, toc
from muse_inference.pymc import PyMCMuseProblem
```

```python nbsphinx="hidden" tags=[]
%config InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 110}
%load_ext autoreload
%autoreload 2
import logging
logging.getLogger("pymc").setLevel(logging.FATAL)
```

As an example, consider the following hierarchical problem, which has the classic [Neal's Funnel](https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html) problem embedded in it. Neal's funnel is a standard example of a non-Gaussian latent space which HMC struggles to sample efficiently without extra tricks. Specifically, we consider the model defined by:

$$
\begin{aligned}
\theta &\sim {\rm Normal(0,3)} \\ 
z_i &\sim {\rm Normal}(0,\exp(\theta/2)) \\ 
x_i &\sim {\rm Normal}(z_i, 1)
\end{aligned}
$$

for $i=1...2048$. This problem can be described by the following PyMC model:

```python
def gen_funnel(x=None, θ=None, rng=None):
    with pm.Model(rng_seeder=rng) as funnel:
        θ = pm.Normal("θ", mu=0, sigma=3) if θ is None else θ
        z = pm.Normal("z", mu=0, sigma=np.exp(θ/2), size=10000)
        x = pm.Normal("x", mu=z, sigma=1, observed=x)
    return funnel
```

Next, lets choose a true value of $\theta=0$ and generate some simulated data, $x$, which we'll use as "observations":

```python
rng = np.random.RandomState(0)
x_obs = pm.sample_prior_predictive(1, model=gen_funnel(θ=0, rng=rng)).prior.x[0,0]
model = gen_funnel(x=x_obs, rng=rng)
```

We can run HMC on the problem to compute the "true" answer to compare against:

```python
with model:
    tic()
    chain = pm.sample(500, tune=500, cores=1, chains=1, discard_tuned_samples=False)
    t_hmc = toc()
```

We next compute the MUSE estimate for the same problem. To reach the same Monte Carlo error as HMC, the number of MUSE simulations should be the same as the effective sample size of the chain we just ran. This is:

```python
nsims = int(pm.ess(chain)["θ"])
nsims
```

Running the MUSE estimate, 

```python
prob = PyMCMuseProblem(model)
tic()
result = prob.solve(θ_start=0, nsims=nsims, rng=np.random.SeedSequence(1), progress=True, save_MAP_history=True)
t_muse = toc()
```

Now lets plot the two estimates. In this case, MUSE gives a nearly perfect answer at a fraction of the computational cost.


```python
sum([soln.nfev for h in result.history for soln in [h["MAP_history_dat"]] + h["MAP_history_sims"]])
```

```python
sum(chain.sample_stats["n_steps"]) + sum(chain.warmup_sample_stats["n_steps"])
```

```python
figure(figsize=(6,5))
axvline(0, c="k", ls="--", alpha=0.5)
hist(
    chain["posterior"]["θ"].to_series(), 
    bins=30, density=True, alpha=0.5, color="C0",
    label="NUTS (%.2fs)"%t_hmc
)
θs = linspace(*xlim())
plot(
    θs, stats.norm(result.θ, sqrt(result.Σ[0,0])).pdf(θs), 
    color="C1", label="MUSE (%.2fs)"%t_muse
)
legend()
xlabel(r"$\theta$")
ylabel(r"$\mathcal{P}(\theta\,|\,x)$")
title("10000-dimensional noisy funnel");
```

The timing difference is indicative of the speedups over HMC that are possible. These get even more dramatic as we increase dimensionality, and 1-3 orders of magnitude are not atypical for high-dimensional problems.

