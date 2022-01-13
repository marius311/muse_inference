.. muse_inference documentation master file, created by
   sphinx-quickstart on Wed Jan 12 15:22:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

muse_inference
==============

.. image:: https://img.shields.io/badge/documentation-latest-blue.svg
   :target: https://cosmicmar.com/muse_inference

.. image:: https://img.shields.io/badge/source-github-blue
   :target: https://github.com/marius311/muse_inference

.. image:: https://github.com/marius311/muse_inference/actions/workflows/doctests.yml/badge.svg
   :target: https://github.com/marius311/muse_inference/actions/workflows/doctests.yml

The Marginal Unbiased Score Expansion (MUSE) method is a generic tool for hierarchical Bayesian inference. MUSE performs approximate marginalization over arbitrary non-Gaussian and high-dimensional latent spaces, providing Gaussianized constraints on hyper parameters of interest. It is much faster than exact methods like Hamiltonian Monte Carlo (HMC), and requires no user input like many Variational Inference (VI), and Likelihood-Free Inference (LFI) or Simulation-Based Inference (SBI) methods. It excels in high-dimensions, which challenge these other methods. It is approximate, so its results may need to be spot-checked against exact methods, but it is itself exact in asymptotic limit of a large number of data modes contributing to each hyperparameter, or in the limit of Gaussian joint likelihood regardless the number of data modes. For more details, see `Millea & Seljak, 2021 <https://arxiv.org/abs/2112.09354>`_.


MUSE works on standard hierarchical problems, where the likelihood is of the form:

.. math::
   \mathcal{P}(x\,|\,\theta) = \int {\rm d}z \, \mathcal{P}(x,z\,|\,\theta)


In our notation, :math:`x` are the observed variables (the "data"), :math:`z` are unobserved "latent" variables, and :math:`\theta` are some "hyperparameters" of interest. MUSE is applicable when the goal of the analysis is to estimate the hyperparameters, :math:`\theta`, but otherwise, the latent variables, :math:`z`, do not need to be inferred (only marginalized out via the integral above). 

The only requirements to run MUSE on a particular problem are that forward simulations from :math:`\mathcal{P}(x,z\,|\,\theta)` can be generated, and gradients of the joint likelihood, :math:`\mathcal{P}(x,z\,|\,\theta)` with respect to :math:`z` and :math:`\theta` can be computed. The marginal likelihood is never required, so MUSE could be considered a form of LFI/SBI. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   demo


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |br| raw:: html

      <br>
