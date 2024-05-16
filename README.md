# AutomaticMALA

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://torfjelde.github.io/AutomaticMALA.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://torfjelde.github.io/AutomaticMALA.jl/dev/)
[![Build Status](https://github.com/torfjelde/AutomaticMALA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/torfjelde/AutomaticMALA.jl/actions/workflows/CI.yml?query=branch%3Amain)

This is a simple implementation of [`"AutoMALA: Locally Adaptive Metropolis-Adjusted Langevin Algorithm" by Biron-Lattes, Surjanovic & Syed et al. (2023)`](https://arxiv.org/abs/2310.16782), i.e. MALA but with a step size that is allowed to change in a reversible manner.

Note that this is mainly meant as a demonstration of how to implement samplers in [AbstractMCMC.jl](https://github.com/TuringLang/AbstractMCMC.jl) and rather than a production ready sampler implementation.
