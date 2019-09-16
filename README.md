# OptimizationLH

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/hendri54/OptimizationLH.jl.svg?branch=master)](https://travis-ci.com/hendri54/OptimizationLH.jl)
[![codecov.io](http://codecov.io/github/hendri54/OptimizationLH.jl/coverage.svg?branch=master)](http://codecov.io/github/hendri54/OptimizationLH.jl?branch=master)

## Purpose

This is a playground for optimization algorithms.

A simple discrete choice model with preference shocks is used to approximate the kind of model an economist might want to solve, but in a form that is inexpensive to solve. 

Simulated moments are compared with target moments during the calibration.

## Model

Desirable features:

1. The number of calibrated parameters is variable.
2. The objective is computed from simulated choices, which mimics the non-smoothness of the typical economic model.

The model is static. There are `N` households, indexed by `j`, who choose from `M` choices.

A household is endowed with a vector of `K` endowments. The first endowment `e(1,j)` is normalized so that it increases linearly in `j` from 0 to 1.

The other endowments are computed as ``e(k,j) = \alpha_k + \beta_k * e(1,j) ^ 2``.

The choices are also endowed with `K` endowments that are constructed analogously. `f(1, m)` is linear in `m`. ``f(k, m) = \gamma_k + \delta_k * f(1, m)``.

The payoff from choosing choice `m` is given by

``u(j, m) = \sum_k e(k, j) * f(k, m)``

The choice of `m` is subject to a preference shock so that 

``Pr(m | j) = u(j, m) / \sum_i u(j, i)``

The objective is to match the fraction of households in each endowent percentile class that chooses each `m` percentile group (e.g. quartile).

The calibrated parameters are `\alpha_k`, `\beta_k`, `\gamma_k`, `\delta_k`.

-----------