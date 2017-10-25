#!/usr/bin/env python3

import numpy as np
import scipy.optimize
import collections
import functools


class Fitter(object):
    """
    Implements extended unbinned maximum likelihood fits for scipy.stats distributions.
    The distribution has to be implemented completly using scipy.stats distributions.
    A mapping function has to be provided, which maps the free parameters of the fit to the scipy.stats shape parameters.
    The fit itself is performed using scipy.optimize.
    """
    def __init__(self, mapping, distributions, normalisation=None, method='nelder-mead', ugly_and_fast=False):
        """
        Parameters
        ----------
        mapping : user-defined function, which maps the free parameters to the shape parameters of the distribution.
                  For a one-dimensional fit the function maps np.array(free-parameters) -> dict(shape parameters)
                  For a multi-dimenional fit the function maps np.array(free-parameters) -> list of dict(shape parameters) for each distribution.
        distributions : A scipy.stats distribution (implies a one-dimensional fit) or a list of scipy.stats distributions (implies a multi-dimensional fit).
        normalisation : user-defined function, which maps the shape parameters of a distribution (returned by the mapping) to the overall norm of the distribution.
                        If None is given, the norm 1.0 is assumed, which reduced the fit to an unbinned M.L fit, instead of a extended unbinned M.L fit.
        method : The method passed to scipy.optimize.minimize
        ugly_and_fast : If true, the calculation of the uncertainty and likelihood profile will speed-up, but loose accuracy.
                        This is achieved by just evaluating the loss-function with the optimal-parameters, instead of fitting all parameters again.
        """
        self.is_multi_dimensional = isinstance(distributions, collections.Sequence)
        self.mapping = mapping if self.is_multi_dimensional else lambda p: [mapping(p)]
        self.distributions = distributions if self.is_multi_dimensional else [distributions]
        self.method = method
        self.normalisation = normalisation
        self.ugly_and_fast = ugly_and_fast
        self.r = None

    def loss(self, free_parameters, data, weights, mapping):
        """
        Calculates the extended unbinned maximum likelihood fit.
        It is assumed that the pdf of the distributions is normed (integral over the whole range is one).
        """
        loss = 0.0 
        parameters = mapping(free_parameters)
        for d, w, p, distribution in zip(data, weights, parameters, self.distributions):
            N = np.sum(w)
            norm = 1.0 if self.normalisation is None else self.normalisation(p)
            average_number_of_events = norm * N
            loss += - N * np.log(average_number_of_events) + average_number_of_events
            loss += - np.sum(w * np.log(distribution.pdf(d, **p)))
        return loss

    def _ensure_dimension(self, data, weights):
        if not self.is_multi_dimensional:
            data = [data]
        if not self.is_multi_dimensional:
            if weights is not None:
                weights = [weights]
        if weights is None:
            weights = [np.ones(len(d)) for d in data]
        return data, weights

    def _fit(self, initial_parameters, data, weights, mapping):
        r = scipy.optimize.minimize(self.loss, initial_parameters, args=(data, weights, mapping), method=self.method)
        return r

    def fit(self, initial_parameters, data, weights=None):
        self.r = self._fit(initial_parameters, *self._ensure_dimension(data, weights), self.mapping)
        return self.r

    def _get_likelihood_profile_function(self, optimal_parameters, fixed):
        optimal_parameters_without_fixed = [p for i, p in enumerate(optimal_parameters) if i not in fixed]
        insert_fixed_parameters = lambda p, f: functools.reduce(lambda l, i: l[:i[1]] + [f[i[0]]] + l[i[1]:], enumerate(fixed), list(p))
        if self.ugly_and_fast:
            return lambda x, data, weights: self.loss(insert_fixed_parameters(optimal_parameters_without_fixed, x), data, weights, self.mapping)
        else:
            return lambda x, data, weights: self._fit(optimal_parameters_without_fixed, data, weights, lambda p: self.mapping(np.array(insert_fixed_parameters(p, x)))).fun
    
    def get_uncertainties(self, parameter_positions, parameter_boundaries, data, weights=None):
        if self.r is None:
            raise RuntimeError("Please call fit first")
        data, weights = self._ensure_dimension(data, weights)
        uncertainties = []
        for i, (lower_boundary, upper_boundary) in zip(parameter_positions, parameter_boundaries):
            likelihood_profile_function = self._get_likelihood_profile_function(list(self.r.x), [i])
            lower = None if lower_boundary is None else scipy.optimize.brentq(lambda x: likelihood_profile_function([x], data, weights) - (self.r.fun + 0.5), lower_boundary, self.r.x[i])
            upper = None if upper_boundary is None else scipy.optimize.brentq(lambda x: likelihood_profile_function([x], data, weights) - (self.r.fun + 0.5), self.r.x[i], upper_boundary)
            uncertainties.append([lower, upper])
        return uncertainties
    
    def likelihood_profile(self, parameter_positions, parameters_generator, data, weights=None):
        if self.r is None:
            raise RuntimeError("Please call fit first")
        data, weights = self._ensure_dimension(data, weights)
        likelihood_profile_function = self._get_likelihood_profile_function(list(self.r.x), parameter_positions)
        return np.array([likelihood_profile_function(parameter_values, data, weights) for parameter_values in parameters_generator])

    def stability_test(self, initial_parameters, true_parameters_generator, size):
        result = []
        for true_parameters in true_parameters_generator:
            parameters = self.mapping(true_parameters)
            data = []
            for p, distribution in zip(parameters, self.distributions):
                data.append(distribution.rvs(size=size, **p))
            weights = [np.ones(len(d)) for d in data]
            r = self._fit(initial_parameters, data, weights, self.mapping)
            result.append(r)
        return result
    
    def get_significance(self, parameter_positions, parameter_values, data, weights=None):
        if self.r is None:
            raise RuntimeError("Please call fit first")
        data, weights = self._ensure_dimension(data, weights)
        likelihood_profile_function = self._get_likelihood_profile_function(list(self.r.x), parameter_positions)
        n = likelihood_profile_function(parameter_values, data, weights)
        return np.sqrt(2*(n-self.r.fun))


def template_test(fit_model, sample, x_range=(-2, 4)):
    """
    Performs Kolmogorov-Test to compare a datasamples with distributions of fit-model components
    @param fit_model: scipy_extra Fit-Model
    @param sample: Sample object. For every component of fit-model there is a pandas dataframe in sample
    @param x_range:
    @return:
    """
    kolmorogov = {}
    X = np.linspace(x_range[0], x_range[1], 300)
    for name, _, distribution in fit_model.get_frozen_components():
        df = getattr(sample, name)
        if len(df) == 0:
            kolmorogov[name] = (0, 0)
            continue
        d = max(abs(_empiric_cdf(x, df) - distribution.cdf(x)) for x in X)
        p = 0
        for i in range(1, 1000):
            p += -1**(i - 1) * np.exp(-2 * i**2 * (np.sqrt(len(df)) * d)**2)
        p *= 2
        kolmorogov[name] = (p, d)
    return kolmorogov


def _empiric_cdf(x, data):
    return len(data[data < x]) / len(data)

