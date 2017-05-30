#!/usr/bin/env python3

import math
import numpy as np
import scipy
import scipy.optimize
import scipy.stats

from . import stats

from multiprocessing import Pool
from itertools import cycle


class LambdaReplacement():
    def __init__(self, dictionary_or_value, parameter=None):
        if parameter is None:
            self.return_value = dictionary_or_value
        else:
            self.dictionary = dictionary_or_value
            self.parameter = parameter
            self.return_value = None
    def __call__(self, p):
        if self.return_value is not None:
            return self.return_value
        else:
            return self.dictionary[self.parameter]


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
            p += pow(-1, i - 1) * math.exp(-2 * pow(i, 2) * pow(math.sqrt(len(df)) * d, 2))
        p *= 2
        kolmorogov[name] = (p, d)
    return kolmorogov


def _empiric_cdf(x, data):
    return len(data[data < x]) / len(data)


class Model(object):
    def __init__(self, name):
        self.name = name
        self.names = []
        self.distributions = []
        self.parameters = {}
        self.constraints = []
      
    @property
    def distribution(self):
        return stats.mixture_gen(list(zip(self.names, self.distributions)), name=self.name)
    
    @property
    def frozen_distribution(self):
        return stats.mixture_gen(list(zip(self.names, self.distributions)), name=self.name)(**self.parameters)
    
    @property
    def norm(self):
        return np.sum([self.parameters['{}_norm'.format(name)] for name in self.names])

    def __iter__(self):
        return iter(dict(zip(self.names, self.distributions)))

    def add_component(self, name, distribution, norm, loc=None, scale=None, **parameters):
         if name in self.names:
             raise RuntimeError("Another component with the same name was already added: " + name)
         self.names.append(name)
         self.distributions.append(distribution)
         self.parameters['{}_norm'.format(name)] = norm
         self.parameters['{}_loc'.format(name)] = loc
         self.parameters['{}_scale'.format(name)] = scale
         for key, value in parameters.items():
             self.parameters['{}_{}'.format(name, key)] = value
         if loc is None:
             self.add_constraint('{}_loc'.format(name), LambdaReplacement(0.0))
         if scale is None:
             self.add_constraint('{}_scale'.format(name), LambdaReplacement(1.0))

    def get_frozen_components(self):
        for name, distribution in zip(self.names, self.distributions):
            shapes = [self.parameters['{}_{}'.format(name, s)] for s in stats._get_shape_parameters(distribution)]
            yield name, self.parameters['{}_norm'.format(name)], distribution(*shapes)

    def set_parameters(self, **parameters):
        for parameter in parameters:
            if parameter in self.parameters:
                self.parameters[parameter] = parameters[parameter]
            else:
                print("WARNING: Setting an unkown parameter {}".format(parameter))
        for parameter, function in self.constraints:
            self.parameters[parameter] = function(self.parameters)

    def add_constraint(self, parameter, function):
        if parameter not in self.parameters:
            print("WARNING: Adding constraint for {} but this parameter is not in the model".format(parameter))
        self.constraints = [(key, value) for key, value in self.constraints if key != parameter]
        self.constraints.append((parameter, function))
        self.set_parameters(**{parameter: function(self.parameters)})

    def fix_parameter(self, parameter):
        self.constraints = [(key, value) for key, value in self.constraints if key != parameter]
        self.constraints = [(parameter, LambdaReplacement(self.parameters, parameter))] + self.constraints
        self.set_parameters(**{parameter: self.parameters[parameter]})

    def get_free_parameters(self):
        fixed_parameters = [c[0] for c in self.constraints]
        return {parameter: value for parameter, value in self.parameters.items() if parameter not in fixed_parameters}
    
    def get_fixed_parameters(self):
        fixed_parameters = [c[0] for c in self.constraints]
        return {parameter: value for parameter, value in self.parameters.items() if parameter in fixed_parameters}


def _likelihood_profile_task(args):
    self, parameters, values, fit_model, data, weights = args
    fit_model.set_parameters(**{parameter: value for parameter, value in zip(parameters, values)})
    _, r = self.fit(fit_model, data, weights)
    return r.fun


def _likelihood_uncert_task(args):
    self, parameter, value, fit_model, data, f, weights = args
    fit_model.set_parameters(**{parameter: value})
    _, r = self.fit(fit_model, data, weights)
    return f - r.fun


def _stability_test_task(args):
    # TODO Setting the parameters correctly for the fit is maybe to easy?
    self, parameters, values, fit_model, data_size = args
    fit_model.set_parameters(**{parameter: value for parameter, value in zip(parameters, values)})
    data = fit_model.frozen_distribution.rvs(size=data_size)
    fit_result, _ = self.fit(fit_model, data)
    return [fit_result[parameter] for parameter in parameters]


class Fitter(object):
    def __init__(self, loss='unbinned-maximum-likelihood', method='nelder-mead'):
        if loss == 'unbinned-maximum-likelihood':
            self.loss = self._unbinned_maximum_likelihood_loss
        elif loss == 'extended-unbinned-maximum-likelihood':
            self.loss = self._extended_unbinned_maximum_likelihood_loss
        else:
            raise RuntimeError("Unkown loss function named " + loss)
        self.method = method

    def _get_current_parameters(self, free_parameters, fit_model):
        parameters = {key: free_parameters[i] for i, key in enumerate(fit_model.get_free_parameters().keys())}
        for parameter, function in fit_model.constraints:
            if parameter in fit_model.parameters:
                parameters[parameter] = function(parameters)
        return parameters

    def _unbinned_maximum_likelihood_loss(self, free_parameters, data, fit_model, weights=None):
        parameters = self._get_current_parameters(free_parameters, fit_model)
        if weights is None:
            weights = np.ones(len(data))
        return -np.sum(weights * np.log(fit_model.distribution.pdf(data, **parameters)))
    
    def _extended_unbinned_maximum_likelihood_loss(self, free_parameters, data, fit_model, weights=None):
        parameters = self._get_current_parameters(free_parameters, fit_model)
        if weights is None:
            weights = np.ones(len(data))
        N = np.sum(weights)
        average_number_of_events = np.sum([max(parameters['{}_norm'.format(name)], 0.0) for name in fit_model.names]) * N
        return self._unbinned_maximum_likelihood_loss(free_parameters, data, fit_model, weights) - N * np.log(average_number_of_events) + average_number_of_events

    def fit(self, fit_model, data, weights=None):
        initial_parameters = np.array(list(fit_model.get_free_parameters().values()))
        if len(initial_parameters) == 0:
            parameters = {}
            r = scipy.optimize.OptimizeResult(x=np.array([]), fun=self.loss(np.array([]), data, fit_model, weights), hessian=None, success=True, status=0)
        else:
            r = scipy.optimize.minimize(self.loss, initial_parameters, args=(data, fit_model, weights), method=self.method)
            parameters = dict(zip(fit_model.get_free_parameters().keys(), r.x))
        for parameter, function in fit_model.constraints:
            parameters[parameter] = function(parameters)
        return parameters, r

    def likelihood_profile(self, fit_model, parameter_spaces, data, weights=None):
        parameters = parameter_spaces.keys()
        linspaces = parameter_spaces.values()
        best_fit_parameters, _ = self.fit(fit_model, data, weights)
        fit_model.set_parameters(**best_fit_parameters)

        for parameter in parameters:
            fit_model.fix_parameter(parameter)

        #with Pool(processes=4) as pool:
        arguments = list(zip(cycle([self]), cycle([parameters]), zip(*linspaces), cycle([fit_model]), cycle([data]), cycle([weights])))
        results = list(map(_likelihood_profile_task, arguments))
        return results

    def get_likelihood_uncertainty(self, parameters, a_opt, f_opt, fit_model, data, weights=None):
        """
        Get Likelhood uncertainties (optimum-likelihood-value + 1/2 gives asymmetric uncertainty for parameter a)
        Works fine with one free Parameter!
        @param parameters: (dict) dict with parameter_space
        @param a_opt: (array) parameter optimum (fit result)
        @param f_opt: (value) likelihood value of optimum
        @param fit_model:
        @param data:
        @return: result_list (list of tupel)
        """
        parameter_keys = parameters.keys()
        parameter_spaces = parameters.values()
        result_list = []
        for parameter_space, parameter_key, a_opt in zip(parameter_spaces, parameter_keys, a_opt):
            results_lower = scipy.optimize.brentq(lambda x: _likelihood_uncert_task((self, parameter_key, x, fit_model, data, f_opt + 0.5, weights)), parameter_space[0], a_opt)
            results_upper = scipy.optimize.brentq(lambda x: _likelihood_uncert_task((self, parameter_key, x, fit_model, data, f_opt + 0.5, weights)), a_opt, parameter_space[len(parameter_space)-1])
            result_list.append((results_upper, results_lower))
        return result_list

    def stability_test(self, fit_model, parameter_spaces, data_size=10000):
        #with Pool(processes=4) as pool:
        parameters = parameter_spaces.keys()
        linspaces = parameter_spaces.values()
        arguments = list(zip(cycle([self]), cycle([parameters]), zip(*linspaces), cycle([fit_model]), cycle([data_size])))
        results = list(map(_stability_test_task, arguments))
        return {parameter: values for parameter, *values in zip(parameters, *results)}

    def fit_significance(self, parameter_max, parameter_null, fit_model, data, weights=None):
        for key in fit_model.parameters.keys():
            fit_model.fix_parameter(key)
        fit_model.set_parameters(**parameter_max)
        L_min = self.loss(np.array([]), data, fit_model, weights)
        fit_model.set_parameters(**parameter_null)
        L_null = self.loss(np.array([]), data, fit_model, weights)
        return math.sqrt(2*abs((L_null-L_min)))

