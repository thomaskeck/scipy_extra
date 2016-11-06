#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.optimize
import scipy.stats

from . import stats

from multiprocessing import Pool
from itertools import cycle


class Model(object):
    def __init__(self, name):
        self.name = name
        self.names = []
        self.distributions = []
        self.parameters = {}
        self.constraints = []
      
    @property
    def distribution(self):
        return stats.mixture_gen(dict(zip(self.names, self.distributions)), name=self.name)
    
    @property
    def frozen_distribution(self):
        return stats.mixture_gen(dict(zip(self.names, self.distributions)), name=self.name)(**self.parameters)
    
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
         for key, value in parameters:
             self.parameters['{}_{}'.format(name, key)] = value
         if loc is None:
             self.add_constraint('{}_loc'.format(name), lambda x: 0.0)
         if scale is None:
             self.add_constraint('{}_scale'.format(name), lambda x: 1.0)

    def get_frozen_components(self):
        for name, distribution in zip(self.names, self.distributions):
            shapes = [self.parameters['{}_{}'.format(name, s)] for s in stats.get_shape_parameters(distribution)]
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

    def fix_parameter(self, parameter, priority=10):
        self.constraints = [(key, value) for key, value in self.constraints if key != parameter]
        self.constraints = [(parameter, lambda p: self.parameters[parameter])] + self.constraints
        self.set_parameters(**{parameter: self.parameters[parameter]})

    def get_free_parameters(self):
        fixed_parameters = [c[0] for c in self.constraints]
        return {parameter: value for parameter, value in self.parameters.items() if parameter not in fixed_parameters}
    
    def get_fixed_parameters(self):
        fixed_parameters = [c[0] for c in self.constraints]
        return {parameter: value for parameter, value in self.parameters.items() if parameter in fixed_parameters}


def _likelihood_profile_task(args):
    self, parameters, values, fit_model, data = args
    fit_model.set_parameters(**{parameter: value for parameter, value in zip(parameters, values)})
    _, r = self.fit(fit_model, data)
    return r.fun


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

    def _unbinned_maximum_likelihood_loss(self, free_parameters, data, fit_model):
        parameters = self._get_current_parameters(free_parameters, fit_model)
        return -np.sum(np.log(fit_model.distribution.pdf(data, **parameters)))
    
    def _extended_unbinned_maximum_likelihood_loss(self, free_parameters, data, fit_model):
        parameters = self._get_current_parameters(free_parameters, fit_model)
        average_number_of_events = np.sum([parameters['{}_norm'.format(name)] for name in fit_model.names])
        return self._unbinned_maximum_likelihood_loss(free_parameters, data, fit_model) + average_number_of_events

    def fit(self, fit_model, data):
        initial_parameters = np.array(list(fit_model.get_free_parameters().values()))
        r = scipy.optimize.minimize(self.loss, initial_parameters, args=(data, fit_model), method=self.method)
        parameters = dict(zip(fit_model.get_free_parameters().keys(), r.x))
        for parameter, function in fit_model.constraints:
            parameters[parameter] = function(parameters)
        return parameters, r

    def likelihood_profile(self, fit_model, parameter_spaces, data):
        parameters = parameter_spaces.keys()
        linspaces = parameter_spaces.values()

        best_fit_parameters, _ = self.fit(fit_model, data)
        fit_model.set_parameters(**best_fit_parameters)

        for parameter in parameters:
            fit_model.fix_parameter(parameter)

        #with Pool(processes=4) as pool:
        arguments = list(zip(cycle([self]), cycle([parameters]), zip(*linspaces), cycle([fit_model]), cycle([data])))
        results = list(map(_likelihood_profile_task, arguments))
        return results

    def stability_test(self, fit_model, parameter_spaces, data_size=10000):
        #with Pool(processes=4) as pool:
        parameters = parameter_spaces.keys()
        linspaces = parameter_spaces.values()
        arguments = list(zip(cycle([self]), cycle([parameters]), zip(*linspaces), cycle([fit_model]), cycle([data_size])))
        results = list(map(_stability_test_task, arguments))
        return {parameter: values for parameter, *values in zip(parameters, *results)}

