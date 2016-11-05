#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.optimize
import scipy.stats

from . import stats

from multiprocessing import Pool

class Model(object):
    def __init__(self, name):
        self.name = name
        self.names = []
        self.distributions = []
        self.parameters = {}
        self.constraints = {}
      
    @property
    def distribution(self):
        return stats.linear_model_gen(dict(zip(self.names, self.distributions)), name=self.name)
    
    @property
    def frozen_distribution(self):
        return stats.linear_model_gen(dict(zip(self.names, self.distributions)), name=self.name)(**self.parameters)

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

    def add_constraint(self, parameter, function):
        if parameter not in self.parameters:
            print("WARNING: Adding constraint for {} but this parameter is not in the model".format(parameter))
        self.constraints[parameter] = function
        self.set_parameters(**{parameter: function(self.parameters)})

    def fix_parameter(self, parameter):
        self.add_constraint(parameter, lambda p: self.parameters[parameter])

    def get_free_parameters(self):
        return {parameter: value for parameter, value in self.parameters.items() if parameter not in self.constraints}
    
    def get_fixed_parameters(self):
        return {parameter: value for parameter, value in self.parameters.items() if parameter in self.constraints}


class Fitter(object):
    def __init__(self, loss='maximum-likelihood', method='nelder-mead'):
        if loss == 'maximum-likelihood':
            self.loss = self._maximum_likelihood_loss
        else:
            raise RuntimeError("Unkown loss function named " + loss)
        self.method = method

    def _maximum_likelihood_loss(self, free_parameters, data, fit_model):
        parameters = {key: free_parameters[i] for i, key in enumerate(fit_model.get_free_parameters().keys())}
        for parameter, function in fit_model.constraints.items():
            if parameter in fit_model.parameters:
                parameters[parameter] = function(parameters)
        return -np.sum(np.log(fit_model.distribution.pdf(data, **parameters)))


    def fit(self, fit_model, data):
        initial_parameters = np.array(list(fit_model.get_free_parameters().values()))
        r = scipy.optimize.minimize(self.loss, initial_parameters, args=(data, fit_model), method=self.method)
        parameters = dict(zip(fit_model.get_free_parameters().keys(), r.x))
        for parameter, function in fit_model.constraints.items():
            parameters[parameter] = function(parameters)
        return parameters, r

    def liklihood_profile(self, fit_model, data, parameter, linspace):
        parameters, _ = self.fit(fit_model, data)
        fit_model.set_parameters(parameters)
        fit_model.fix_parameter(parameter)

        def task(value):
            fit_model.set_parameters(parameter=value)
            _, r = self.fit(fit_model, data)
            return r

        with Pool(processes=4) as pool:
            results = pool.map(task, linspace)
        print(results)
        return results

    def stability_test(self, fit_model, parameter, linspace):
        def task(value):
            fit_model.set_parameters(parameter=value)
            data = fit_model.frozen_distribution.rvs(size=10000)
            parameters, _ = self.fit(fit_model, data)
            return parameters
        
        with Pool(processes=4) as pool:
            results = pool.map(task, linspace)
        print(results)
        return results

