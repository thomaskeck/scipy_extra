#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.stats
import scipy.special

def acticvate_post_mortem_debugger():
    import sys 

    def info(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type, value, tb) 
        else:
            import traceback, pdb 
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type, value, tb) 
            # ...then start the debugger in post-mortem mode.
            pdb.post_mortem(tb)

    sys.excepthook = info

acticvate_post_mortem_debugger()


def get_shape_parameters(distribution):
    shape_parameters = []
    if distribution.shapes is not None:
        shape_parameters += distribution.shapes.split(', ')
    shape_parameters += ['loc', 'scale']
    return shape_parameters


class linear_model_gen(scipy.stats.rv_continuous):
    def __init__(self, distributions, *args, **kwargs):
        """
        distributions: {'component': scipy.stats.rv_continuous}
        """
        self.distributions = distributions
        self.components = distributions.keys()
        self.distribution_norms = []
        self.distribution_shapes = []
        for component, distribution in distributions.items():
            self.distribution_norms.append('{}_norm'.format(component))
            self.distribution_shapes.append(['{}_{}'.format(component, s) for s in get_shape_parameters(distribution)])
        kwargs['shapes'] = ', '.join(sum(self.distribution_shapes, self.distribution_norms))
        super(linear_model_gen, self).__init__(*args, **kwargs)

    def extract_positional_arguments(self, parameters):
        norm_values, rest = parameters[:len(self.distribution_norms)], parameters[len(self.distribution_norms):]
        shape_values = []
        for shape in self.distribution_shapes:
            shape_values.append(parameters[:len(shape)])
            parameters = parameters[len(shape):]
        return norm_values, shape_values

    def _pdf(self, x, *args):
        norm_values, shape_values = self.extract_positional_arguments(args)
        return np.sum(norm * distribution.pdf(x, *shape) for norm, distribution, shape in zip(norm_values, self.distributions.values(), shape_values))
    
    def _cdf(self, x, *args):
        norm_values, shape_values = self.extract_positional_arguments(args)
        return np.sum(norm * distribution.cdf(x, *shape) for norm, distribution, shape in zip(norm_values, self.distributions.values(), shape_values))
    
    def _rvs(self, *args):
        norm_values, shape_values = self.extract_positional_arguments(args)
        choices = np.random.choice(len(norm_values), size=self._size, p=np.array(norm_values) / np.sum(norm_values))
        result = np.zeros(self._size)
        for i, (distribution, shape) in enumerate(zip(self.distributions.values(), shape_values)):
            mask = choices == i
            result[mask] = distribution.rvs(size=mask.sum(), *shape)
        return result

    def _updated_ctor_param(self):
        dct = super(linear_model_gen, self)._updated_ctor_param()
        dct['distributions'] = self.distributions
        return dct

    def _argcheck(self, *args):
        return 1


class template_gen(scipy.stats.rv_continuous):
    def __init__(self, histogram, *args, **kwargs):
        self.histogram = histogram
        pdf, bins = self.histogram
        pdf = pdf / float(np.sum(pdf))
        cdf = np.cumsum(pdf)[1:]
        self.template_bins = bins
        self.template_pdf = np.hstack([0.0, pdf, 0.0])
        self.template_cdf = np.hstack([0.0, cdf, 1.0])
        self.template_bin_centers = (bins - (bins - np.roll(bins, 1)) / 2.0) [1:]
        super(template_gen, self).__init__(*args, **kwargs)

    def _pdf(self, x):
        return self.template_pdf[np.digitize(x, bins=self.template_bins)]
    
    def _cdf(self, x):
        return self.template_cdf[np.digitize(x, bins=self.template_bins)]
    
    def _rvs(self):
        probabilities = self.template_pdf[1:-1]
        probabilities /= probabilities.sum()
        choices = np.random.choice(len(self.template_pdf) - 2, size=self._size, p=probabilities)
        return self.template_bin_centers[choices]
    
    def _updated_ctor_param(self):
        dct = super(template_gen, self)._updated_ctor_param()
        dct['histogram'] = self.histogram
        return dct


class crystalball_gen(scipy.stats.rv_continuous):
    def _pdf(self, alpha, n):
        A = (n / np.abs(alpha) ) ** n * np.exp( - alpha ** 2 / 2.0 )
        B = n / np.abs(alpha)  - np.abs(alpha)
        C = n / np.abs(alpha) * 1.0 / (n - 1.0) * np.exp( - alpha ** 2 / 2.0 )
        D = np.sqrt(np.pi / 2.0) * (1 + scipy.special.erf(np.abs(alpha) / np.sqrt(2) ))
        N = 1.0 / (C + D)
        return np.where(x > -alpha, np.exp(- x**2 / 2), A * (B - x) ** (-n))
		
crystalball = crystalball_gen(name='crystalball')
