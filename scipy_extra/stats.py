#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.stats
import scipy.special


def get_shape_parameters(distribution):
    """
    Returns all shape parameters of a scipy distribution
    including loc and scale
    @param distribution scipy.stats.rv_continuous
    """
    shape_parameters = []
    if distribution.shapes is not None:
        shape_parameters += distribution.shapes.split(', ')
    shape_parameters += ['loc', 'scale']
    return shape_parameters


class mixture_gen(scipy.stats.rv_continuous):
    """
    Mixture Distribution Generator

    Model = A * PDF_A + B * PDF_B + C * PDF_C + ...

    The shape parameters of this distribution are the
    the shape parameters of the individual sub-distributions
    with a prefix given by the component name.
    In addition there is one shape parameter per sub-distribution for the 
    relative normalisation of this component.
    E.g.

    mixture_gen({'Gauss': scipy.stats.norm, 'Gamma': scipy.stats.gamma})
    has the following shape parameters:
      - Gauss_norm
      - Gauss_loc
      - Gauss_scale
      - Gamma_norm
      - Gamma_loc
      - Gamma_scale
      - Gamma_a
    """
    def __init__(self, distributions, *args, **kwargs):
        """
        Create a new linear model given a number of distributions
        @param distributions: dict(string: scipy.stats.rv_continuous)
        """
        self.distributions = distributions
        self.components = distributions.keys()
        self.distribution_norms = []
        self.distribution_shapes = []
        for component, distribution in distributions.items():
            self.distribution_norms.append('{}_norm'.format(component))
            self.distribution_shapes.append(['{}_{}'.format(component, s) for s in get_shape_parameters(distribution)])
        kwargs['shapes'] = ', '.join(sum(self.distribution_shapes, self.distribution_norms))
        super(mixture_gen, self).__init__(*args, **kwargs)

    def _extract_positional_arguments(self, parameters):
        """
        scipy passes the arguments as positional arguments,
        this method splits up the individual positional arguments
        into the norm arguments and the shape arguments for each individual component
        """
        norm_values, parameters = parameters[:len(self.distribution_norms)], parameters[len(self.distribution_norms):]
        shape_values = []
        for shape in self.distribution_shapes:
            shape_values.append(parameters[:len(shape)])
            parameters = parameters[len(shape):]
        total_norm = np.sum(np.array(norm_values), axis=0)
        norm_values = [norm_value / total_norm for norm_value in norm_values]
        return norm_values, shape_values

    def _pdf(self, x, *args):
        """
        The combined PDF is the sum of the distribution PDFs weighted by their individual norm factors
        """
        norm_values, shape_values = self._extract_positional_arguments(args)
        return np.sum((norm * distribution.pdf(x, *shape) for norm, distribution, shape in zip(norm_values, self.distributions.values(), shape_values)), axis=0)
    
    def _cdf(self, x, *args):
        """
        The combined CDF is the sum of the distribution CDFs weighted by their individual norm factors
        """
        norm_values, shape_values = self._extract_positional_arguments(args)
        return np.sum((norm * distribution.cdf(x, *shape) for norm, distribution, shape in zip(norm_values, self.distributions.values(), shape_values)), axis=0)
    
    def _rvs(self, *args):
        """
        Generates random numbers using the individual distribution random generator
        with a probability given by their individual norm factors 
        """
        norm_values, shape_values = self._extract_positional_arguments(args)
        choices = np.random.choice(len(norm_values), size=self._size, p=norm_values)
        result = np.zeros(self._size)
        for i, (distribution, shape) in enumerate(zip(self.distributions.values(), shape_values)):
            mask = choices == i
            result[mask] = distribution.rvs(size=mask.sum(), *shape)
        return result

    def _updated_ctor_param(self):
        """
        scipy requires the arguments of the constructor otherwise
        freezing distributions does not work.
        """
        dct = super(mixture_gen, self)._updated_ctor_param()
        dct['distributions'] = self.distributions
        return dct

    def _argcheck(self, *args):
        """
        We allow for arbitrary arguments.
        The original scipy code restricts the arguments to be positive
        """
        return 1


class template_gen(scipy.stats.rv_continuous):
    """
    Generates a distribution given by a histogram.
    This is useful to generate a template distribution from some MC or data
    """
    def __init__(self, histogram, *args, **kwargs):
        """
        Create a new distribution using the given histogram
        @param histogram the return value of np.histogram
        """
        self.histogram = histogram
        pdf, bins = self.histogram
        bin_widths = (np.roll(bins, -1) - bins)[:-1]
        pdf = pdf / float(np.sum(pdf * bin_widths)) 
        cdf = np.cumsum(pdf * bin_widths)[:-1]
        self.template_bins = bins
        self.template_bin_widths = bin_widths
        self.template_pdf = np.hstack([0.0, pdf, 0.0])
        self.template_cdf = np.hstack([0.0, cdf, 1.0])
        super(template_gen, self).__init__(*args, **kwargs)

    def _pdf(self, x):
        """
        PDF of the histogram
        """
        return self.template_pdf[np.digitize(x, bins=self.template_bins)]
    
    def _cdf(self, x):
        """
        CDF calculated from the histogram
        """
        return self.template_cdf[np.digitize(x, bins=self.template_bins)]
    
    def _rvs(self):
        """
        Random numbers distributed like the original histogram
        """
        probabilities = self.template_pdf[1:-1]
        choices = np.random.choice(len(self.template_pdf) - 2, size=self._size, p=probabilities / probabilities.sum())
        uniform = np.random.uniform(size=self._size)
        return self.template_bins[choices] + uniform * self.template_bin_widths[choices]
    
    def _updated_ctor_param(self):
        """
        scipy requires the arguments of the constructor otherwise
        freezing distributions does not work.
        """
        dct = super(template_gen, self)._updated_ctor_param()
        dct['histogram'] = self.histogram
        return dct


class crystalball_gen(scipy.stats.rv_continuous):
    """
    Generates a crystalball distribution
    see https://en.wikipedia.org/wiki/Crystal_Ball_function

    At the moment we only implement the pdf method.
    Scipy calculates all the other methods for us,
    but for speed reasons one should implement _cdf as well
    """
    def _pdf(self, alpha, n):
        """
        Return PDF of the crystalball function
        """
        A = (n / np.abs(alpha) ) ** n * np.exp( - alpha ** 2 / 2.0 )
        B = n / np.abs(alpha)  - np.abs(alpha)
        C = n / np.abs(alpha) * 1.0 / (n - 1.0) * np.exp( - alpha ** 2 / 2.0 )
        D = np.sqrt(np.pi / 2.0) * (1 + scipy.special.erf(np.abs(alpha) / np.sqrt(2) ))
        N = 1.0 / (C + D)
        return np.where(x > -alpha, np.exp(- x**2 / 2), A * (B - x) ** (-n))
		
crystalball = crystalball_gen(name='crystalball')
