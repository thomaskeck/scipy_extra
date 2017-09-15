#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.stats
from scipy.stats import rv_continuous
import scipy.special as sc
import functools

_norm_pdf_C = np.sqrt(2*np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)


def _norm_pdf(x):
    return np.exp(-x**2/2.0) / _norm_pdf_C


def _norm_cdf(x):
    return sc.ndtr(x)


def _get_shape_parameters(distribution):
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


class mixture_gen(rv_continuous):
    """ Mixture of distributions

    %(before_notes)s

    Notes
    -----
    The probability density function for `mixture` is given
    by the weighted sum of probability densities of all sub distributions

        mixture.pdf(x, *shapes) = c_a a.pdf(x, shapes_a) + c_b b.pdf(x, shapes_b) + ...

    `mixture` takes all shape parameters of its sub distributions as a shape parameters.
    And in addition there is one shape parameter per sub distribution for the relative normalisation.
    The names of the shape parameters are the same as in the sub distribution with a prefix:
    
    E.g.

    mixture_gen([('Gauss', scipy.stats.norm), ('Gamma', scipy.stats.gamma)])

    has the following shape parameters (in this order):
      - Gauss_norm
      - Gamma_norm
      - Gauss_loc
      - Gauss_scale
      - Gamma_a
      - Gamma_loc
      - Gamma_scale

    So, first all the normalisation parameters, secondly the shape parameters of the individual distributions.

    See Also
    --------

    References
    ----------


    %(example)s

    """
    def __init__(self, distributions, *args, **kwargs):
        """
        Create a new mixture model given a number of distributions
        @param distributions: list( tuple(string, scipy.stats.rv_continuous) )
        """
        self.ctor_argument = distributions
        self.distributions = []
        self.components = []
        self.distribution_norms = []
        self.distribution_shapes = []
        for component, distribution in distributions:
            self.distributions.append(distribution)
            self.components.append(component)
            self.distribution_norms.append('{}_norm'.format(component))
            self.distribution_shapes.append(['{}_{}'.format(component, s) for s in _get_shape_parameters(distribution)])
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
        return np.sum((norm * distribution.pdf(x, *shape) for norm, distribution, shape in zip(norm_values, self.distributions, shape_values)), axis=0)
    
    def _cdf(self, x, *args):
        """
        The combined CDF is the sum of the distribution CDFs weighted by their individual norm factors
        """
        norm_values, shape_values = self._extract_positional_arguments(args)
        return np.sum((norm * distribution.cdf(x, *shape) for norm, distribution, shape in zip(norm_values, self.distributions, shape_values)), axis=0)
    
    def _rvs(self, *args):
        """
        Generates random numbers using the individual distribution random generator
        with a probability given by their individual norm factors 
        """
        norm_values, shape_values = self._extract_positional_arguments(args)
        choices = np.random.choice(len(norm_values), size=self._size, p=norm_values)
        result = np.zeros(self._size)
        for i, (distribution, shape) in enumerate(zip(self.distributions, shape_values)):
            mask = choices == i
            result[mask] = distribution.rvs(size=mask.sum(), *shape)
        return result

    def _updated_ctor_param(self):
        """
        Set distributions as additional constructor argument
        """
        dct = super(mixture_gen, self)._updated_ctor_param()
        dct['distributions'] = self.ctor_argument
        return dct

    def _argcheck(self, *args):
        """
        We allow for arbitrary arguments.
        The sub distributions will check their arguments later anyway.
        The default _argcheck method restricts the arguments to be positive.
        """
        return 1


class rv_support(rv_continuous):
    """
    Generates a distribution with a restricted support using a unrestricted distribution.

    %(before_notes)s

    Notes
    -----
    There are no additional shape parameters except for the loc and scale.
    The pdf and cdf are defined as stepwise functions from the provided histogram.
    In particular the cdf is not interpolated between bin boundaries and not differentiable.

    %(after_notes)s

    %(example)s

    data = scipy.stats.norm.rvs(size=100000, loc=0, scale=1.5)
    hist = np.histogram(data, bins=100)
    template = scipy_extra.stats.template_gen(hist)

    """

    def __init__(self, distribution, low, high, *args, **kwargs):
        """
        Create a new distribution using the given distribution
        @param a scipy.stats distribution
        @param low lower limit of the support
        @param high higher limit of the support
        """
        self.distribution = distribution
        self.low = low
        self.high = high
        kwargs['shapes'] = distribution.shapes
        super(rv_support, self).__init__(*args, **kwargs)

    #@functools.lru_cache(maxsize=1000)
    def _get_norm(self, args):
        norm = self.distribution.cdf(self.high, *args) - self.distribution.cdf(self.low, *args)
        return np.where(norm <= 0, 1e9, norm)
    
    #@functools.lru_cache(maxsize=1000)
    def _get_low(self, args):
        return self.distribution.cdf(low, *args)

    def _pdf(self, x, *args):
        """
        PDF of the distribution
        """
        norm = self._get_norm(args)
        return np.where((x < self.low) | (x > self.high), 0.0, self.distribution.pdf(x, *args) / norm)
    
    def _cdf(self, x, *args):
        """
        CDF calculated from the distribution
        """
        low = self._get_low_probability_mass(args)
        norm = self._get_norm(args)
        return np.where(x < self.low, 0.0, np.where(x > self.high, 1.0, (self.distribution.cdf(x, *args) - low) / norm))
    
    def _rvs(self, *args):
        """
        Random numbers distributed like the original distribution with restricted support
        """
        rvs = self.distribution.rvs(*args, size=self._size)
        while len(rvs) < self._size[0]:
            rvs = np.append(rvs, self.distribution.rvs(*args, size=self._size))
        return rvs[:self._size[0]]
    
    def _updated_ctor_param(self):
        """
        Set additional constructor argument
        """
        dct = super(rv_support, self)._updated_ctor_param()
        dct['distribution'] = self.distribution
        dct['low'] = self.low
        dct['high'] = self.high
        return dct



class polynom_gen(rv_continuous):
    """
    Polynom distribution

    %(before_notes)s

    Notes
    -----

    %(after_notes)s

    %(example)s
    """
    def __init__(self, degree, *args, **kwargs):
        """
        """
        self.degree = degree
        kwargs['shapes'] = ', '.join(['a_{}'.format(i) for i in range(self.degree + 1)])
        super(polynom_gen, self).__init__(*args, **kwargs)

    def _pdf(self, x, *args):
        """
        Return PDF of the polynom function
        """
        pdf_not_normed = np.sum([args[i]*x**i for i in range(self.degree + 1)], axis=0)
        norm = np.sum([2 * args[i] / (i+1) for i in range(0, self.degree + 1, 2)], axis=0)
        pdf = np.where(np.abs(x) > 1.0, 0.0, pdf_not_normed / norm)
        return pdf
    
    def _cdf(self, x, *args):
        """
        Return CDF of the polynom function
        """
        cdf_not_normed = np.sum([args[i]*(x**(i+1) - (-1)**(i+1))/ (i+1) for i in range(self.degree + 1)], axis=0)
        norm = np.sum([2 * args[i] / (i+1) for i in range(0, self.degree + 1, 2)], axis=0)
        cdf = np.where(x < -1.0, 0.0, np.where(x > 1.0, 1.0, cdf_not_normed / norm))
        return cdf

    def _argcheck(self, *args):
        """
        TODO Check if chosen a_n lead to a positive definite pdf
        """
        return True
    
    def _updated_ctor_param(self):
        """
        Set the n degree as additional constructor argument
        """
        dct = super(polynom_gen, self)._updated_ctor_param()
        dct['degree'] = self.degree
        return dct

    def _munp(self, n):
        """Compute the n-th non-central moment."""
        return np.sum([args[i] / (n + self.degree + 1) for i in range(self.degree + 1) if (self.degree + n) % 2 == 0])
polynom_1 = polynom_gen(1, name='polynom_1', longname="A Polynom Function of degree 1")
polynom_2 = polynom_gen(2, name='polynom_2', longname="A Polynom Function of degree 2")
polynom_3 = polynom_gen(3, name='polynom_3', longname="A Polynom Function of degree 3")
polynom_4 = polynom_gen(4, name='polynom_4', longname="A Polynom Function of degree 4")
polynom_5 = polynom_gen(5, name='polynom_5', longname="A Polynom Function of degree 5")
polynom_6 = polynom_gen(6, name='polynom_6', longname="A Polynom Function of degree 6")
