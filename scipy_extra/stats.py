#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.stats
from scipy.stats import rv_continuous
import scipy.special as sc

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


class template_gen(rv_continuous):
    """
    Generates a distribution given by a histogram.
    This is useful to generate a template distribution from a binned datasample.

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
        self.template_cdf = np.hstack([0.0, cdf, 1.0, 1.0])
        # Set support
        epsilon = 1e-7
        kwargs['a'] = self.template_bins[0] - epsilon
        kwargs['b'] = self.template_bins[-1] + epsilon
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
        Set the histogram as additional constructor argument
        """
        dct = super(template_gen, self)._updated_ctor_param()
        dct['histogram'] = self.histogram
        return dct


class crystalball_gen(rv_continuous):
    """
    Crystalball distribution

    %(before_notes)s

    Notes
    -----
    Named after the crystal ball experiment.
    Used in elementary particle physics to model background components.

    For details see: https://en.wikipedia.org/wiki/Crystal_Ball_function

    %(after_notes)s

    %(example)s
    """
    def _pdf(self, x, alpha, n):
        """
        Return PDF of the crystalball function
        """
        A = (n / np.abs(alpha) ) ** n * np.exp( - alpha ** 2 / 2.0 )
        B = n / np.abs(alpha)  - np.abs(alpha)
        C = n / np.abs(alpha) * 1.0 / (n - 1.0) * np.exp( - alpha ** 2 / 2.0 )
        D = np.sqrt(np.pi / 2.0) * (1 + sc.erf(np.abs(alpha) / np.sqrt(2) ))
        N = 1.0 / (C + D)
        # Using np.where we also calculate powers of negative numbers,
        # because (B - x) ** (..) is also executed for x <= -alpha
        # But these values are not used in the end, therefore we ignore the errors
        with np.errstate(invalid='ignore', divide='ignore'):
            return N * np.where(x > -alpha, np.exp(- x**2 / 2), A * (B - x) ** (-n))		
    
    def _cdf(self, x, alpha, n):
        """
        Return CDF of the crystalball function
        """
        A = (n / np.abs(alpha) ) ** n * np.exp( - alpha ** 2 / 2.0 )
        B = n / np.abs(alpha)  - np.abs(alpha)
        C = n / np.abs(alpha) * 1.0 / (n - 1.0) * np.exp( - alpha ** 2 / 2.0 )
        D = np.sqrt(np.pi / 2.0) * (1 + sc.erf(np.abs(alpha) / np.sqrt(2) ))
        N = 1.0 / (C + D)
        # Using np.where we also calculate powers of negative numbers,
        # because (B - x) ** (..) is also executed for x <= -alpha
        # But these values are not used in the end, therefore we ignore the errors
        with np.errstate(invalid='ignore', divide='ignore'):
            return N * np.where(x > -alpha,
                    A * (B + alpha) ** (-n+1) / (n-1) + _norm_pdf_C * (_norm_cdf(x) - _norm_cdf(-alpha)),
                    A * (B - x) ** (-n+1) / (n-1) )
    
    def _argcheck(self, alpha, n):
        """
        In HEP crystal-ball is also defined for n = 1 (see plot on wikipedia)
        But the function doesn't have a finite integral in this corner case,
        and isn't a PDF anymore (but can still be used on a finite range).
        Here we restrict the function to n > 1.
        """
        return (n > 1)
crystalball = crystalball_gen(name='crystalball', longname="A Crystalball Function")


class polynom_gen(rv_continuous):
    """
    Polynom distribution

    %(before_notes)s

    Notes
    -----

    %(after_notes)s

    %(example)s
    """
    def __init__(self, n, *args, **kwargs):
        """
        """
        self.n = n
        kwargs['shapes'] = ', '.join(['a_{}'.format(i) for i in range(self.n+1)])
        super(polynom_gen, self).__init__(*args, **kwargs)

    def _pdf(self, x, *args):
        """
        Return PDF of the polynom function
        """
        pdf_not_normed = np.sum([args[i]*x**i for i in range(self.n+1)], axis=0)
        norm = np.sum([args[i] / (i+1) for i in range(self.n+1)], axis=0)
        pdf = np.where((x < 0) | (x>1), 0.0, pdf_not_normed / norm)
        return pdf
    
    def _cdf(self, x, *args):
        """
        Return CDF of the polynom function
        """
        cdf_not_normed = np.sum([args[i]*x**(i+1) / (i+1) for i in range(self.n+1)], axis=0)
        norm = np.sum([args[i] / (i+1) for i in range(self.n+1)], axis=0)
        cdf = np.where(x<0, 0.0, np.where(x>1, 1.0, cdf_not_normed / norm))
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
        dct['n'] = self.n
        return dct
polynom_1 = polynom_gen(1, name='polynom_1', longname="A Polynom Function of degree 1")
polynom_2 = polynom_gen(2, name='polynom_2', longname="A Polynom Function of degree 2")
polynom_3 = polynom_gen(3, name='polynom_3', longname="A Polynom Function of degree 3")
polynom_4 = polynom_gen(4, name='polynom_4', longname="A Polynom Function of degree 4")
polynom_5 = polynom_gen(5, name='polynom_5', longname="A Polynom Function of degree 5")
polynom_6 = polynom_gen(6, name='polynom_6', longname="A Polynom Function of degree 6")


def _argus_phi(chi):
    """
    Utility function for the argus distribution
    used in the CDF and norm of the Argus Funktion
    """
    return  _norm_cdf(chi) - chi * _norm_pdf(chi) - 0.5


class argus_gen(rv_continuous):
    """
    Argus distribution

    %(before_notes)s

    Notes
    -----
    Named after the argus experiment.
    Used in elementary particle physics to model invariant mass distributions from continuum background

    For details see: https://en.wikipedia.org/wiki/ARGUS_distribution

    The parameter c of the Argus distributions is named scale in scipy.

    %(after_notes)s

    %(example)s
    """
    def __init__(self, *args, **kwargs):
        """
        Set finite support for argus function.
        """
        # Set support
        kwargs['a'] = 0.0
        kwargs['b'] = 1.0
        super(argus_gen, self).__init__(*args, **kwargs)

    def _pdf(self, x, chi):
        """
        Return PDF of the argus function
        """
        return chi**3 / (_norm_pdf_C * _argus_phi(chi)) * x * np.sqrt(1.0 - x**2) * np.exp(- 0.5 * chi**2 * (1.0 - x**2) )

    def _cdf(self, x, chi):
        """
        Return CDF of the argus function
        """
        return 1.0 - _argus_phi(chi * np.sqrt(1 - x**2)) / _argus_phi(chi)
argus = argus_gen(name='argus', longname="An Argus Function")
