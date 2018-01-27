from numpy.testing import (TestCase, run_module_suite, assert_equal,
    assert_array_equal, assert_almost_equal, assert_array_almost_equal,
    assert_allclose, assert_, assert_raises, assert_warns, dec)

import numpy as np
import scipy.stats

from scipy_extra import fit


class TestFitter(object):
    def setup_method(self):
        np.random.seed(1234)
        self.distribution = scipy.stats.norm
        self.mapping = lambda p: dict(loc=p[0], scale=p[1])

    def test_unbinned_fit(self):
        fitter = fit.Fitter(self.mapping, self.distribution)
        data = self.distribution.rvs(1, 3, size=100)
        scipy_result = self.distribution.fit(data)
        r = fitter.fit([0.8, 3.3], data)
        assert_almost_equal(scipy_result, r.x, 3)
        uncertainties = fitter.get_uncertainties([[0,2], [2,4]], data)
        # The extended fit yields the same result in this case 
        extended_r = fitter.fit([0.8, 3.3], data, expected_number_of_events=90)
        assert_almost_equal(extended_r.x, r.x, 3)
        extended_uncertainties = fitter.get_uncertainties([[0,2], [2,4]], data, expected_number_of_events=90)
        assert_almost_equal(extended_uncertainties, uncertainties, 3)
    
    def test_significance_fit(self):
        fitter = fit.Fitter(self.mapping, self.distribution)
        significance = []
        for size in [10, 20, 40, 80, 160, 320]:
            data = self.distribution.rvs(1, 3, size=size)
            r = fitter.fit([0.8, 3.3], data)
            significance.append(fitter.get_significance([0.0, None], data))
        assert_equal(significance, sorted(significance))

    def test_binned_fit(self):
        """
        The binned fit requires much more data and a fine binning to reproduce the ML solution given by the analytical formulas
        used by scipy itself
        """
        fitter = fit.Fitter(self.mapping, self.distribution, binnings=dict(bins=np.linspace(-10,10,1000)))
        data = self.distribution.rvs(1, 3, size=100000)
        scipy_result = self.distribution.fit(data)
        r = fitter.fit([0.8, 3.3], data)
        assert_almost_equal(scipy_result, r.x, 2)
