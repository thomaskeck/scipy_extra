from numpy.testing import (TestCase, run_module_suite, assert_equal,
    assert_array_equal, assert_almost_equal, assert_array_almost_equal,
    assert_allclose, assert_, assert_raises, assert_warns, dec)

import numpy as np
import scipy.stats

from scipy_extra import stats


class TestMixture(object):
    def setup_method(self):
        np.random.seed(1234)
        self.mixture = stats.rv_mixture([('a', scipy.stats.uniform), ('b', scipy.stats.uniform)])
        self.frozen_mixture1 = self.mixture(a_norm=2.0, a_loc=-2, a_scale=2.0, b_norm=1.0, b_loc=1.0, b_scale=1.0)
        # PDF should look like this:
        #
        #           ##################         ###########
        #           ##################         ###########
        # -3 ----- -2 ----- -1 ----- +0 ----- +1 ------ +2 ----- +3
        self.frozen_mixture2 = self.mixture(a_norm=2.0, a_loc=-2, a_scale=2.0, b_norm=1.0, b_loc=-1.0, b_scale=2.0)
        # PDF should look like this:
        #
        #                    #########
        #           ##################
        #           #############################
        # -3 ----- -2 ----- -1 ----- +0 ----- +1 ------ +2 ----- +3

    def test_pdf(self):
        space = np.linspace(-10, -2.0001, 100)
        assert_allclose(self.frozen_mixture1.pdf(space), 0.0)
        space = np.linspace(-1.9999, -0.0001, 100)
        assert_allclose(self.frozen_mixture1.pdf(space), 1.0/3.0)
        space = np.linspace(0.0001, 0.9999, 100)
        assert_allclose(self.frozen_mixture1.pdf(space), 0.0)
        space = np.linspace(1.0001, 1.9999, 100)
        assert_allclose(self.frozen_mixture1.pdf(space), 1.0/3.0)
        space = np.linspace(2.0001, 10.0, 100)
        assert_allclose(self.frozen_mixture1.pdf(space), 0.0)

        space = np.linspace(-10.0, 10.0, 100)
        assert_allclose(self.mixture.pdf(space, a_norm=2.0, a_loc=-2, a_scale=2.0, b_norm=1.0, b_loc=1.0, b_scale=1.0), self.frozen_mixture1.pdf(space))

        space = np.linspace(-10, -2.0001, 100)
        assert_allclose(self.frozen_mixture2.pdf(space), 0.0)
        space = np.linspace(-1.9999, -1.0001, 100)
        assert_allclose(self.frozen_mixture2.pdf(space), 2.0/6.0)
        space = np.linspace(-0.9999, -0.0001, 100)
        assert_allclose(self.frozen_mixture2.pdf(space), 3.0/6.0)
        space = np.linspace(0.0001, 0.9999, 100)
        assert_allclose(self.frozen_mixture2.pdf(space), 1.0/6.0)
        space = np.linspace(1.0001, 10.0, 100)
        assert_allclose(self.frozen_mixture2.pdf(space), 0.0)

        space = np.linspace(-10.0, 10.0, 100)
        assert_allclose(self.mixture.pdf(space, a_norm=2.0, a_loc=-2, a_scale=2.0, b_norm=1.0, b_loc=-1.0, b_scale=2.0), self.frozen_mixture2.pdf(space))

    def test_cdf(self):
        space = np.linspace(-10, -2.0001, 100)
        assert_allclose(self.frozen_mixture1.cdf(space), 0.0)
        space = np.linspace(-1.9999, -0.0001, 100)
        assert_allclose(self.frozen_mixture1.cdf(space), 1.0/3.0 * (space+2.0))
        space = np.linspace(0.0001, 0.9999, 100)
        assert_allclose(self.frozen_mixture1.cdf(space), 2.0/3.0)
        space = np.linspace(1.0001, 1.9999, 100)
        assert_allclose(self.frozen_mixture1.cdf(space), 2.0/3.0 + 1.0/3.0 * (space-1.0))
        space = np.linspace(2.0001, 10.0, 100)
        assert_allclose(self.frozen_mixture1.cdf(space), 1.0)

        space = np.linspace(-10.0, 10.0, 100)
        assert_allclose(self.mixture.cdf(space, a_norm=2.0, a_loc=-2, a_scale=2.0, b_norm=1.0, b_loc=1.0, b_scale=1.0), self.frozen_mixture1.cdf(space))

        space = np.linspace(-10, -2.0001, 100)
        assert_allclose(self.frozen_mixture2.cdf(space), 0.0)
        space = np.linspace(-1.9999, -1.0001, 100)
        assert_allclose(self.frozen_mixture2.cdf(space), 2.0/6.0 * (space+2.0))
        space = np.linspace(-0.9999, -0.0001, 100)
        assert_allclose(self.frozen_mixture2.cdf(space), 2.0/6.0 + 3.0/6.0 * (space+1.0))
        space = np.linspace(0.0001, 0.9999, 100)
        assert_allclose(self.frozen_mixture2.cdf(space), 5.0/6.0 + 1.0/6.0 * space)
        space = np.linspace(1.0001, 10.0, 100)
        assert_allclose(self.frozen_mixture2.cdf(space), 1.0)

        space = np.linspace(-10.0, 10.0, 100)
        assert_allclose(self.mixture.cdf(space, a_norm=2.0, a_loc=-2, a_scale=2.0, b_norm=1.0, b_loc=-1.0, b_scale=2.0), self.frozen_mixture2.cdf(space))

    def test_rvs(self):
        N = 10000
        sample = self.frozen_mixture1.rvs(size=N, random_state=123)
        assert_equal(np.sum(np.abs(sample) <= 2.0), N)
        assert_equal(np.sum((sample > 0.0) & (sample < 1.0)), 0)
        assert_allclose(np.sum(sample < 0.0), 2.0/3.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample > 1.0), 1.0/3.0 * N, rtol=0.05)

        sample = self.frozen_mixture2.rvs(size=N, random_state=123)
        assert_equal(np.sum(np.abs(sample) <= 2.0), N)
        assert_equal(np.sum(sample > 1.0), 0)
        assert_allclose(np.sum(sample < -1.0), 2.0/6.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample < 0.0), 5.0/6.0 * N, rtol=0.05)
