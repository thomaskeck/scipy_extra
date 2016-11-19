from numpy.testing import (TestCase, run_module_suite, assert_equal,
    assert_array_equal, assert_almost_equal, assert_array_almost_equal,
    assert_allclose, assert_, assert_raises, assert_warns, dec)

import numpy as np
import scipy.stats as stats

from scipy_extra import stats as extra_stats


class TestTemplate(TestCase):
    def setUp(self):
        # We have 8 bins
        # [1,2), [2,3), [3,4), [4,5), [5,6), [6,7), [7,8), [8,9)
        # But actually np.histogram will put the last 9 also in the [8,9) bin!
        # Therefore there is a slight difference below for the last bin, from what you might
        # have expected.
        histogram = np.histogram([1,2,2,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,7,7,7,8,8,9], bins=8)
        self.template = extra_stats.template_gen(histogram)
    
    def test_pdf(self):
        assert_almost_equal(self.template.pdf(0.0), 0.0/25.0)
        assert_almost_equal(self.template.pdf(1.0), 1.0/25.0)
        assert_almost_equal(self.template.pdf(2.0), 2.0/25.0)
        assert_almost_equal(self.template.pdf(3.0), 3.0/25.0)
        assert_almost_equal(self.template.pdf(4.0), 4.0/25.0)
        assert_almost_equal(self.template.pdf(5.0), 5.0/25.0)
        assert_almost_equal(self.template.pdf(6.0), 4.0/25.0)
        assert_almost_equal(self.template.pdf(7.0), 3.0/25.0)
        # As stated above the pdf in the bin [8,9) is greater than
        # one would naively expect because np.histogram putted the 9
        # into the [8,9) bin. 
        assert_almost_equal(self.template.pdf(8.0), 3.0/25.0)
        # 9 is outside our defined bins [8,9) hence the pdf is already 0
        # for a continuous distribution this is fine, because a single value
        # does not have a finite probability!
        assert_almost_equal(self.template.pdf(9.0), 0.0/25.0)
        assert_almost_equal(self.template.pdf(10.0), 0.0/25.0)
    
    def test_cdf(self):
        assert_almost_equal(self.template.cdf(0.0), 0.0/25.0)
        assert_almost_equal(self.template.cdf(1.0), 1.0/25.0)
        assert_almost_equal(self.template.cdf(2.0), 3.0/25.0)
        assert_almost_equal(self.template.cdf(3.0), 6.0/25.0)
        assert_almost_equal(self.template.cdf(4.0), 10.0/25.0)
        assert_almost_equal(self.template.cdf(5.0), 15.0/25.0)
        assert_almost_equal(self.template.cdf(6.0), 19.0/25.0)
        assert_almost_equal(self.template.cdf(7.0), 22.0/25.0)
        assert_almost_equal(self.template.cdf(8.0), 25.0/25.0)
        assert_almost_equal(self.template.cdf(9.0), 25.0/25.0)
        assert_almost_equal(self.template.cdf(10.0), 25.0/25.0)

    def test_rvs(self):
        N = 10000
        sample = self.template.rvs(size=N)
        assert_equal(np.sum(sample < 1.0), 0.0)
        assert_allclose(np.sum(sample <= 2.0), 1.0/25.0 * N, rtol=0.05) 
        assert_allclose(np.sum(sample <= 3.0), 3.0/25.0 * N, rtol=0.05) 
        assert_allclose(np.sum(sample <= 4.0), 6.0/25.0 * N, rtol=0.05) 
        assert_allclose(np.sum(sample <= 5.0), 10.0/25.0 * N, rtol=0.05) 
        assert_allclose(np.sum(sample <= 6.0), 15.0/25.0 * N, rtol=0.05) 
        assert_allclose(np.sum(sample <= 7.0), 19.0/25.0 * N, rtol=0.05) 
        assert_allclose(np.sum(sample <= 8.0), 22.0/25.0 * N, rtol=0.05) 
        assert_allclose(np.sum(sample <= 9.0), 25.0/25.0 * N, rtol=0.05) 
        assert_allclose(np.sum(sample <= 9.0), 25.0/25.0 * N, rtol=0.05) 
        assert_equal(np.sum(sample > 9.0), 0.0) 


class TestMixture(TestCase):
    def setUp(self):
        self.mixture = extra_stats.mixture_gen([('a', stats.uniform), ('b', stats.uniform)])
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
        sample = self.frozen_mixture1.rvs(size=N)
        assert_equal(np.sum(np.abs(sample) <= 2.0), N)
        assert_equal(np.sum((sample > 0.0) & (sample < 1.0)), 0)
        assert_allclose(np.sum(sample < 0.0), 2.0/3.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample > 1.0), 1.0/3.0 * N, rtol=0.05)
        
        sample = self.frozen_mixture2.rvs(size=N)
        assert_equal(np.sum(np.abs(sample) <= 2.0), N)
        assert_equal(np.sum(sample > 1.0), 0)
        assert_allclose(np.sum(sample < -1.0), 2.0/6.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <  0.0), 5.0/6.0 * N, rtol=0.05)
        

def test_argus_function():
    # There is no usable reference implementation (RooFit implementation returns unreasonable results which are not normalized correctly)
    # Instead we do some tests if the distribution behaves as expected for different shapes and scales
    for i in range(1, 10):
        for j in range(1, 10):
            assert_equal(stats.argus.pdf(i + 0.001, chi=j, scale=i), 0.0)
            assert_(stats.argus.pdf(i - 0.001, chi=j, scale=i) > 0.0)
            assert_equal(stats.argus.pdf(-0.001, chi=j, scale=i), 0.0)
            assert_(stats.argus.pdf(+0.001, chi=j, scale=i) > 0.0)

    for i in range(1, 10):
        assert_equal(stats.argus.cdf(1.0, chi=i), 1.0)


def test_crystalball_function():
    """
    All values are calculated using the independent implementation of the ROOT framework (see https://root.cern.ch/).
    Corresponding ROOT code is given in the comments.
    """
    X = np.linspace(-5.0, 5.0, 21)[:-1]

    # for(float x = -5.0; x < 5.0; x+=0.5) std::cout << ROOT::Math::crystalball_pdf(x, 1.0, 2.0, 1.0) << ", ";
    calculated = extra_stats.crystalball.pdf(X, alpha=1.0, n=2.0)
    expected = np.array([0.0202867, 0.0241428, 0.0292128, 0.0360652, 0.045645, 0.059618, 0.0811467, 0.116851, 0.18258, 0.265652, 0.301023, 0.265652, 0.18258, 0.097728, 0.0407391, 0.013226, 0.00334407, 0.000658486, 0.000100982, 1.20606e-05])
    assert_allclose(expected, calculated, rtol=0.001)

    # for(float x = -5.0; x < 5.0; x+=0.5) std::cout << ROOT::Math::crystalball_pdf(x, 2.0, 3.0, 1.0) << ", ";
    calculated = extra_stats.crystalball.pdf(X, alpha=2.0, n=3.0)
    expected = np.array([0.0019648, 0.00279754, 0.00417592, 0.00663121, 0.0114587, 0.0223803, 0.0530497, 0.12726, 0.237752, 0.345928, 0.391987, 0.345928, 0.237752, 0.12726, 0.0530497, 0.0172227, 0.00435458, 0.000857469, 0.000131497, 1.57051e-05])
    assert_allclose(expected, calculated, rtol=0.001)
    
    # for(float x = -5.0; x < 5.0; x+=0.5) std::cout << ROOT::Math::crystalball_pdf(x, 2.0, 3.0, 2.0, 0.5) << ", ";
    calculated = extra_stats.crystalball.pdf(X, alpha=2.0, n=3.0, loc=0.5, scale=2.0)
    expected = np.array([0.00785921, 0.0111902, 0.0167037, 0.0265249, 0.0423866, 0.0636298, 0.0897324, 0.118876, 0.147944, 0.172964, 0.189964, 0.195994, 0.189964, 0.172964, 0.147944, 0.118876, 0.0897324, 0.0636298, 0.0423866, 0.0265249])
    assert_allclose(expected, calculated, rtol=0.001)

    # for(float x = -5.0; x < 5.0; x+=0.5) std::cout << ROOT::Math::crystalball_cdf(x, 1.0, 2.0, 1.0) << ", ";
    calculated = extra_stats.crystalball.cdf(X, alpha=1.0, n=2.0)
    expected = np.array([0.12172, 0.132785, 0.146064, 0.162293, 0.18258, 0.208663, 0.24344, 0.292128, 0.36516, 0.478254, 0.622723, 0.767192, 0.880286, 0.94959, 0.982834, 0.995314, 0.998981, 0.999824, 0.999976, 0.999997])
    assert_allclose(expected, calculated, rtol=0.001)

    # for(float x = -5.0; x < 5.0; x+=0.5) std::cout << ROOT::Math::crystalball_cdf(x, 2.0, 3.0, 1.0) << ", ";
    calculated = extra_stats.crystalball.cdf(X, alpha=2.0, n=3.0)
    expected = np.array([0.00442081, 0.00559509, 0.00730787, 0.00994682, 0.0143234, 0.0223803, 0.0397873, 0.0830763, 0.173323, 0.320592, 0.508717, 0.696841, 0.844111, 0.934357, 0.977646, 0.993899, 0.998674, 0.999771, 0.999969, 0.999997])
    assert_allclose(expected, calculated, rtol=0.001)

    # for(float x = -5.0; x < 5.0; x+=0.5) std::cout << ROOT::Math::crystalball_cdf(x, 2.0, 3.0, 2.0, 0.5) << ", ";
    calculated = extra_stats.crystalball.cdf(X, alpha=2.0, n=3.0, loc=0.5, scale=2.0)
    expected = np.array([0.0176832, 0.0223803, 0.0292315, 0.0397873, 0.0567945, 0.0830763, 0.121242, 0.173323, 0.24011, 0.320592, 0.411731, 0.508717, 0.605702, 0.696841, 0.777324, 0.844111, 0.896192, 0.934357, 0.960639, 0.977646])
    assert_allclose(expected, calculated, rtol=0.001)


if __name__ == "__main__":
    run_module_suite()
