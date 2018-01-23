from distutils.core import setup

setup(name='scipy_extra',
      author='Thomas Keck',
      author_email='t.keck@online.de',
      url='https://github.com/thomaskeck/scipy_extra',
      keywords='scipy, maximum likelihood, fitting, HEP',
      install_requires=['scipy', 'numpy', 'matplotlib'],
      python_requires='>=3',
      version='0.1',
      packages=['scipy_extra'])

