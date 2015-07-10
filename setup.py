import sys

from setuptools import setup, Extension
from setuptools.command.test import test as TestCommand

import numpy as np

try:
    from Cython.Build import cythonize
except:
    print ('You must have Cython installed. '
           'Run sudo pip install cython to do so.')
    raise

# Declare C extensions
extensions = [Extension("rpforest.rpforest_fast", ['rpforest/rpforest_fast.pyx'],
                        language="c++",
                        include_dirs=[np.get_include()],
                        extra_compile_args=["-std=c++11"],
                        extra_link_args=["-std=c++11"])]

reqs = open('requirements.txt', 'r').read().split('\n')


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='rpforest',
    version='0.9',
    description='Random Projection Trees for approximate nearest neighbours search.',
    long_description='',
    packages=['rpforest'],
    install_requires=reqs,
    test_requires=['pytest', 'scikit-learn'],
    cmdclass={'test': PyTest},
    author='Maciej Kula',
    license='Apache 2.0',
    classifiers=['Development Status :: 3 - Alpha'],
    ext_modules=cythonize(extensions)
)
