import os
import subprocess
import sys

from setuptools import setup, Command, Extension
from setuptools.command.test import test as TestCommand


def define_extensions(file_ext):

    try:
        import numpy as np
    except ImportError:
        print('Please install numpy first.')
        raise

    return [Extension("rpforest.rpforest_fast",
                      ['rpforest/rpforest_fast%s' % file_ext],
                      language="c++",
                      include_dirs=[np.get_include()],
                      extra_compile_args=["-std=c++11"],
                      extra_link_args=["-std=c++11"])]


class Cythonize(Command):
    """
    Compile the extension .pyx files.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        import Cython
        from Cython.Build import cythonize

        assert float(Cython.__version__) >= 0.22

        cythonize(define_extensions('.pyx'))


class Clean(Command):
    """
    Clean build files.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        pth = os.path.dirname(os.path.abspath(__file__))

        subprocess.call(['rm', '-rf', os.path.join(pth, 'build')])
        subprocess.call(['rm', '-rf', os.path.join(pth, 'rpforest.egg-info')])
        subprocess.call(['find', pth, '-name', 'rpforest*.pyc', '-type', 'f', '-delete'])
        subprocess.call(['rm', os.path.join(pth, 'rpforest', 'rpforest_fast.so')])


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
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


reqs = open('requirements.txt', 'r').read().split('\n')


setup(
    name='rpforest',
    version='1.0',
    description='Random Projection Forest for approximate nearest neighbours search.',
    long_description='',
    packages=['rpforest'],
    install_requires=reqs,
    tests_require=['pytest', 'scikit-learn', 'scipy'],
    cmdclass={'test': PyTest, 'cythonize': Cythonize, 'clean': Clean},
    author='LYST Ltd (Maciej Kula)',
    license='Apache 2.0',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: Apache Software License'],
    ext_modules=define_extensions('.cpp')
)
