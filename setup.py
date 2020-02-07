from os import path
import subprocess
import sys

from setuptools import setup, Command, Extension
from distutils.command.clean import clean as distutils_clean

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

try:
    import numpy

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


try:
    from Cython.Build import cythonize

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


class NumpyNotInstalled(Exception):
    def __str__(self):
        return "NumPY is not installed"


class CythonNotInstalled(Exception):
    def __str__(self):
        return "Cython is not installed"


def _get_extension(extension, file_ext):
    kwargs = {
        "name": ".".join(extension),
        "sources": ["%s.%s" % ("/".join(extension), file_ext)],
        "language": "c++",
        "extra_compile_args": ["-ffast-math"],
    }
    if NUMPY_AVAILABLE:
        # most of the time it's fine if the include_dirs aren't there
        kwargs["include_dirs"] = [numpy.get_include()]
    else:
        color_red_bold = "\033[1;31m"
        color_reset = "\033[m"
        sys.stderr.write(
            "%sNumpy is not available so we cannot include the libraries\n"
            "It will result in compilation failures where the numpy headers "
            "are missing.\n%s" % (color_red_bold, color_reset),
        )
    return Extension(**kwargs)


def define_extensions(file_ext):
    extensions = [["rpforest", "rpforest_fast"]]
    return [_get_extension(ext, file_ext) for ext in extensions]


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
        if not CYTHON_AVAILABLE:
            raise CythonNotInstalled
        if not NUMPY_AVAILABLE:
            raise NumpyNotInstalled
        extensions = define_extensions("pyx")
        # language_level sets it py2 compatible
        cythonize(extensions, compiler_directives={"language_level": "2"})


class clean(distutils_clean):
    def run(self):
        distutils_clean.run(self)
        subprocess.call(["rm", "-rf", path.join(here, "build")])
        subprocess.call(["rm", "-rf", path.join(here, "rpforest.egg-info")])
        subprocess.call(["find", here, "-name", "rpforest*.pyc", "-type", "f", "-delete"])
        subprocess.call(["find", here, "-name", "rpforest_fast*.so", "-type", "f", "-delete"])


__version__ = open(path.join(here, "rpforest", "VERSION")).read().strip()


setup(
    name="rpforest",
    version=__version__,
    description="Random Projection Forest for approximate nearest neighbours search.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["rpforest"],
    install_requires=["numpy>=1.8.0,<2.0.0"],
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    setup_requires=["numpy>=1.8.0,<2.0.0"],
    cmdclass={"cythonize": Cythonize, "clean": clean},
    author="LYST Ltd (Maciej Kula)",
    author_email="data@lyst.com",
    url="https://github.com/lyst/rpforest",
    download_url="https://github.com/lyst/rpforest/tarball/%s" % (__version__,),
    license="Apache 2.0",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    ext_modules=define_extensions("cpp"),
    project_urls={"Source": "https://github.com/lyst/rpforest"},
)
