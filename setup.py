from os import path

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.sdist import sdist as _sdist

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

cython_modules = [["rpforest", "rpforest_fast"]]


def _cythonize(extensions, apply_cythonize):
    import numpy

    ext = ".pyx" if apply_cythonize else ".cpp"
    extensions = [
        Extension(
            ".".join(mod),
            ["/".join(mod) + ext],
            language="c++",
            extra_compile_args=["-ffast-math"],
        )
        for mod in extensions
    ]
    for extension in extensions:
        extension.include_dirs.append(numpy.get_include())
        # Add signature for Sphinx
        extension.cython_directives = {"embedsignature": True}
    if apply_cythonize:
        from Cython.Build import cythonize

        extensions = cythonize(extensions)
    return extensions


class lazy_cythonize(list):
    # Adopted from https://stackoverflow.com/a/26698408/7820599

    def __init__(self, extensions, apply_cythonize=False):
        super(lazy_cythonize, self).__init__()
        self._list = extensions
        self._apply_cythonize = apply_cythonize
        self._is_cythonized = False

    def _cythonize(self):
        self._list = _cythonize(self._list, self._apply_cythonize)
        self._is_cythonized = True

    def c_list(self):
        if not self._is_cythonized:
            self._cythonize()
        return self._list

    def __iter__(self):
        for e in self.c_list():
            yield e

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


class sdist(_sdist):
    def run(self):
        # Force cythonize for sdist
        _cythonize(cython_modules, True)
        _sdist.run(self)


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
    author="LYST Ltd (Maciej Kula)",
    author_email="data@lyst.com",
    url="https://github.com/lyst/rpforest",
    download_url="https://github.com/lyst/rpforest/tarball/1.6",
    license="Apache 2.0",
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
    ext_modules=lazy_cythonize(cython_modules),
    project_urls={"Source": "https://github.com/lyst/rpforest"},
)
