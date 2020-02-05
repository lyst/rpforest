from __future__ import absolute_import

from os import path

from rpforest.rpforest import RPForest  # noqa

here = path.abspath(path.dirname(__file__))

__version__ = open(path.join(here, "VERSION")).read().strip()
