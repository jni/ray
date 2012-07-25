from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

exts = [Extension("_metrics", 
                ["_metrics.pyx"])]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=exts,
    include_dirs=[np.get_include()]
)
