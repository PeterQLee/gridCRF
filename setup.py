 
from distutils.core import setup, Extension
import numpy as np
import os

setup_kwargs={
    'extra_compile_args':['-msse','-msse4','-mavx','-mavx2','-O0','-fno-inline']
}

setup(name="BoardPy", version="1.0",
      include_dirs= [np.get_include(),'src/'],
      ext_modules=[Extension("gridCRF", ["src/gridCRF.c","src/optimize.c"],**setup_kwargs)
])

