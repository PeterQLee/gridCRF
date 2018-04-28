 
from distutils.core import setup, Extension
import numpy as np
import os

setup_kwargs={
    'extra_compile_args':['-msse','-msse4','-mavx','-mavx2','-fno-inline','-O0'],
    'libraries': ['lbfgs','cudart'],
    'library_dirs':['/usr/local/lib','/usr/local/cuda/lib64'],
    #'library_dirs':['/usr/local/lib','build/gpu'],
    'extra_objects':['build/gpu/loopy_gpu.o','build/gpu/loopy_gpu_dlink.o']
}

setup(name="BoardPy", version="1.0",
      include_dirs= [np.get_include(),'src/'],
      ext_modules=[Extension("gridCRF", [ "src/gridCRF.c","src/common.c", "src/loopy.c"],**setup_kwargs)
])

