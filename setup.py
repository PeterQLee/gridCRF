 
from distutils.core import setup, Extension
import numpy as np
import os
os.makedirs('build/gpu', exist_ok=True)

def compile_gpu_and_sse():
    setup_kwargs={
        'extra_compile_args':['-msse','-msse2','-msse3','-msse4','-fno-inline','-O0'],
        'libraries': ['lbfgs','cudart', 'cudadevrt'],#, 'loopygpu'],
        'library_dirs':['/usr/local/lib','/usr/local/cuda/lib64'],#, 'build/gpu/'],
        #'library_dirs':['/usr/local/lib','build/gpu'],
        'extra_objects':['build/gpu/loopy_gpu.o','build/gpu/train_gpu.o', 'build/gpu/gpu_link.o',
        ]
    }

    os.system('make')
    setup(name="gridCRF", version="1.0",
          include_dirs= [np.get_include(),'src/'],
        ext_modules=[Extension("gridCRF", [ "src/gridCRF.c","src/common.c", "src/loopy.c", "src/train_cpu.c"],**setup_kwargs)
    ])
    
def compile_gpu_and_avx():
    mode = 'gpu'

    setup_kwargs={
        'extra_compile_args':['-msse','-msse4','-mavx','-mavx2','-fno-inline','-O0'],
        'libraries': ['cudart', 'cudadevrt'],#, 'loopygpu'],
        'library_dirs':['/usr/local/lib','/usr/local/cuda/lib64'],#, 'build/gpu/'],
        #'library_dirs':['/usr/local/lib','build/gpu'],
        'extra_objects':['build/gpu/loopy_gpu.o','build/gpu/train_gpu.o', 'build/gpu/gpu_link.o',
        ]
    }

    os.system('make')
    setup(name="gridCRF", version="1.0",
          include_dirs= [np.get_include(),'src/'],
        ext_modules=[Extension("gridCRF", [ "src/gridCRF.c","src/common.c", "src/loopy.c", "src/train_cpu.c"],**setup_kwargs)
    ])
    
def compile_avx():    
    setup_kwargs={
        'extra_compile_args':['-msse','-msse4','-mavx','-mavx2','-fno-inline','-O0'],
        
    }
    setup(name="gridCRF", version="1.0",
          include_dirs= [np.get_include(),'src/'],
          ext_modules=[Extension("gridCRF", [ "src/gridCRF.c","src/common.c", "src/loopy.c", "src/train_cpu.c", "src/dummy_gpu.c"],**setup_kwargs)
          ])


r= open('/proc/cpuinfo','r')
avxflag= False
for s in r:
    if "avx2" in s:
        avxflag=True
        break
if not avxflag:
    print('compiling only with gpu')
    compile_gpu_and_sse()

else:
    compile_gpu_and_avx()
