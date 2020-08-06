from __future__ import division, absolute_import, with_statement, print_function
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

_ext_srcs = glob.glob('yi/src/*.cpp') + glob.glob('yi/src/*.cu') + glob.glob('flann/ext/*.c') + glob.glob('flann/algorithms/*.cu')
_cxx_args = ['-O2', '-Iyi/include', '-I.', '-DFLANN_USE_CUDA', '-D_FLANN_VERSION=1.9.1', '-DFLANN_STATIC']
_nvc_args = ['-O2', '-Iyi/include', '-I.', '-DFLANN_USE_CUDA', '-D_FLANN_VERSION=1.9.1', '-DFLANN_STATIC', '-gencode=arch=compute_61,code=sm_61', '-use_fast_math']

setup(
    name='yi',
    author='Yi Yu',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='yi',
            sources=_ext_srcs,
            extra_compile_args={
                'cxx': _cxx_args,
                'nvcc': _nvc_args,
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
