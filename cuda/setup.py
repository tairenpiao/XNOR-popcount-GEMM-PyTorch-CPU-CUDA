"""
PyTorch-XNOR-GEMM-Extention
Authors: Taaron (ptrandpxq@gmail.com)
This code can be used only for research purposes.
For other purposes (e.g., commercial), please contact me.
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
      name='xnor_cuda',
      ext_modules=[
            CUDAExtension('xnor_cuda', [
                  'xnor_cuda.cpp',
                  'xnor_kernel.cu',])],
      cmdclass={
            'build_ext':BuildExtension
      })
