"""
PyTorch-XNOR-GEMM-Extention
Authors: Taaron (ptrandpxq@gmail.com)
This code can be used only for research purposes.
For other purposes (e.g., commercial), please contact me.
"""

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='xnor_cpu',
      ext_modules=[cpp_extension.CppExtension('xnor_cpu', ['xnor_cpu.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

Extension(
   name='xnor_cpu',
   sources=['xnor_cpu.cpp'],
   include_dirs=cpp_extension.include_paths(),
   language='c++')
