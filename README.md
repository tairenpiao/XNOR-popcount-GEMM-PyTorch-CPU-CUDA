# Real-XNOR-popcount-GEMM-Linear-PyTorch-Extension-CPU_GPU_C++_CUDA_Python

This repository provides the real 1-bit XNOR GEMM (GEneral Matrix Multiplication) PyTorch extension for research purpose. It may helps for those who are doing research or project related to binary neural networks (1-bit quantization).

## Introduction
XNOR GEMM is a fast and efficient GEMM for binary matrices multiplication (all elements are +1 or -1).

This implementation provides 

    (1) Both CPU and CUDA XNOR GEMM implementation of PyTorch extensions.
    (2) The implementation of training a simple binary neural network.

So, if you want to save your pytorch model and make inference using real 1-bit numbers (XNOR GEMM), it may helps.


### Code structure
```
Real-XNOR-popcount-GEMM-Linear-PyTorch-Extension-CPU_GPU_C++_CUDA_Python
  │ 
  ├── cpu
  │    ├── setup.py: setup modules
  │    ├── test.py: test modules
  │    └── xnor_cpu.cpp: C++ implementation of XNOR GEMM
  ├── cuda
  │    ├── setup.py: setup modules
  │    ├── test.py: test modules
  │    ├── xnor_cuda.cpp:  Pytorch C++ interface of CUDA XNOR GEMM
  │    └── xnor_kernel.cu: CUDA implementation of XNOR operations 
  │
  ├── binarized_modules.py: Python (PyTorch) implementation of XNOR GEMM
  └── main.py: The binary (1-bit) MLP model that uses XNOR GEMM
```
For CPU/CUDA implementation, it includes 
1. The C++/CUDA implementation of XNOR GEMM
2. A simple python test file to check if XNOR GEMM works well
3. A setup file

The repo also provide a simple binary MLP for test the XNOR Linear layer.

binarized_modules.py provides the PyTorch implementation of the XNOR Linear layer.
main.py provides a simple MLP model for testing the XNOR Linear modules.


## Dependencies
    Ubuntu 18.04 LTS / Mac Os Catalina/ Windows 10 or later
    C++/CUDA
    Python >= 3.6
    PyTorch >= 1.4.0 (pytorch 1.9.0 is recommened)

## How to Use

### 1. Simulated quantization-aware training (simulated quantization)
    
    python main.py

### 2. Real XNOR GEMM inference (1-bit inference)

You first need to uncommnet some code (e.g., the xnor_cpu and xnor_cuda in the binarized_modules.py)
    
    cd cuda
    python setup.py install
    python test.py
    cd ..
    python main.py


## Contact
Email: ptrandpxq@gmail.com

## Donations
I am interested in deep learning, and your support becomes my main motivations.
If this repo helps, you can give me a star, or buy me a coffee.

<img swidth="440" height="300" src="./images/z.jpg"/>
<img swidth="440" height="300" src="./images/w.jpg"/>
