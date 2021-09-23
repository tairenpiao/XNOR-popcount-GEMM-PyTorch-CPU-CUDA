# Real-XNOR-popcount-GEMM-Linear-PyTorch-Extension-CPU_GPU_C++_CUDA_Python

This is a PyTorch extension aims to provide the real 1-bit XNOR GEMM (GEneral Matrix Multiplication) for research purpose. It may helps for those who are doing research or project related to binary neural networks (1-bit quantization).

## Introduction
XNOR GEMM is a fast and efficient GEMM for binary matrices multiplication (all elements are +1 or -1).

This implementation provides 

(1) the implementation of training a simple binary neural network.

(2) both CPU and CUDA XNOR GEMM PyTorch extensions.

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
  │    └── xnor_cuda.cpp: C++ implementation of XNOR GEMM
  │
  ├── binarized_modules.py: PyTorch implementations of XNOR GEMM
  └── main.py: The binary MLP model that uses XNOR GEMM
```
For CPU/CUDA implementation, it includes 
1. The C++/CUDA implementation of XNOR GEMM
2. A simple python test file to check if XNOR GEMM works well
3. A setup file

We also provide a simple binary MLP for test the XNOR Linear

binarized_modules.py provides the PyTorch implementation of the XNOR Linear layer.
main.py provides a simple MLP model for testing the XNOR Linear modules.


## Dependencies
    Ubuntu 18.04 LTS
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
If this code helps, you can give me a star, and donations are always welcome.

<img swidth="440" height="300" src="./images/z.jpg"/>
<img swidth="440" height="300" src="./images/w.jpg"/>
