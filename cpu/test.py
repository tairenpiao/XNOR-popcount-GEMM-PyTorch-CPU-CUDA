"""
PyTorch-XNOR-GEMM-Extention
Authors: Tairen (tairenpiao@gmail.com)
This code can be used only for research purposes.
For other purposes (e.g., commercial), please contact me.
"""

import torch
import xnor_cpu
import time


def main():
    m = 50
    n = 256 # must be diveded by 32
    k = 30
    # We need to use binary matrices (all elments are 1/-1)
    inputt = torch.randn(m, n)
    weight = torch.randn(n, k)

    ### !! Do not transpose any matrix
    ### !! Do not transpose any matrix
    ### !! Do not transpose any matrix

    start_time_1 = time.time()
    bin_input = xnor_cpu.encode_rows_cpu(inputt)
    diff_en = time.time() - start_time_1 # row encoding time
    print("row encoding time is: ",diff_en * 1000, "ms")
    bin_weight = xnor_cpu.encode_cols_cpu(weight) # Because we do not need this process in real inference process 

    # print(bin_input, bin_input.size())
    # print(bin_weight, bin_weight.size())

    start_time_2 = time.time()
    # out_xnor_gemm = xnor_cpu.xnor_gemm_cpu(bin_input,bin_weight,inputt.size(1))
    out_xnor_gemm = xnor_cpu.xnor_gemm_cpu(bin_input, bin_weight)
    diff_2 = time.time() - start_time_2 
    print("xnor_gemm time is: ",diff_2 * 1000, "ms")


    # Binarize the original FP32 numbers cause their already have binary operation in XNOR GEMM
    start_time_3 = time.time()
    inputt2 = torch.where(inputt > 0, torch.ones_like(inputt), torch.full(inputt.shape, -1))
    diff_en_pytorch = time.time() - start_time_3
    print("Binarize time is:", diff_en_pytorch * 1000, "ms")
    weight = torch.where(weight > 0, torch.ones_like(weight), torch.full(weight.shape, -1))



    '''
    The time of naive gemm
    '''
    start_time1 = time.time()
    out_naive_gemm = xnor_cpu.naive_gemm_cpu(inputt2,weight)
    diff1 = time.time() - start_time1
    print("naive time is",diff1 * 1000,"ms")


    '''
    The time of pytorch gemm
    '''
    start_time3 = time.time()
    out_pytorch_gemm = torch.matmul(inputt2,weight)
    diff3 = time.time() - start_time3
    print("pytorch_gemm time is",diff3 * 1000,"ms")


    print("The output of XNOR GEMM is ")
    print(out_xnor_gemm, out_xnor_gemm.shape)
    print("The output of naive GEMM is ")
    print(out_naive_gemm, out_naive_gemm.shape)
    print("The output of pytorch GEMM is ")
    print(out_pytorch_gemm, out_pytorch_gemm.shape)

    print("XNOR GEMM == Naive GEMM: ", torch.equal(out_naive_gemm, out_xnor_gemm))
    print("XNOR GEMM == PyTorch GEMM: ", torch.equal(out_xnor_gemm, out_pytorch_gemm))
    print("Naive GEMM == PyTorch GEMM: ", torch.equal(out_naive_gemm, out_pytorch_gemm))


if __name__ == '__main__':
    main()