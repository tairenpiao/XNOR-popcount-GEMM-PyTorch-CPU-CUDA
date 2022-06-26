"""
PyTorch-XNOR-GEMM-Extention
Authors: Taaron (ptrandpxq@gmail.com)
This code can be used only for research purposes.
For other purposes (e.g., commercial), please contact me.
"""

import torch
import xnor_cpu
import time


m = 512
n = 512
k = 512
inputt = torch.randn(m,n)
weight = torch.randn(n,k)


start_time_1 = time.time()
bin_input = xnor_cpu.encode_rows_cpu(inputt)
end_time_1 = time.time()
diff_en = end_time_1 - start_time_1 # row encoding time
print("row encoding time is: ",diff_en * 1000, "ms")
bin_weight = xnor_cpu.encode_cols_cpu(weight) # Because we do not need this process in real inference process 



start_time_2 = time.time()
ans2 = xnor_cpu.xnor_gemm_cpu(bin_input,bin_weight,inputt.size(1))
end_time_2 = time.time()
diff_2 = end_time_2 - start_time_2 
print("xnor_gemm time is: ",diff_2*1000, "ms")



start_time_3 = time.time()
inputt2 = torch.where(((inputt>0) | (inputt==0)),torch.ones_like(inputt),torch.full(inputt.shape,-1))
diff_en_pytorch = time.time() - start_time_3
print("Binarize time is:",diff_en_pytorch*1000,"ms")
weight = torch.where(((weight>0) | (weight==0)),torch.ones_like(weight),torch.full(weight.shape,-1))

'''
The time of naive gemm
'''

start_time1 = time.time()
ans1 = xnor_cpu.gemm(inputt2,weight)
end_time1 = time.time()
diff1 =end_time1 - start_time1
print("my_gemm time is",diff1*1000,"ms")


'''
The time of pytorch gemm
'''
start_time3 = time.time()
ans3 = torch.matmul(inputt2,weight)
end_time3 = time.time()
diff3 = end_time3 - start_time3
print("pytorch_gemm time is",diff3*1000,"ms")



print("The answer of naive gemm is ")
print(ans1,ans1.shape)
print("The XNOR answer is ")
print(ans2,ans2.shape)
print("The pytorch gemm answer is ")
print(ans3,ans3.shape)

print("Correct:",torch.equal(ans1,ans2))
print("Correct:",torch.equal(ans1,ans3))
print("Correct:",torch.equal(ans2,ans3))