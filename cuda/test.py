"""
PyTorch-XNOR-GEMM-Extention
Authors: Tairen (tairenpiao@gmail.com)
This code can only be used for research purposes.
For other purposes (e.g., commercial), please contact me.
"""

import torch
import xnor_cuda
import time

def Binarize(tensor):
    binarized = torch.where(tensor>0, torch.ones_like(tensor,dtype=torch.float32, device='cuda'), torch.full((tensor.shape),-1, dtype=torch.float32, device='cuda'))
    return binarized


m = 1000
n1 = 768
n2 = 768
k = 3072
inputt = torch.randn(m,n1).to(device="cuda:0")
weight = torch.randn(k,n2).to(device="cuda:0")


weight_t = Binarize(weight.t())


bin_weight = xnor_cuda.encode_cols(weight_t)
# bin_input = xnor_cuda.encode_rows(inputt)
# print(bin_weight.shape)


# torch.cuda.synchronize(device="cuda:0")
# t1 = time.time()


# torch.cuda.synchronize(device="cuda:0")
# diff_en = time.time() - t1
# print("encoding row time: ",diff_en*1000,"ms")


# confirm correctness
for i in range(10000):
    input = torch.randn(m,n1).to(device="cuda:0")
    weight = torch.randn(n2,k).to(device="cuda:0")
    # weight_col = weight.t()
    bin_input = xnor_cuda.encode_rows(input)
    bin_weight = xnor_cuda.encode_cols(weight)
    ans1 = xnor_cuda.test_gemm(input,bin_weight)
    inputt2 = torch.where(input>0,torch.ones_like(input,dtype=torch.float32,device="cuda:0"),torch.full(input.shape,-1, dtype=torch.float32,device="cuda:0")).to(device="cuda:0")
    weightt = torch.where(weight>0,torch.ones_like(weight,dtype=torch.float32,device="cuda:0"),torch.full(weight.shape,-1,dtype=torch.float32,device="cuda:0")).to(device="cuda:0")
    # print(ans1,ans1.shape)
    ans3 = inputt2.matmul(weightt)
    # print(ans3,ans3.shape)
    print("Correct:",torch.equal(ans2,ans3))


torch.cuda.synchronize(device="cuda:0")
t5 = time.time()
for i in range(100):
    ans4 = xnor_cuda.test_gemm(inputt,bin_weight)
print(ans4)
torch.cuda.synchronize(device="cuda:0")
diff_en5 = time.time() - t5
print("new kernel time is: ",diff_en5*1000,"ms")





inputt2 = torch.where(((inputt>0) | (inputt==0)),torch.ones_like(inputt,dtype=torch.float32,device="cuda:0"),torch.full(inputt.shape,-1, dtype=torch.float32,device="cuda:0")).to(device="cuda:0")
weightt = torch.where(((weight>0) | (weight==0)),torch.ones_like(weight,dtype=torch.float32,device="cuda:0"),torch.full(weight.shape,-1,dtype=torch.float32,device="cuda:0")).to(device="cuda:0")


# print('-------------------------pytorch-----------------------------------------------')
torch.cuda.synchronize(device="cuda:0")
start_time3 = time.time()
for i in range(100):
    ans3 = inputt2.matmul(weightt.t())
torch.cuda.synchronize(device="cuda:0")
end_time3 = time.time()
diff3 = end_time3 - start_time3
print("pytorch_gemm time is",diff3*1000,"ms")



print("Correct:",torch.equal(ans4,ans3))


# # print("my gemm answer is")
# # print(ans1,ans1.shape)
# print("The XNOR answer is")
# print(ans2,ans2.shape)
print("The pytorch gemm answer is")
print(ans3,ans3.shape)

# # print("Correct:",torch.equal(ans1,ans2))
# # print("Correct:",torch.equal(ans1,ans3))
# print("Correct:",torch.equal(ans2,ans3))
# print(torch.nonzero(ans2!=ans3))



# m1 = torch.full((m,n),127).to(dtype=torch.int8,device="cuda:0")
# m2 = torch.full((n,k),127).to(dtype=torch.int8,device="cuda:0")

# print(m1.matmul(m2))

# torch.cuda.synchronize(device="cuda:0")
# start_time4 = time.time()
# for i in range(10):
#     m3 = xnor_cuda.int8_gemm(m1,m2)
# torch.cuda.synchronize(device="cuda:0")
# end_time4 = time.time()

# diff4 = end_time4 - start_time4
# print("int8_gemm time is",diff4*1000,"ms")
# print(m3)

# print("Correct:",torch.equal(ans1,m3))

# print(ans1)
# print(ans3)
# print(torch.nonzero(ans1!=ans3))
