"""
PyTorch-XNOR-GEMM-Extention
Authors: Taaron (ptrandpxq@gmail.com)
This code can be used only for research purposes.
For other purposes (e.g., commercial), please contact me.
"""

import torch
import math
import torch.nn as nn
# import xnor_cpu
# import xnor_cuda
import time


def Binarize(tensor):
    binarized = torch.where(tensor>0, torch.ones_like(tensor,dtype=torch.float32), torch.full((tensor.shape),-1, dtype=torch.float32))
    return binarized

def xnor_linear(input, weight,bias=True):
    output = xnor_cuda.test_gemm(input,weight)
    if bias is not None:
        output += bias
    ret = output

    return ret


class BinarizeLinear_training(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear_training, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        
        if self.weight.data.dtype == torch.float:
            self.weight.data = Binarize(self.weight.org)
            out = nn.functional.linear(input, self.weight, self.bias)

        return out


class BinarizeLinear_inference(nn.Module):
    """ 
        BinarizeLinear_inference class
        This class is for xnor inference which modified the original nn.Linear that fit the xnor linear
    """
    def __init__(self, in_features, out_features, bias = True):
        super(BinarizeLinear_inference, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        input.data = Binarize(input.data)
        out = xnor_linear(input, self.quantized_weight, self.bias)
        # out = nn.functional.linear(input, self.quantized_weight, self.bias)
        return out