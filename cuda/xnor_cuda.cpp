/*
PyTorch-XNOR-GEMM-Extention
Author: Taaron (ptrandpxq@gmail.com)
This code can only be used for research purposes.
For other purposes (e.g., commercial), please contact me.
*/

#include <torch/extension.h>
#include <iostream>
#include <torch/types.h>


torch::Tensor encode_rows_cuda(torch::Tensor);
torch::Tensor encode_cols_cuda(torch::Tensor);
torch::Tensor test_gemm_cuda(torch::Tensor, torch::Tensor);


torch::Tensor encode_rows(torch::Tensor input) {
  return encode_rows_cuda(input);
}

torch::Tensor encode_cols(torch::Tensor input) {
  return encode_cols_cuda(input);
}

torch::Tensor test_gemm(torch::Tensor input_a, torch::Tensor intput_b) {
  return test_gemm_cuda(input_a,intput_b);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_rows",&encode_rows,"encode_rows");
    m.def("encode_cols",&encode_cols,"encode_cols");
    m.def("test_gemm",&test_gemm,"test_gemm");
  }
