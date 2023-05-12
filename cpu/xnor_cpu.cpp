/*
    PyTorch-XNOR-GEMM-Extention
    Authors: Tairen (tairenpiao@gmail.com)
    This code can only be used for research purposes.
    For other purposes (e.g., commercial), please contact me.
*/

#include <iostream>
#include <typeinfo>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <torch/extension.h>
#include <torch/types.h>

#define ENCODE_BIT 32


void encode_rows_cpu_kernel(float *columns, int32_t *columns_binary, int m, int l) {
    for (int i = 0; i < m * l; ++i) {
        int32_t r = 0;
        for (int j = 0; j < ENCODE_BIT; ++j) {
            int32_t sign = (columns[ENCODE_BIT * i + j] > 0);
            r |= (sign << j);
        }
        columns_binary[i] = r;
    }
}


torch::Tensor encode_rows_cpu(torch::Tensor &input) {
    const int m = input.size(0);
    const int n = input.size(1);
    const int l = (n - 1) / ENCODE_BIT + 1;
    torch::Tensor output = torch::ones({m, l}, torch::kInt);

    // Convert tensor to the C pointer
    float *a = (float *)input.data_ptr<float>();
    int32_t *b = (int32_t *)output.data_ptr<int>();

    encode_rows_cpu_kernel(a, b, m, l);
    return output;
}


void encode_cols_cpu_kernel(float *columns, int32_t *columns_binary, int l, int n)  {
    int col_bin_m = l;

    for (int i = 0; i < col_bin_m; ++i) {
        for (int k = 0; k < n; ++k) {
            int32_t r = 0;
            for (int j = 0; j < ENCODE_BIT; ++j) {
                int32_t sign = (columns[(i * n * ENCODE_BIT + k) + j * n] > 0);
                r |= (sign << j);
            }
            columns_binary[i * n + k] = r;
        }
        
    }
}


torch::Tensor encode_cols_cpu(torch::Tensor &input) {
    const int n = input.size(0);
    const int k = input.size(1);
    const int l = 1 + (n - 1) / ENCODE_BIT;

    torch::Tensor output = torch::ones({l, k}, torch::kInt);

    float *a = input.data_ptr<float>();
    int32_t *b = (int32_t *)output.data_ptr<int>();
    encode_cols_cpu_kernel(a, b, l, k);

    return output;
}


torch::Tensor xnor_gemm_cpu(torch::Tensor &a, torch::Tensor &b) {
    torch::Tensor output = torch::zeros({a.size(0), b.size(1)});

    // row dimension of matrix A
    for(int i = 0; i < a.size(0); ++i) {
        // column dimension of matirx B
        for(int j = 0; j < b.size(1); ++j) {
            int32_t c = 0; // every j loop make c equals to 0
            // column dimension of A = row dimension of B
            for (int i2 = 0; i2 < a.size(1); ++i2) {
                // Use ~XOR to implement XNOR because of no XNOR operation in C++ std
                c += 2 * __builtin_popcount(  ~( a[i][i2].item<int>() ^ b[i2][j].item<int>() ) );
            }
            output[i][j] = c - (a.size(1) * ENCODE_BIT);  // Sum and minus length of input
        }
    }
    return output;
}


torch::Tensor naive_gemm_cpu(torch::Tensor a, torch::Tensor b) {
    torch::Tensor output = torch::zeros({a.size(0),b.size(1)});
    for (int i = 0; i < a.size(0); i++) {   
        for(int j = 0; j < b.size(1); j++) {
            int32_t c = 0;
            for (int i2 = 0; i2 < a.size(1); i2++) {
                c += a[i][i2].item<float>() * b[i2][j].item<float>();
            }
            output[i][j] = c;
        }
    }
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_rows_cpu",&encode_rows_cpu,"encode_rows_cpu");
    m.def("encode_cols_cpu",&encode_cols_cpu,"encode_cols_cpu");
    m.def("xnor_gemm_cpu",&xnor_gemm_cpu,"xnor_gemm_cpu");
    m.def("naive_gemm_cpu",&naive_gemm_cpu,"naive_gemm_cpu");
}