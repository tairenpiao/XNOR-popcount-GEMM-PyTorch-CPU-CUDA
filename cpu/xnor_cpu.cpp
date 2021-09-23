/*
PyTorch-XNOR-GEMM-Extention
Authors: Taaron (ptrandpxq@gmail.com)
This code can be used only for research purposes.
For other purposes (e.g., commercial), please contact me.
*/
#include <torch/extension.h>
#include <iostream>
#include <torch/types.h>
#include <typeinfo>
#include <stdint.h>
#include <stdlib.h>

#include <iterator>

#define ENCODE_BIT 32

inline uint32_t encode_val(float* array, int n)
{
    int sign, r = 0;
    for (int i=0; i<ENCODE_BIT && i<n; i++) {
        sign = array[i]>0;
        r |= (sign<<i);
    }
    return r;
}

void encode_rows_cpu_kernel(float *columns,  uint32_t *columns_binary, int m, int n)
{
    int i, l = 1+(n-1)/ENCODE_BIT;

    for (i = 0; i < m*l; i++) {
        int p = n*(i/l)+ENCODE_BIT*(i%l);
        columns_binary[i] = encode_val(&columns[p], n-ENCODE_BIT*(i%l));
    }
}


at::Tensor encode_rows_cpu(at::Tensor input)
{
    const int m = input.size(0);
    const int n = input.size(1);
    const int l = 1+(n-1)/ENCODE_BIT;
    at::Tensor output = at::ones({m,l}, at::kInt);

    float* a = (float*)input.data<float>();
    // Convert to tensor
    uint32_t* b = (uint32_t*)output.data<int>();

    encode_rows_cpu_kernel(a, b, m, n);
    return output;
}


void encode_cols_cpu_kernel(float *columns, uint32_t *columns_binary, int m, int n) 
{
    int col_bin_m = 1 + (m-1) / ENCODE_BIT;
    int i, j, k;
    //#pragma omp parallel for
    for (i = 0; i < col_bin_m; i++) {
        int i64 = i * ENCODE_BIT;
        for (j = 0; j < n && i64<m ; j++) {

            uint32_t sign, rvalue = 0;

            for (k = 0; j + n * (i64 + k) < m*n && k < ENCODE_BIT; k++) {
                sign = columns[j + n * (i64 + k)]>0;
                rvalue |= (sign << k);
            }
            columns_binary[j + n * i] = rvalue;
        }
    }
}


at::Tensor encode_cols_cpu(torch::Tensor* input) 
{
    const int n = input->size(0);
    const int k = input->size(1);
    const int l = 1+(n-1)/ENCODE_BIT;

    at::Tensor output = at::ones({l,k}, at::kInt);
    float* a = input->data<float>();
    uint32_t* b = (uint32_t*)output.data<int>();

    encode_cols_cpu_kernel(a, b, n, k);
    return output;
}



at::Tensor xnor_gemm_cpu(at::Tensor a, at::Tensor b)
{
    at::Tensor output = at::randn({a.size(0), b.size(1)});
    int c;

    // row dimension of matrix A
    for(int i = 0; i < a.size(0); i++) {
        // column dimension of matirx B
        for(int j = 0; j < b.size(1); j++) {
            c = 0; // every j loop make c equals to 0
            // column dimension of A = row dimension of B
            for (int i2 = 0; i2 < a.size(1); i2++) {
                // Use ~XOR to implement XNOR because of no XNOR operation support
                c += 2 * __builtin_popcount(  ~( a[i][i2].item<int>() ^ b[i2][j].item<int>() ) );
            }
            output[i][j] = c - (a.size(1));  // Sum and minus length of input
        }
    }
    return output;
    
}
at::Tensor my_gemm(at::Tensor a, at::Tensor b)
{
    at::Tensor output = at::randn({a.size(0),b.size(1)});
    int c;

    for(int i =0; i < a.size(0); i++) {   
        for(int j=0; j < b.size(1); j++) {
            c = 0;
            for (int i2=0; i2<a.size(1);i2++) {
                c +=  a[i][i2].item<float>() * b[i2][j].item<float>();
            }
            output[i][j] = c;
        }
        
    }
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("xnor_gemm_cpu",&xnor_gemm_cpu,"xnor_gemm_cpu");
  m.def("naive_gemm_cpu",&my_gemm,"naive_gemm_cpu");
  m.def("encode_rows_cpu",&encode_rows_cpu,"encode_rows_cpu");
  m.def("encode_cols_cpu",&encode_cols_cpu,"encode_cols_cpu");
}