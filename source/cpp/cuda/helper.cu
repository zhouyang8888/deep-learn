#include <helper.h>

__global__
void helper::copy_array(float* dest, float* src, int len)
{
    int stride = blockDim.x;
    for (int i = 0; i < len; i += stride)
        if (i + threadIdx < len)
            dest[i + threadIdx.x] = src[i + threadIdx.x];
}

__global__ __device__
void helper:set_val(float* arr, float val, int len)
{
    int stride = blockDim.x;
    for (int i = 0; i < len; i += stride)
        if (i + threadIdx < len)
            dest[i + threadIdx.x] = val;
}


