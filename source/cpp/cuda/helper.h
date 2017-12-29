#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

class helper {
    __global__
        static void copy_array(float* dest, float* src, int len);

    __global__ __device__
        void set_val(float* arr, float val, int len);

};
