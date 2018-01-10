
#ifndef __cuda_error___
#define __cuda_error___

#include "cuda_runtime.h"
#include <cstdio>

static void handle_cuda_error(cudaError_t err, const char* file, int line)
{
    if (cudaSuccess != err) {
        printf("In file %s, line %d failed.\n", file, line);
        printf("Err message: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

#define HANDLE_CUDA_ERROR(err)  (handle_cuda_error(err, __FILE__, __LINE__))

#endif
