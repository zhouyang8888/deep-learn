
#ifndef __cuda_error___
#define __cuda_error___

#include "cuda_runtime.h"
#include <iostream>

static void handle_cuda_error(cudaError_t err, const char* file, int line)
{
    if (cudaSuccess != err) {
        std::cerr << "In file " << file << ", line " << line << "failed." << std::endl;
        std::cerr << "Err message : " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

#define HANDLE_CUDA_ERROR(err)  (handle_cuda_error(err, __FILE__, __LINE__))

#endif
