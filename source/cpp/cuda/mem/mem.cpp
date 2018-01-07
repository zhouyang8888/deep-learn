
#include "mem.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_error.h"
#include <cassert>

#pragma pack(4)

mem::mem(int dsize, int hsize) : h2d(197), d2h(197), dm(), df(), hm(), hf()
{
    void* p_d = 0;
    HANDLE_CUDA_ERROR(cudaMalloc(&p_d, dsize));

    void* p_h = malloc(hsize);
    assert(p_h);

    mem_block2* mb_df = new mem_block2(p_d, dsize); 
    df.insert(mb_df);

    mem_block2* mb_hf = new mem_block2(p_h, hsize);
    hf.insert(mb_hf);
}

// TODO:xxxx
m_info* mem::new_block(int size)
{

}

void mem::free_block(m_info* info)
{

}

float* mem::get(m_info* info)
{

}

