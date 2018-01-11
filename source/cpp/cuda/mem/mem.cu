
#include "mem.h"
#include "cuda_error.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int mem::DEVICE = 0;
const int mem::HOST   = 1;
const int mem::MALLOC = 0;
const int mem::FREE   = 1;
const int mem::ADDR   = 0;
const int mem::SIZE   = 1;

// #define LOG printf("%s(%d), %s\n", __FILE__, __LINE__, __FUNCTION__)
#define LOG ;
mem::mem(int dsize, int hsize) : hp2info(197), dp2info(197), host_capacity(hsize), device_capacity(dsize)
{
    LOG;
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            for(int k = 0; k < 2; ++k) 
                tables[i][j][k] = new jump_table(5);

    device_start = 0;
    HANDLE_CUDA_ERROR(cudaMalloc(&device_start, dsize));
    cudaDeviceSynchronize();

    host_start = malloc(hsize);
    assert(host_start);

    mem_block* dfp = new mem_block(device_start, device_capacity); 
    tables[DEVICE][FREE][ADDR]->insert(dfp);
    mem_block2* dfs = new mem_block2(device_start, device_capacity); 
    tables[DEVICE][FREE][SIZE]->insert(dfs);
    
    mem_block* hfp = new mem_block(host_start, host_capacity); 
    tables[HOST][FREE][ADDR]->insert(hfp);
    mem_block2* hfs = new mem_block2(host_start, host_capacity); 
    tables[HOST][FREE][SIZE]->insert(hfs);
}

m_info* mem::alloc_block(mem_block2& b_s)
{
    LOG;

    jump_node* node = tables[DEVICE][FREE][SIZE]->ge(b_s);
    if (node) {
        mem_block2* freeb2 = dynamic_cast<mem_block2*>(tables[DEVICE][FREE][SIZE]->remove(node));
        mem_block freeb_p(freeb2->start, freeb2->len);
        mem_block* freeb = dynamic_cast<mem_block*>(tables[DEVICE][FREE][ADDR]->remove(freeb_p));

        m_info* info = new m_info();
        info->p_d = freeb->start;
        info->p_h = 0;
        info->sz = 0;

        tables[DEVICE][MALLOC][SIZE]->insert(new mem_block2(freeb->start, b_s.len, ++mem_block2::cnt));
        tables[DEVICE][MALLOC][ADDR]->insert(new mem_block(freeb->start, b_s.len, mem_block2::cnt));

        if (freeb->len > b_s.len) {
            freeb->start = (void*)(((uint64_t) freeb->start) + b_s.len);
            freeb->len -= b_s.len;
            tables[DEVICE][FREE][ADDR]->insert(freeb); 

            freeb2->start = (void*)(((uint64_t) freeb2->start) + b_s.len);
            freeb2->len -= b_s.len;
            tables[DEVICE][FREE][SIZE]->insert(freeb2);
        } else {
            delete freeb2;
            delete freeb;
        }
        return info;
    }

    return 0;
}
jump_node* mem::select_malloc_node(mem_block2& b_s)
{
    LOG;
    jump_node* node = tables[DEVICE][MALLOC][SIZE]->ge(b_s);
    if (!node)
        node = tables[DEVICE][MALLOC][SIZE]->last();
    
    return node;
}
m_info* mem::new_block(int size)
{
    LOG;
    int align_size = mem::align(size);
    mem_block2 b_s(0, align_size, 0);

    m_info* info = 0;
    while (!(info = alloc_block(b_s))) {
        jump_node* node = select_malloc_node(b_s);
        if (node) {
            mem_block2* malloc_device_block_s = dynamic_cast<mem_block2*>(node->b);
            tables[DEVICE][MALLOC][SIZE]->remove(node);

            mem_block copy_device_block_p(*malloc_device_block_s);
            mem_block* malloc_device_block_p = dynamic_cast<mem_block*>(tables[DEVICE][MALLOC][ADDR]->remove(copy_device_block_p));

            void* host_p = swap_out(*malloc_device_block_p);

            m_info* tmp_info = *dp2info.get(addr_key(malloc_device_block_p->start));
            dp2info.remove(addr_key(malloc_device_block_p->start));
            tmp_info->p_h = host_p;
            tmp_info->p_d = 0;
            hp2info.insert(addr_key(tmp_info->p_h), tmp_info);

            drop_block_into_free(tables[DEVICE][FREE][ADDR], tables[DEVICE][FREE][SIZE]
                                 , malloc_device_block_p, malloc_device_block_s);
        } else {
            break;
        }
    }
    if (info) {
        info->sz = size;
        dp2info.insert(addr_key(info->p_d), info);
    } 

    /*
       tables[DEVICE][FREE][SIZE]->dump();
       tables[DEVICE][MALLOC][SIZE]->dump();
       tables[DEVICE][FREE][ADDR]->dump();
       tables[DEVICE][MALLOC][ADDR]->dump();
     */
    return info;
}

void mem::free_block(m_info* info)
{
    LOG;
    if (info->p_h) {
        free_block(tables[HOST][MALLOC][ADDR], tables[HOST][MALLOC][SIZE],
                   tables[HOST][FREE][ADDR], tables[HOST][FREE][SIZE],
                   info->p_h, info->sz);
        hp2info.remove(addr_key(info->p_h));
    }
    if (info->p_d) {
        free_block(tables[DEVICE][MALLOC][ADDR], tables[DEVICE][MALLOC][SIZE],
                   tables[DEVICE][FREE][ADDR], tables[DEVICE][FREE][SIZE],
                   info->p_d, info->sz);
        dp2info.remove(addr_key(info->p_d));
    }
}

void* mem::swap_out(mem_block& block)
{
    LOG;
    // get free host block
    mem_block2 host_block_s(0, block.len, 0);

    jump_node* free_host_node_s = tables[HOST][FREE][SIZE]->ge(host_block_s);
    if (!free_host_node_s) {
        host_memory_fix();
        free_host_node_s = tables[HOST][FREE][SIZE]->ge(host_block_s);
        if (!free_host_node_s) {
            printf(":( Host memory too small! EXIT!!!\n");
            exit(-1);
        }
    }
    mem_block2* free_host_block_s = dynamic_cast<mem_block2*>(free_host_node_s->b);
    tables[HOST][FREE][SIZE]->remove(free_host_node_s);

    mem_block host_block_p(*free_host_block_s);
    mem_block* free_host_block_p = dynamic_cast<mem_block*>(tables[HOST][FREE][ADDR]->remove(host_block_p)); 

    // alloc host block
    mem_block2* malloc_host_block_s = new mem_block2(free_host_block_p->start
                                                     , block.len, ++mem_block2::cnt);
    mem_block* malloc_host_block_p = new mem_block(*malloc_host_block_s);
    tables[HOST][MALLOC][ADDR]->insert(malloc_host_block_p);
    tables[HOST][MALLOC][SIZE]->insert(malloc_host_block_s);

    if (free_host_block_s->len > block.len) {
        free_host_block_s->start = (void*)((uint64_t)free_host_block_s->start + block.len);
        free_host_block_s->len -= block.len;

        free_host_block_p->start = free_host_block_s->start;
        free_host_block_p->len = free_host_block_s->len;

        drop_block_into_free(tables[HOST][FREE][ADDR], tables[HOST][FREE][SIZE]
                             , free_host_block_p, free_host_block_s);
    } else {
        delete free_host_block_p;
        delete free_host_block_s;
    }

    // move data
    HANDLE_CUDA_ERROR(cudaMemcpy(malloc_host_block_p->start, block.start, block.len, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // return host addr.
    return malloc_host_block_p->start;
}
void mem::drop_block_into_free(jump_table* freed_p, jump_table* freed_s, mem_block* b_p, mem_block2* b_s)
{
    LOG;
    b_p->sn = 0;
    b_s->sn = 0;

    bool merge = false;
    jump_node* prev_node_p = freed_p->le(*b_p);
    if (prev_node_p) {
        mem_block* prev_block_p = dynamic_cast<mem_block*>(prev_node_p->b);
        if ((uint64_t)prev_block_p->start + prev_block_p->len == (uint64_t)b_p->start) {
            merge = true;
            // merge prev_node(addr)
            mem_block2 copy_prev_block_s(*prev_block_p);
            mem_block2* prev_block_s = dynamic_cast<mem_block2*>(freed_s->remove(copy_prev_block_s));
            prev_block_s->len += b_p->len;
            freed_s->insert(prev_block_s);

            prev_block_p->len += b_p->len;

            jump_node* next_node_p = freed_p->next(prev_node_p);
            if (next_node_p) {
                mem_block* next_block_p = dynamic_cast<mem_block*>(next_node_p->b);
                if ((uint64_t)b_p->start + b_p->len == (uint64_t)next_block_p->start) {
                    // merge next_node(addr)
                    freed_s->remove(*prev_block_s);
                    prev_block_s->len += next_block_p->len;
                    freed_s->insert(prev_block_s);

                    prev_block_p->len = prev_block_s->len;

                    // delete next_node(addr)
                    mem_block2 copy_next_block_s(*next_block_p);
                    block* next_block_s = freed_s->remove(copy_next_block_s);
                    delete next_block_s;

                    freed_p->remove(next_node_p);
                    delete next_block_p;
                }
            }
        }
    }
    if (!merge) {
        jump_node* next_node_p = freed_p->ge(*b_p);
        if (next_node_p) {
            mem_block* next_block_p = dynamic_cast<mem_block*>(next_node_p->b);
            if ((uint64_t)b_p->start + b_p->len == (uint64_t)next_block_p->start) {
                merge = true;
                // merge into next node
                mem_block2 copy_next_block_s(*next_block_p);
                mem_block2* next_block_s = dynamic_cast<mem_block2*>(freed_s->remove(copy_next_block_s));
                next_block_s->start = b_p->start;
                next_block_s->len += b_p->len;
                freed_s->insert(next_block_s);

                next_block_p->start = b_p->start;
                next_block_p->len += b_p->len;
            }
        }
    }
    if (!merge) {
        freed_p->insert(b_p);
        freed_s->insert(b_s);
    } else {
        delete b_p;
        delete b_s;
    }
}
void mem::free_block(jump_table* malloced_p, jump_table* malloced_s, 
                     jump_table* freed_p, jump_table* freed_s, 
                     void* p, int size)
{
    LOG;
    int aligned_size = this->align(size);
    mem_block b(p, aligned_size, 0);
    block* block_p = malloced_p->remove(b);
    if (block_p) {
        mem_block2 copy_block_s(*dynamic_cast<mem_block*>(block_p));
        block* block_s = malloced_s->remove(copy_block_s);

        // drop into free table.
        drop_block_into_free(freed_p, freed_s
                             , dynamic_cast<mem_block*>(block_p)
                             , dynamic_cast<mem_block2*>(block_s));
    }
}
void* mem::get(m_info* info)
{
    LOG;
    if (!info->p_d && !info->p_h) return 0;
    if (info->p_h) {
        m_info* new_device_info = new_block(info->sz);
        cudaMemcpy(new_device_info->p_d, info->p_h, info->sz, cudaMemcpyHostToDevice);

        free_block(info);

        info->p_h = 0;
        info->p_d = new_device_info->p_d;

        dp2info.remove(addr_key(new_device_info->p_d));
        delete new_device_info;

        dp2info.insert(addr_key(info->p_d), info);
    }
    return info->p_d;
}
void mem::host_memory_fix()
{
    LOG;
    hash<addr_key, m_info*>& hash_table = hp2info;
    jump_table* malloced_addr = tables[HOST][MALLOC][ADDR];
    jump_table* malloced_size = tables[HOST][MALLOC][SIZE];
    jump_table* freed_addr = tables[HOST][FREE][ADDR];
    jump_table* freed_size = tables[HOST][FREE][SIZE];

    jump_node* itr = malloced_addr->first();
    if (itr) {
        // update by addr
        void* desc = host_start;
        while (itr) {
            mem_block* b = dynamic_cast<mem_block*>(itr->b);
            void* src = b->start;
            int len = b->len;
            memmove(desc, src, len);
            b->start = desc;

            addr_key srckey(src);
            m_info* p_info = *hash_table.get(srckey);
            hash_table.remove(srckey);
            p_info->p_h = desc;
            hash_table.insert(addr_key(desc), p_info);

            // 
            desc = (void*)((uint64_t)desc + len);
            itr = malloced_addr->next(itr);
        }

        printf("yyyyy\n");
        malloced_addr->dump();

        // update size table
        itr = malloced_size->first();
        while (itr) {
            block* pb = malloced_size->remove(itr);
            delete pb;
            itr = malloced_size->first();
        }
        itr = malloced_addr->first();
        while (itr) {
            mem_block* mb = dynamic_cast<mem_block*>(itr->b);
            mem_block2* mb2 = new mem_block2(*mb);
            malloced_size->insert(mb2);
            itr = malloced_addr->next(itr);
        }

        // free table
        itr = freed_addr->first();
        while (itr) {
            block* pb = freed_addr->remove(itr);
            delete pb;
            itr = freed_addr->first();
        }
        itr = freed_size->first();
        while (itr) {
            block* pb = freed_size->remove(itr);
            delete pb;
            itr = freed_size->first();
        }

        freed_addr->insert(new mem_block(desc, host_capacity - ((uint64_t)desc - (uint64_t)host_start)));
        freed_size->insert(new mem_block2(desc, host_capacity - ((uint64_t)desc - (uint64_t)host_start)));
    }
}
m_info* mem::get_device_addr_info(void* addr)
{
    m_info* const* ret = dp2info.get(addr_key(addr));
    if (ret) return *ret;
    else return 0;
}

m_info* mem::get_host_addr_info(void* addr)
{
    m_info* const* ret = hp2info.get(addr_key(addr));
    if (ret) return *ret;
    else return 0;
}
#undef LOG

#ifdef __TEST_MEM__
#undef __TEST_MEM__

#include <time.h>
#include <cstdlib>

int main(int argc, char** argv)
{
    mem mm(1024, 51200 - 1024);

    srand(time(0));
    
#define SIZE 160
    m_info* infos[SIZE];
    int size = 0;
    for (int i = 0; i < 1500000; ++i) {
        int sz = 1 + rand() % 512;
        int align_sz = ((sz + 3) >> 2 << 2);
        int idx = i % SIZE;

        if (i >= SIZE) {
            void* p = mm.get(infos[idx]);
            assert(p);
            int result;
            cudaMemcpy(&result, p, sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            assert(result == infos[idx]->sz);

            printf("Free %d\n", ((infos[idx]->sz + 3) >> 2) << 2);
            size -= ((infos[idx]->sz + 3) >> 2) << 2;
            mm.free_block(infos[idx]);
            delete infos[idx];
        }

        printf("[%d th, %d]: Host size: %d - %d = %d\n", i, sz, 51200, size, 51200 - size);
        infos[idx] = mm.new_block(sz);
        cudaMemcpy(infos[idx]->p_d, &infos[idx]->sz, sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemset(infos[idx]->p_d, (infos[idx]->sz & 0xFF), infos[idx]->sz);
        cudaDeviceSynchronize();
        size += align_sz;
        printf("Alloc %d\n", align_sz);
    }
#undef SIZE
    /*
       m_info* p1 = mm.new_block(512);
       m_info* p2 = mm.new_block(256);
       m_info* p3 = mm.new_block(512);
       m_info* p4 = mm.new_block(256);
       m_info* p5 = mm.new_block(256);
    // mm.get(p2);
    m_info* p6 = mm.new_block(128);
    mm.get(p2);
     */

    return 0;
}
#endif
