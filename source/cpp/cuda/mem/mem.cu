
#include "mem.h"
#include "cuda_error.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int mem::DEVICE = 0;
const int mem::HOST   = 1;
const int mem::MALLOC = 0;
const int mem::FREE   = 1;
const int mem::ADDR   = 0;
const int mem::SIZE   = 1;

mem::mem(int dsize, int hsize) : hp2info(197), dp2info(197)
{
    printf("%ld\n", sizeof(void*));
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            for(int k = 0; k < 2; ++k) {
                tables[i][j][k] = new jump_table();
                printf("%lx, %lx, %ld\n", (uint64_t)tables[i][j][k], (uint64_t)tables[i][j][k]->head, sizeof(tables[i][j][k]->head));
            }
    void* p_d = 0;
    HANDLE_CUDA_ERROR(cudaMalloc(&p_d, dsize));

    void* p_h = malloc(hsize);
    assert(p_h);

    mem_block* dfp = new mem_block(p_d, dsize); 
    tables[DEVICE][FREE][ADDR]->insert(dfp);
    mem_block2* dfs = new mem_block2(p_d, dsize); 
    tables[DEVICE][FREE][SIZE]->insert(dfs);
    
    mem_block* hfp = new mem_block(p_h, hsize); 
    tables[HOST][FREE][ADDR]->insert(hfp);
    mem_block2* hfs = new mem_block2(p_h, hsize); 
    tables[HOST][FREE][SIZE]->insert(hfs);
}

m_info* mem::new_block(int size)
{
    m_info* info = new m_info();
    int align_size = mem::align(size);

    mem_block2 b2(0, align_size);
    jump_node* node = tables[DEVICE][FREE][SIZE]->ge(b2);
    if (node) {
        mem_block2* freeb2 = const_cast<mem_block2*>(dynamic_cast<const mem_block2*>(tables[DEVICE][FREE][SIZE]->remove(node)));
        mem_block freeb_p(freeb2->start, freeb2->len);
        mem_block* freeb = const_cast<mem_block*>(dynamic_cast<const mem_block*>(tables[DEVICE][FREE][ADDR]->remove(freeb_p)));

        info->p_d = freeb->start;
        info->sz = size;
        info->p_h = 0;
        tables[DEVICE][MALLOC][ADDR]->insert(new mem_block(freeb->start, align_size));
        tables[DEVICE][MALLOC][SIZE]->insert(new mem_block2(freeb->start, align_size));

        if (freeb->len > align_size) {
            freeb->start = (void*)(((uint64_t) freeb->start) + align_size);
            freeb->len -= align_size;
            tables[DEVICE][FREE][ADDR]->insert(freeb); 
            
            freeb2->start = (void*)(((uint64_t) freeb2->start) + align_size);
            freeb2->len -= align_size;
            tables[DEVICE][FREE][SIZE]->insert(freeb2);
        } else {
            delete freeb2;
            delete freeb;
        }
    } else {
        jump_node* node = tables[DEVICE][MALLOC][SIZE]->ge(b2);
        if (node) {
            // remove device memory.
            mem_block2* db2 = const_cast<mem_block2*>(dynamic_cast<const mem_block2*>(tables[DEVICE][MALLOC][SIZE]->remove(node)));
            mem_block tmpdb(db2->start, db2->len);
            mem_block* db = const_cast<mem_block*>(dynamic_cast<const mem_block*>(tables[DEVICE][MALLOC][ADDR]->remove(tmpdb)));
            m_info* tmp_info = *dp2info.get(addr_key(db->start));
            dp2info.remove(addr_key(db->start));

            // alloc host memory.
            mem_block2 hb2(0, db->len);
            jump_node* phnode_s = tables[HOST][FREE][SIZE]->ge(hb2);
            mem_block2* phb2 = const_cast<mem_block2*>(dynamic_cast<const mem_block2*>(tables[HOST][FREE][SIZE]->remove(phnode_s)));
            mem_block hb(phb2->start, phb2->len);
            mem_block* phb = const_cast<mem_block*>(dynamic_cast<const mem_block*>(tables[HOST][FREE][ADDR]->remove(hb)));

            mem_block* newhb = new mem_block(phb->start, db->len);
            mem_block2* newhb2 = new mem_block2(phb->start, db->len);
            tables[HOST][MALLOC][ADDR]->insert(newhb);
            tables[HOST][MALLOC][SIZE]->insert(newhb2);
            tmp_info->p_h = phb->start;

            if (phb->len > db->len) {
                phb->start = (void*)(((uint64_t)phb->start) + db->len);
                phb->len -= db->len;
                phb2->start = (void*)(((uint64_t)phb2->start) + db->len);
                phb2->len -= db->len;

                tables[HOST][FREE][SIZE]->insert(phb2);
                tables[HOST][FREE][ADDR]->insert(phb);
            } else {
                delete phb;
                delete phb2;
            }

            // copy from  tmp_info->p_d to tmp_info->p_h, size: tmp_info->size;
            HANDLE_CUDA_ERROR(cudaMemcpy(tmp_info->p_h, tmp_info->p_d, tmp_info->sz, cudaMemcpyDeviceToHost));
            tmp_info->p_d = 0;
            hp2info.insert(addr_key(tmp_info->p_h), tmp_info);

            // alloc tmp_info->p_d to info->p_d 
            mem_block* newdb = new mem_block(db->start, align_size);
            mem_block2* newdb2 = new mem_block2(db->start, align_size);
            tables[DEVICE][MALLOC][ADDR]->insert(newdb);
            tables[DEVICE][MALLOC][SIZE]->insert(newdb2);
            info->p_d = db->start;
            info->p_h = 0;
            info->sz = size;
            if (db->len > align_size) {
                db->start = (void*)(((uint64_t)db->start) + align_size);
                db->len -= align_size;
                db2->start = (void*)(((uint64_t)db2->start) + align_size);
                db2->len -= align_size;

                tables[DEVICE][FREE][ADDR]->insert(db);
                tables[DEVICE][FREE][SIZE]->insert(db2);
            } else {
                delete db;
                delete db2;
            }
        } else {
            node = tables[DEVICE][MALLOC][SIZE]->last();
            // TODO:
            printf("NOT implemented yet.\n");
            exit(-1);
        }
    }

    dp2info.insert(addr_key(info->p_d), info);
    return info;
}

void mem::free_block(m_info* info)
{
    if (info->p_h) {
        free_block(tables[HOST][MALLOC][ADDR], tables[HOST][MALLOC][SIZE],
                   tables[HOST][FREE][ADDR], tables[HOST][FREE][SIZE],
                   info->p_h, info->sz);
    }
    if (info->p_d) {
        free_block(tables[DEVICE][MALLOC][ADDR], tables[DEVICE][MALLOC][SIZE],
                   tables[DEVICE][FREE][ADDR], tables[DEVICE][FREE][SIZE],
                   info->p_d, info->sz);
    }
}

void mem::free_block(jump_table* malloced_p, jump_table* malloced_s, 
                     jump_table* freed_p, jump_table* freed_s, 
                     void* p, int size)
{
    int aligned_size = this->align(size);
    mem_block b(p, aligned_size);
    mem_block2 b2(p, aligned_size);
    jump_node* node_p = malloced_p->eq(b);
    jump_node* node_s = malloced_s->eq(b2);
    if (node_p) {
        block* b_p = const_cast<block*>(node_p->b);
        malloced_p->remove(node_p);
        block* b_s = const_cast<block*>(node_s->b);
        malloced_s->remove(node_s);

        jump_node* prev_node = freed_p->le(b);
        if (prev_node 
            && ((uint64_t)dynamic_cast<mem_block*>(prev_node->b)->start) + dynamic_cast<mem_block*>(prev_node->b)->len == (uint64_t)p) {
            mem_block2 mb2(dynamic_cast<mem_block*>(prev_node->b)->start, dynamic_cast<mem_block*>(prev_node->b)->len);
            mem_block2* pmb2 = const_cast<mem_block2*>(dynamic_cast<const mem_block2*>(freed_s->remove(mb2)));
            pmb2->len += aligned_size;
            freed_s->insert(pmb2);

            dynamic_cast<mem_block*>(prev_node->b)->len += aligned_size;

            jump_node* next_node = freed_p->next(prev_node);
            if (next_node && (uint64_t) p + aligned_size == (uint64_t)dynamic_cast<mem_block*>(next_node->b)->start) {
                freed_s->remove(*pmb2);
                pmb2->len += dynamic_cast<mem_block*>(next_node->b)->len;
                freed_s->insert(pmb2);

                dynamic_cast<mem_block*>(prev_node->b)->len += dynamic_cast<mem_block*>(next_node->b)->len;

                mem_block2 next_mb2(dynamic_cast<mem_block*>(next_node->b)->start, dynamic_cast<mem_block*>(next_node->b)->len);
                block* pnext_mb2 = freed_s->remove(next_mb2);
                block* pnext_mb = freed_p->remove(next_node);
                delete pnext_mb2;
                delete pnext_mb;
            }
        } else {
            jump_node* next_node = freed_p->ge(b);
            if (next_node && (uint64_t) p + aligned_size == (uint64_t) dynamic_cast<mem_block*>(next_node->b)->start) {
                mem_block2 mb2(dynamic_cast<mem_block*>(next_node->b)->start, dynamic_cast<mem_block*>(next_node->b)->len);
                mem_block2* pmb2 = const_cast<mem_block2*>(dynamic_cast<const mem_block2*>(freed_s->remove(mb2)));
                pmb2->start = p;
                pmb2->len += aligned_size;
                freed_s->insert(pmb2);

                dynamic_cast<mem_block*>(next_node->b)->start = p;
                dynamic_cast<mem_block*>(next_node->b)->len += aligned_size;
            } else {
                freed_p->insert(b_p);
                freed_s->insert(b_s);
                b_p = 0;
                b_s = 0;
            }
        }
        if (b_p) delete b_p;
        if (b_s) delete b_s;
    }
}

void* mem::get(m_info* info)
{
    if (!info->p_d) {
        m_info* new_device_info = new_block(info->sz);
        cudaMemcpy(new_device_info->p_d, info->p_h, info->sz, cudaMemcpyHostToDevice);

        free_block(info);
        hp2info.remove(addr_key(info->p_h));

        info->p_h = 0;
        info->p_d = new_device_info->p_d;

        dp2info.remove(addr_key(new_device_info->p_d));
        delete new_device_info;

        dp2info.insert(addr_key(info->p_d), info);
    }
    return info->p_d;
}

const m_info* mem::get_device_addr_info(void* addr)
{
    m_info* const* ret = dp2info.get(addr_key(addr));
    if (ret) return *ret;
    else return 0;
}

const m_info* mem::get_host_addr_info(void* addr)
{
    m_info* const* ret = hp2info.get(addr_key(addr));
    if (ret) return *ret;
    else return 0;
}

#ifdef __TEST_MEM__
#undef __TEST_MEM__

int main(int argc, char** argv)
{
    mem mm(1024, 2048);

    m_info* p1 = mm.new_block(256);
    m_info* p2 = mm.new_block(256);
    m_info* p3 = mm.new_block(512);
    m_info* p4 = mm.new_block(128);
    m_info* p5 = mm.new_block(128);
    m_info* p6 = mm.new_block(128);
    mm.get(p2);

    return 0;
}
#endif
