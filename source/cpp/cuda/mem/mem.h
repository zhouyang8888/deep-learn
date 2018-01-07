/*
 * =====================================================================================
 *
 *       Filename:  mem.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/30 12时51分34秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __mem__
#define __mem__
#pragma pack(4)

#include "hash.h"
#include "jump_table.h"
#include "mem_block.h"

struct m_info {
	void* p_h;
	void* p_d;
	int sz;
};

class addr_key : public hashkey {
public:
    void* p;
    inline addr_key(void* p): p(p){}
    inline uint64_t hash_code() {
        return (uint64_t) p;
    }
    inline bool operator==(const hashkey& other) {
        return p == other.p;
    }
};

class mem {
	public:
        hash<addr_key, void*> h2d(197);
        hash<addr_key, void*> d2h(197);

        jump_table dm;
        jump_table df;
        jump_table hm;
        jump_table hf;

        mem(int dsize, int hsize);
        m_info* new_block(int size);
        void free_block(m_info* info);
        float* get(m_info* info);
};

#endif
