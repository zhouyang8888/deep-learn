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
        hash<addr_key, m_info*> hp2info(197);
        hash<addr_key, m_info*> dp2info(197);

        jump_table tables[2][2][2];

        mem(int dsize, int hsize);
        m_info* new_block(int size);
        void free_block(m_info* info);
        void* get(m_info* info);


        inline int align(int size) { return ((size + 3) >> 2) << 2; }
        void free_block(jump_table& malloced, jump_table& freed, void* p, int size);
        void free_block(m_info* info);
    private:
        static const int DEVICE;
        static const int HOST;
        static const int MALLOC;
        static const int FREE;
        static const int ADDR;
        static const int SIZE;
};

#endif
