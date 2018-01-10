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

#include "hash.h"
#include "jump_table.h"
#include "mem_block.h"
#include <cstdint>

struct m_info {
	void* p_h;
	void* p_d;
	int sz;
};

class addr_key : public hash_key {
public:
    void* p;
    inline addr_key(void* p): p(p){}
    inline uint64_t hash_code() const {
        return (uint64_t) p >> 2;
    }
    inline bool operator==(const hash_key& other) const {
        const addr_key& o = dynamic_cast<const addr_key&>(other);
        return p == o.p;
    }
};

class mem {
	public:
        hash<addr_key, m_info*> hp2info;
        hash<addr_key, m_info*> dp2info;

        jump_table tables[2][2][2];

        mem(int dsize, int hsize);
        m_info* new_block(int size);
        void free_block(m_info* info);
        void* get(m_info* info);


        inline int align(int size) { return ((size + 3) >> 2) << 2; }
        const m_info* get_device_addr_info(void* addr);
        const m_info* get_host_addr_info(void* addr);
    private:
        static const int DEVICE;
        static const int HOST;
        static const int MALLOC;
        static const int FREE;
        static const int ADDR;
        static const int SIZE;
    private:
        void free_block(jump_table& malloced_p, jump_table& malloced_s, 
                        jump_table& freed_p, jump_table& freed_s, 
                        void* p, int size);
};

#endif
