#ifndef __mem_block__
#define __mem_block__

#include "block.h"
#include <cassert>
#include <cstdio>
#include <cstdint>

class mem_block : public block 
{
public:
    void* start;
    size_t len;
    uint32_t sn;

    inline mem_block(void* start, size_t len, uint32_t sn=0) : start(start), len(len), sn(sn) {}
    inline mem_block(const mem_block& b) : start(b.start), len(b.len), sn(b.sn) {}

    inline bool operator<(const block& t) const {
        const mem_block& o = dynamic_cast<const mem_block&>(t);
        return ((uint64_t) start < (uint64_t) o.start);
    }
    inline bool operator<=(const block& t) const {
        const mem_block& o = dynamic_cast<const mem_block&>(t);
        return ((uint64_t) start <= (uint64_t) o.start);
    }
    inline bool operator==(const block& t) const {
        const mem_block& o = dynamic_cast<const mem_block&>(t);
        return ((uint64_t) start == (uint64_t) o.start);
    }

    inline void dump() {
        printf("(%lx, %lx, %u)", (uint64_t)(start), len, sn);
    }

};

class mem_block2 : public mem_block 
{
public:
    inline mem_block2(void* start, size_t len, uint32_t sn=0) : mem_block(start, len, sn) {}
    inline mem_block2(const mem_block& b) : mem_block(b) {}
    inline bool operator<(const block& t) const {
        const mem_block2& o = dynamic_cast<const mem_block2&>(t);
        return len < o.len || len == o.len && sn < o.sn
            || len == o.len && sn == o.sn && (uint64_t) start < (uint64_t) o.start;
    }
    inline bool operator<=(const block& t) const {
        const mem_block2& o = dynamic_cast<const mem_block2&>(t);
        return len < o.len || len == o.len && sn < o.sn
            || len == o.len && sn == o.sn && (uint64_t) start <= (uint64_t) o.start;
    }
    inline bool operator==(const block& t) const {
        const mem_block2& o = dynamic_cast<const mem_block2&>(t);
        return len == o.len && sn == o.sn && (uint64_t) start == (uint64_t) o.start;
    }
    inline void dump() {
        printf("(%lx, %lx, %u)", (uint64_t)(start), len, sn);
    }
public:
    static uint32_t cnt;
};

uint32_t mem_block2::cnt = 0;
#endif

