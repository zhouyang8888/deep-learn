#ifndef __mem_block__
#define __mem_block__

#include "block.h"
#include <cassert>

struct mem_block : public block 
{
    void* start;
    uint64_t len;

    inline mem_block(void* start, uint64_t len) : start(start), len(len) {}
    inline mem_block(const mem_block& b) : start(b.start), len(b.len) {}

    inline bool operator<(const block& t) const {
        const mem_block& o = dynamic_cast<const mem_block&>(t);
        return ((uint64_t) start < (uint64_t) o.start) || ((uint64_t) start == (uint64_t) o.start && len < o.len);
    }
    inline bool operator<=(const block& t) const {
        const mem_block& o = dynamic_cast<const mem_block&>(t);
        return ((uint64_t) start < (uint64_t) o.start) || ((uint64_t) start == (uint64_t) o.start && len <= o.len);
    }
    inline bool operator==(const block& t) const {
        const mem_block& o = dynamic_cast<const mem_block&>(t);
        return ((uint64_t) start == (uint64_t) o.start && len == o.len);
    }

    inline void merge(const mem_block& nxt) {
        assert((uint64_t)start + len == (uint64_t)nxt.start);
        len += nxt.len;
    }

    inline mem_block get(int len) {
        mem_block b(start, len);

        start = (void*)((uint64_t) start + len);
        len -= len;

        return b;
    }

    inline const void dump() const {
        std::cout << "(" << std::hex << (uint64_t)start << ", " << std::dec << len << ")";
    }

};

struct mem_block2 : public mem_block {
    inline mem_block2(void* start, uint64_t len) : mem_block(start, len) {}
    inline mem_block2(const mem_block2& b) : mem_block(b) {}
    inline bool operator<(const block& t) const {
        const mem_block2& o = dynamic_cast<const mem_block2&>(t);
        return len < o.len || len == o.len && (uint64_t) start < (uint64_t) o.start;
    }
    inline bool operator<=(const block& t) const {
        const mem_block2& o = dynamic_cast<const mem_block2&>(t);
        return len < o.len || len == o.len && (uint64_t) start <= (uint64_t) o.start;
    }
    inline bool operator==(const block& t) const {
        const mem_block2& o = dynamic_cast<const mem_block2&>(t);
        return len == o.len && (uint64_t) start == (uint64_t) o.start;
    }
};

#endif

