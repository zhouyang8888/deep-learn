#ifndef __jump_table__
#define __jump_table__

#include <iostream>
#include <cstdint>
#include <cassert>

struct block 
{
    void* start;
    uint64_t len;

    inline block(void* start, uint64_t len) : start(start), len(len) {}
    inline block(const block& b) : start(b.start), len(b.len) {}

    inline bool operator<(const block& o) {
        return ((uint64_t) start < (uint64_t) o.start) || ((uint64_t) start == (uint64_t) o.start && len < o.len);
    }
    inline bool operator<=(const block& o) {
        return ((uint64_t) start < (uint64_t) o.start) || ((uint64_t) start == (uint64_t) o.start && len <= o.len);
    }
    inline bool operator==(const block& o) {
        return ((uint64_t) start == (uint64_t) o.start && len == o.len);
    }

    inline void merge(const block& nxt) {
        assert((uint64_t)start + len == (uint64_t)nxt.start);
        len += nxt.len;
    }

    inline block get(int len) {
        block b(start, len);

        start = (void*)((uint64_t) start + len);
        len -= len;

        return b;
    }

    inline const void dump() const {
        std::cout << "(" << std::hex << (uint64_t)start << ", " << std::dec << len << ")";
    }
};

struct jump_node 
{
    block b;

    jump_node *prv, *nxt, *up, *down;
    int sub_cnt;

    inline jump_node(const block& b, jump_node* prv=0, jump_node* nxt=0, jump_node* up=0, jump_node* down=0, int sub_cnt=0)
        : b(b), prv(prv), nxt(nxt), up(up), down(down), sub_cnt(sub_cnt) {}
};

class jump_table 
{
public:
    int stride;
public:
    jump_table(int stride=5);
    ~jump_table();
    jump_node* find(const block& b);
    void insert(jump_node* node);
    void remove(jump_node* node);

    void dump();

private:
    jump_node* search(const jump_node* node);
    void split(jump_node* node);
    void physically_remove(jump_node* node);
    jump_node* merge(jump_node* node);

    void reclaim(jump_node* node);
    void dump(const jump_node* node, int level);
private:
    jump_node* head;
};

#endif
