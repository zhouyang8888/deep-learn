#ifndef __jump_table__
#define __jump_table__

#include <iostream>
#include <cstdint>
#include <cassert>
#include "block.h"

struct jump_node 
{
    const block* b;

    jump_node *prv, *nxt, *up, *down;
    int sub_cnt;

    inline jump_node(const block* b, jump_node* prv=0, jump_node* nxt=0, jump_node* up=0, jump_node* down=0, int sub_cnt=0)
        : b(b), prv(prv), nxt(nxt), up(up), down(down), sub_cnt(sub_cnt) {}
};

class jump_table 
{
public:
    int stride;
public:
    jump_table(int stride=5);
    ~jump_table();
    jump_node* eq(const block& b);
    jump_node* le(const block& b);
    jump_node* ge(const block& b);
    void insert(const block* pb);
    const block* remove(const block& pb);

    void dump();

private:
    jump_node* search(const jump_node* node);
    void split(jump_node* node);
    const block* physically_remove(jump_node* node);
    jump_node* merge(jump_node* node);

    void reclaim(jump_node* node);
    void dump(const jump_node* node, int level);
private:
    jump_node* head;
};

#endif
