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
    jump_table();
    jump_table(int stride);
    ~jump_table();
    jump_node* eq(const block& b);
    jump_node* le(const block& b);
    jump_node* ge(const block& b);
    jump_node* first();
    jump_node* next(jump_node*);
    jump_node* last();
    jump_node* prev(jump_node*);
    void insert(const block* pb);
    const block* remove(const block& pb);
    const block* remove(jump_node* toremovenode);

    void dump();

private:
    jump_node* search(const jump_node* node);
    void split(jump_node* node);
    jump_node* merge(jump_node* node);

    void reclaim(jump_node* node);
    void dump(const jump_node* node, int level);
    void delete_node(jump_node* node);
private:
    jump_node* head;
};

#endif
