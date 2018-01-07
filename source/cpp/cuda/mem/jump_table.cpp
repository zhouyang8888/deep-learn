#include "jump_table.h"
#include <malloc.h>
#include <cassert>

jump_table::jump_table(int stride) : stride(stride)
{
    head = (jump_node*) malloc(sizeof(jump_node));
    head->prv = 0;
    head->nxt = 0;
    head->up = 0;
    head->down = 0;
}
jump_table::~jump_table()
{
    reclaim(head->nxt);
    free(head);
}
jump_node* jump_table::find(const block& b)
{
    jump_node* pb = new jump_node(&b);
    jump_node* ret = search(pb);
    delete pb;
    if (ret == head || *ret->b < b) {
        return 0;
    } else {
        return ret;
    }
}
jump_node* jump_table::search(const jump_node* node)
{
    jump_node* itr = head;
    while (true) {
        if (itr->nxt && *itr->nxt->b <= *node->b) {
            itr = itr->nxt;
        } else {
            if (itr->down) itr = itr->down;
            else {
                return itr;
            }
        }
    }
}
void jump_table::insert(const block* pb)
{
    jump_node* node = new jump_node(pb);
    jump_node* itr = search(node);

    node->prv = itr;
    node->nxt = itr->nxt;
    node->up = itr->up;
    node->down = 0;
    node->sub_cnt = 0;

    itr->nxt = node;
    if (node->nxt) node->nxt->prv = node;

    itr = node->up;
    while (itr) {
        itr->sub_cnt++;
        if (itr->sub_cnt > stride) {
            split(itr);
            itr = itr->up;
        } else {
            break;
        }
    }

    if (!itr) {
        jump_node* p = head->nxt;
        int first_line_cnt = 0;
        while (p) {
            ++first_line_cnt;
            p = p->nxt;
        }
        if (first_line_cnt > stride) {
            p = head->nxt;
            p->prv = 0;
            head->nxt = new jump_node(p->b, head, 0, 0, p, first_line_cnt);
            while (p) {
                p->up = head->nxt;
                p = p->nxt;
            }
            split(head->nxt);
        }
    }
}
/**
 * split subtree rooted at node.
 */
void jump_table::split(jump_node* node)
{
    jump_node* itr = node->down;
    int half_size = node->sub_cnt >> 1;

    while (half_size-- > 0) itr = itr->nxt;

    jump_node* newnode = new jump_node(itr->b, node, node->nxt, node->up, itr, node->sub_cnt - (node->sub_cnt >> 1));  //copy load;
    node->nxt = newnode;
    if (newnode->nxt) newnode->nxt->prv = newnode;
    itr->prv->nxt = 0;
    itr->prv = 0;

    while (itr) {
        itr->up = newnode;
        itr = itr->nxt;
    }

    node->sub_cnt >>= 1;
}
const block* jump_table::remove(const block& b)
{
    const block* ret = 0;
    jump_node* node = new jump_node(&b);

    jump_node* last_node = search(node);
    if (last_node != head && b == *last_node->b) {
        ret = physically_remove(last_node);
    }

    delete node;
    return ret;
}
const block* jump_table::physically_remove(jump_node* node)
{
    const block* ret = node->b;

    if (node->prv) node->prv->nxt = node->nxt;
    if (node->nxt) node->nxt->prv = node->prv;

    jump_node* pp = node->up;
    while (pp) {
        pp->sub_cnt--;
        if (pp->down == node) {
            pp->down = node->nxt;
            if (pp->down) pp->b = pp->down->b;
        }
        if (pp->sub_cnt > 0) {
            jump_node* toremove = merge(pp);
            if (toremove) {
                pp = toremove->up;
                delete toremove;
            } else {
                break;
            }
        } else {
            delete node;
            node = pp;
            pp = node->up;

            if (node->prv) node->prv->nxt = node->nxt;
            if (node->nxt) node->nxt->prv = node->prv;
        }
    }
    delete node;

    if (!pp) {
        // merege happend at upmost level.
        if (head->nxt && !head->nxt->nxt && head->nxt->down) {
            jump_node* nxt = head->nxt;

            head->nxt = nxt->down;
            head->nxt->prv = head;

            delete nxt;
            nxt = head->nxt;
            while (nxt) {
                nxt->up = 0;
                nxt = nxt->nxt;
            }

        }
    }

    return ret;
}
jump_node* jump_table::merge(jump_node* node)
{
    jump_node *first = 0;
    jump_node *second = 0;
    if (node->prv && node->prv != head && node->prv->sub_cnt + node->sub_cnt <= stride) {
        first = node->prv;
        second = node;
    } else if (node->nxt && node->nxt->sub_cnt + node->sub_cnt <= stride) {
        first = node;
        second = node->nxt;
    } 

    if (first && second) {
        jump_node* first_little_child = first->down;
        while (first_little_child->nxt) first_little_child = first_little_child->nxt;
        jump_node* second_child = second->down;
        first_little_child->nxt = second_child;
        second_child->prv = first_little_child;
        while (second_child) {
            second_child->up = first;
            second_child = second_child->nxt;
        }

        first->nxt = second->nxt;
        if (second->nxt) second->nxt->prv = first;

        return second;
    } else {
        return 0;
    }
}
void jump_table::reclaim(jump_node* node)
{
    if (node) {
        reclaim(node->down);
        reclaim(node->nxt);

        delete node;
    }
}
void jump_table::dump()
{
    jump_node* node = head->nxt;
    int level = 1;
    dump(node, level);
    std::cout << "====================" << std::endl;
}

void jump_table::dump(const jump_node* node, int level)
{
    while (node) {
        for (int i = 0; i < level; ++i) std::cout << "\t";
        node->b->dump();
        std::cout << std::endl;
        if (node->down) {
            dump(node->down, level + 1);
        }

        node = node->nxt;
    }
}


#ifdef __TEST_JUMP_TABLE__
#undef __TEST_JUMP_TABLE__

#include "mem_block.h"

void test_addr()
{
    jump_table t;

    mem_block b((void*)11, 77);

    jump_node* ret = t.find(b);
    assert(!ret);

    t.insert(&b);
    t.dump();

    ret = t.find(b);
    assert(ret);

    mem_block b2((void*)22, 66);
    t.insert(&b2);
    t.dump();

    mem_block b3((void*)33, 55);
    t.insert(&b3);
    t.dump();

    mem_block b4((void*)44, 44);
    t.insert(&b4);
    t.dump();

    mem_block b5((void*)55, 33);
    t.insert(&b5);
    t.dump();

    mem_block b6((void*)0, 88);
    t.insert(&b6);
    t.dump();

    mem_block b7((void*)40, 48);
    t.insert(&b7);
    t.dump();

    ret = t.find(b6);
    assert(ret);
    t.remove(b6);
    t.dump();

    ret = t.find(b7);
    assert(ret);
    t.remove(b7);
    t.dump();


    ret = t.find(b5);
    assert(ret);
    t.remove(b5);
    t.dump();

    ret = t.find(b4);
    assert(ret);
    t.remove(b4);
    t.dump();

    ret = t.find(b3);
    assert(ret);
    t.remove(b3);
    t.dump();

    ret = t.find(b);
    assert(ret);
    t.remove(b);
    t.dump();

    ret = t.find(b2);
    assert(ret);
    t.remove(b2);
    t.dump();

}

void test_size()
{
    jump_table t;

    mem_block2 b((void*)11, 77);

    jump_node* ret = t.find(b);
    assert(!ret);

    t.insert(&b);
    t.dump();

    ret = t.find(b);
    assert(ret);

    mem_block2 b2((void*)22, 66);
    t.insert(&b2);
    t.dump();

    mem_block2 b3((void*)33, 55);
    t.insert(&b3);
    t.dump();

    mem_block2 b4((void*)44, 44);
    t.insert(&b4);
    t.dump();

    mem_block2 b5((void*)55, 33);
    t.insert(&b5);
    t.dump();

    mem_block2 b6((void*)0, 88);
    t.insert(&b6);
    t.dump();

    mem_block2 b7((void*)40, 48);
    t.insert(&b7);
    t.dump();

    ret = t.find(b6);
    assert(ret);
    t.remove(b6);
    t.dump();

    ret = t.find(b7);
    assert(ret);
    t.remove(b7);
    t.dump();


    ret = t.find(b5);
    assert(ret);
    t.remove(b5);
    t.dump();

    ret = t.find(b4);
    assert(ret);
    t.remove(b4);
    t.dump();

    ret = t.find(b3);
    assert(ret);
    t.remove(b3);
    t.dump();

    ret = t.find(b);
    assert(ret);
    t.remove(b);
    t.dump();

    ret = t.find(b2);
    assert(ret);
    t.remove(b2);
    t.dump();

}

int main(int argc, char** argv)
{
    test_addr();
    test_size();
}

#endif

