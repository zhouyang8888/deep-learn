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
    jump_node* pb = new jump_node(b);
    jump_node* ret = search(pb);
    delete pb;
    if (ret == head || ret->b < b) {
        return 0;
    } else {
        return ret;
    }
}
jump_node* jump_table::search(const jump_node* node)
{
    jump_node* itr = head;
    while (true) {
        if (itr->nxt && itr->nxt->b <= node->b) {
            itr = itr->nxt;
        } else {
            if (itr->down) itr = itr->down;
            else {
                return itr;
            }
        }
    }
}
void jump_table::insert(jump_node* node)
{
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
void jump_table::remove(jump_node* node)
{
    jump_node* last_node = search(node);
    if (last_node == head || last_node->b < node->b) return;

    physically_remove(last_node);
}
void jump_table::physically_remove(jump_node* node)
{
    if (!node->prv) {
        block b = node->b;
        node->b = node->nxt->b;
        node = node->nxt;
        node->b = b;
    }
    
    node->prv->nxt = node->nxt;
    if (node->nxt) node->nxt->prv = node->prv;

    jump_node* pp = node->up;
    delete node;
    
    while (pp) {
        pp->sub_cnt--;
        jump_node* toremove = merge(pp);
        if (toremove) {
            pp = toremove->up;
            delete toremove;
        } else {
            break;
        }
    }

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
        node->b.dump();
        std::cout << std::endl;
        if (node->down) {
            dump(node->down, level + 1);
        }

        node = node->nxt;
    }
}


#ifdef __TEST_JUMP_TABLE__
#undef __TEST_JUMP_TABLE__

int main(int argc, char** argv)
{
    jump_table t;

    block b((void*)11, 32);

    jump_node* ret = t.find(b);
    assert(!ret);

    t.insert(new jump_node(b));
    t.dump();

    ret = t.find(b);
    assert(ret);

    block b2((void*)22, 32);
    t.insert(new jump_node(b2));
    t.dump();

    block b3((void*)33, 32);
    t.insert(new jump_node(b3));
    t.dump();

    block b4((void*)44, 32);
    t.insert(new jump_node(b4));
    t.dump();

    block b5((void*)55, 32);
    t.insert(new jump_node(b5));
    t.dump();

    block b6((void*)0, 32);
    t.insert(new jump_node(b6));
    t.dump();

    block b7((void*)40, 32);
    t.insert(new jump_node(b7));
    t.dump();

    ret = t.find(b6);
    assert(ret);
    t.remove(ret);
    t.dump();

    ret = t.find(b7);
    assert(ret);
    t.remove(ret);
    t.dump();


    ret = t.find(b5);
    assert(ret);
    t.remove(ret);
    t.dump();

    ret = t.find(b4);
    assert(ret);
    t.remove(ret);
    t.dump();

    ret = t.find(b3);
    assert(ret);
    t.remove(ret);
    t.dump();

    ret = t.find(b);
    assert(ret);
    t.remove(ret);
    t.dump();

    ret = t.find(b2);
    assert(ret);
    t.remove(ret);
    t.dump();

}

#endif
