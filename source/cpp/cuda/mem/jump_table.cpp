#include "jump_table.h"
#include <malloc.h>
#include <cassert>

jump_table::jump_table() : jump_table(5)
{

}
jump_table::jump_table(int stride) : stride(stride)
{
    assert(stride > 1);
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
jump_node* jump_table::eq(const block& b)
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
jump_node* jump_table::le(const block& b)
{
    jump_node* pb = new jump_node(&b);
    jump_node* ret = search(pb);
    delete pb;
    if (ret == head) {
        return 0;
    } else {
        return ret;
    }
}
jump_node* jump_table::ge(const block& b)
{
    jump_node* pb = new jump_node(&b);
    jump_node* ret = search(pb);
    delete pb;
    if (ret == head) {
        return first();
    } else {
        while (ret && *ret->b < b) {
            ret = next(ret);
        }
        return ret;
    }
}
jump_node* jump_table::first()
{
    jump_node* cur = head->nxt;
    if (!cur) return 0;

    while (cur->down) cur = cur->down;
    return cur;
}
jump_node* jump_table::next(jump_node* cur)
{
    if (cur->nxt) return cur->nxt;

    while (cur->up) {
        cur = cur->up;
        if (cur->nxt) {
            cur = cur->nxt;
            while (cur->down)
                cur = cur->down;
            return cur;
        }
    }
    return 0;
}
jump_node* jump_table::last()
{
    jump_node* cur = head;
LOOP:
    while (cur->nxt) cur = cur->nxt;
    if (cur->down) {
        cur = cur->down;
        goto LOOP;
    }

    if (cur != head) return cur;
    else return 0;
}
jump_node* jump_table::prev(jump_node* cur)
{
    if (cur->prv && cur->prv != head) return cur->prv;

    while (cur->up) {
        cur = cur->up;
        if (cur->prv && cur->prv != head) {
            cur = cur->prv;
            while (cur->down) {
                cur = cur->down;
                while (cur->nxt) cur = cur->nxt;
            };
            
            return cur;
        }
    }
    return 0;
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
    if (itr == head) {
        jump_node* f = first();
        if (f) {
            f->prv = node;
            node->nxt = f;

            if (head->nxt == f) head->nxt = node;
            else {
                    node->up = f->up;
                    f->up->down = node;
                    do {
                        f = f->up;
                        f->b = node->b;
                    } while (f->up);
            }
        } else {
            head->nxt = node;
            node->prv = head;
        }
    } else {
        node->prv = itr;
        node->nxt = itr->nxt;
        node->up = itr->up;
        node->down = 0;
        node->sub_cnt = 0;

        itr->nxt = node;
        if (node->nxt) node->nxt->prv = node;
    }

    itr = node->up;
    while (itr) {
        itr->sub_cnt++;
        if (itr->sub_cnt > stride) {
            split(itr);

            jump_node* left = merge(itr);
            if (left) {
                delete_node(left);
            }
            jump_node* right = merge(itr->nxt);
            if (right) {
                delete_node(right);
            }
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

            jump_node* right = merge(head->nxt->nxt);
            if (right) {
                delete_node(right);
            }
        }
    }
}
/**
 * split subtree rooted at node.
 */
void jump_table::split(jump_node* node)
{
    jump_node* itr = node->down;
    int half_size = (node->sub_cnt + 1) >> 1;

    for (int i = 0; i < half_size; ++i)
        itr = itr->nxt;

    jump_node* newnode = new jump_node(itr->b, node, node->nxt, node->up, itr, node->sub_cnt - half_size);  //copy load;
    node->nxt = newnode;
    if (newnode->nxt) newnode->nxt->prv = newnode;
    itr->prv->nxt = 0;
    itr->prv = 0;

    while (itr) {
        itr->up = newnode;
        itr = itr->nxt;
    }

    node->sub_cnt = half_size;
}
void jump_table::delete_node(jump_node* node)
{
    if (node) {
        if (node->prv) node->prv->nxt = node->nxt;
        if (node->nxt) node->nxt->prv = node->prv;
        jump_node* pp = node->up;
        if (pp) {
            if (pp->down == node) {
                pp->down = node->nxt;
                if (pp->down) {
                    jump_node* tmppp = pp;
                    do {
                        tmppp->b = tmppp->down->b;
                        tmppp = tmppp->up;
                    } while (tmppp && tmppp->b != tmppp->down->b);
                }
            }
            pp->sub_cnt--;
        }
        delete node;
    }
}
const block* jump_table::remove(const block& b)
{
    const block* ret = 0;
    jump_node* node = new jump_node(&b);

    jump_node* toremovenode = search(node);
    delete node;

    if (toremovenode != head && b == *toremovenode->b) {
        return remove(toremovenode);
    }
    return 0;
}
const block* jump_table::remove(jump_node* toremovenode) 
{ 
    const block* ret = toremovenode->b;
    jump_node* pp = 0;
    do {
        pp = toremovenode->up;
        delete_node(toremovenode);

        if (!pp) break;
        if (0 == pp->sub_cnt) toremovenode = pp;
        else toremovenode = merge(pp);
    } while (toremovenode);

    if (!pp) {
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
    if (!node) return 0;

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

        first->sub_cnt += second->sub_cnt;

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
        std::cout << node->sub_cnt << ";" << std::endl;
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
    jump_table t(3);

    mem_block b((void*)11, 77);

    jump_node* ret = t.eq(b);
    assert(!ret);

    t.insert(&b);
    t.dump();

    ret = t.eq(b);
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

    mem_block b8((void*)23, 65);
    ret = t.eq(b8);
    assert(!ret);

    ret = t.eq(b6);
    assert(ret);
    t.remove(b6);
    t.dump();

    ret = t.eq(b7);
    assert(ret);
    t.remove(b7);
    t.dump();


    ret = t.eq(b5);
    assert(ret);
    t.remove(b5);
    t.dump();

    ret = t.eq(b4);
    assert(ret);
    t.remove(b4);
    t.dump();

    ret = t.eq(b3);
    assert(ret);
    t.remove(b3);
    t.dump();

    ret = t.eq(b);
    assert(ret);
    t.remove(b);
    t.dump();

    ret = t.eq(b2);
    assert(ret);
    t.remove(b2);
    t.dump();

}

void test_size()
{
    jump_table t(3);

    mem_block2 b((void*)11, 77);

    jump_node* ret = t.eq(b);
    assert(!ret);

    t.insert(&b);
    t.dump();

    ret = t.eq(b);
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

    mem_block2 be1((void*)55, 32);
    t.insert(&be1);
    t.dump();


    mem_block2 be2((void*)55, 31);
    t.insert(&be2);
    t.dump();

    mem_block2 be3((void*)55, 30);
    t.insert(&be3);
    t.dump();


    mem_block2 be4((void*)55, 29);
    t.insert(&be4);
    t.dump();

    mem_block2 be5((void*)55, 28);
    t.insert(&be5);
    t.dump();


    mem_block2 be6((void*)55, 27);
    t.insert(&be6);
    t.dump();

    mem_block2 be7((void*)55, 26);
    t.insert(&be7);
    t.dump();


    mem_block2 be8((void*)55, 25);
    t.insert(&be8);
    t.dump();

    mem_block2 b6((void*)0, 88);
    t.insert(&b6);
    t.dump();

    mem_block2 b7((void*)40, 48);
    t.insert(&b7);
    t.dump();

    mem_block2 b8((void*)23, 65);
    ret = t.eq(b8);
    assert(!ret);
   
    ////////////////////////////////
    mem_block2 blocks[] = {{(void*)40, 39}, {(void*)55, 33}, {(void*)0, 99}, {(void*)0, 88}, {(void*)40, 0}};
    for (int i = 0; i < sizeof(blocks) / sizeof(mem_block2); ++i) {
        ret = t.ge(blocks[i]);
        std::cout << "ge:" << std::endl;
        blocks[i].dump();
        std::cout << " => ";
        if (ret)
            ret->b->dump();
        else
            std::cout << ret;
        std::cout << std::endl;
    }
    ////////////////////////////////

    ret = t.eq(b6);
    assert(ret);
    t.remove(b6);
    t.dump();

    ret = t.eq(b7);
    assert(ret);
    t.remove(b7);
    t.dump();


    ret = t.eq(b5);
    assert(ret);
    t.remove(b5);
    t.dump();

    ret = t.eq(b4);
    assert(ret);
    t.remove(b4);
    t.dump();

    ret = t.eq(b3);
    assert(ret);
    t.remove(b3);
    t.dump();

    ret = t.eq(b);
    assert(ret);
    t.remove(b);
    t.dump();

    ret = t.eq(b2);
    assert(ret);
    t.remove(b2);
    t.dump();
}
void print_tranverse(jump_table& t)
{
    std::cout << "Forward: " << std::endl;
    jump_node* itr = t.first();
    int i = 0;
    while (itr) {
        itr->b->dump();
        itr = t.next(itr);
        std::cout << " -> ";
        if (!(++i % 5)) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Backward: " << std::endl;
    itr = t.last();
    i = 0;
    while (itr) {
        itr->b->dump();
        itr = t.prev(itr);
        std::cout << " -> ";
        if (!(++i % 5)) std::cout << std::endl;
    }
    std::cout << std::endl;
}
void test_tranverse()
{
    jump_table t(3);

    mem_block b((void*)11, 77);

    jump_node* ret = t.eq(b);
    assert(!ret);
    print_tranverse(t);

    t.insert(&b);
    t.dump();
    print_tranverse(t);

    ret = t.eq(b);
    assert(ret);

    mem_block b2((void*)22, 66);
    t.insert(&b2);
    t.dump();
    print_tranverse(t);

    mem_block b3((void*)33, 55);
    t.insert(&b3);
    t.dump();
    print_tranverse(t);

    mem_block b4((void*)44, 44);
    t.insert(&b4);
    t.dump();
    print_tranverse(t);

    mem_block b5((void*)55, 33);
    t.insert(&b5);
    t.dump();
    print_tranverse(t);

    mem_block b6((void*)0, 88);
    t.insert(&b6);
    t.dump();
    print_tranverse(t);

    mem_block b7((void*)40, 48);
    t.insert(&b7);
    t.dump();
    print_tranverse(t);
}
void test_tranverse2()
{
    jump_table t(2);

    mem_block2 b((void*)11, 77);

    jump_node* ret = t.eq(b);
    assert(!ret);
    print_tranverse(t);

    t.insert(&b);
    t.dump();
    print_tranverse(t);

    ret = t.eq(b);
    assert(ret);

    mem_block2 b2((void*)22, 66);
    t.insert(&b2);
    t.dump();
    print_tranverse(t);

    mem_block2 b3((void*)33, 55);
    t.insert(&b3);
    t.dump();
    print_tranverse(t);

    mem_block2 b4((void*)44, 44);
    t.insert(&b4);
    t.dump();
    print_tranverse(t);

    mem_block2 b5((void*)55, 33);
    t.insert(&b5);
    t.dump();
    print_tranverse(t);

    mem_block2 b6((void*)0, 88);
    t.insert(&b6);
    t.dump();
    print_tranverse(t);

    mem_block2 b7((void*)40, 48);
    t.insert(&b7);
    t.dump();
    print_tranverse(t);
}
int main(int argc, char** argv)
{
    test_addr();
    test_size();
    test_tranverse();
    test_tranverse2();
}

#endif

