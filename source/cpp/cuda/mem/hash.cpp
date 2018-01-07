/*
 * =====================================================================================
 *
 *       Filename:  hash.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/30 13时10分17秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifdef __TEST_HASH__
#undef __TEST_HASH__

#include "hash.h"
#include <iostream>
#include <cassert>

int main(int argc, char** argv)
{
    class intkey : public hash_key {
    public:
        int v;
        inline intkey(int i) : v(i) {};
        inline uint64_t hash_code() const {
            return v;
        }
        inline bool operator==(const hash_key& other) const {
            const intkey& o = dynamic_cast<const intkey&>(other);
            return v == o.v;
        }
    };

    hash<intkey, int> ht(5);
    for (int i = 0; i < 10; ++i)
        ht.insert(intkey(i), i);

    for (int i = 0; i < 10; ++i) {
        const int* p = ht.get(intkey(i));
        assert(p);

        std::cout << i << " => " << *p << std::endl;
    }

    const int* p = ht.get(intkey(-1));
    assert(!p);

    for (int i = 0; i < 5; ++i) {
        assert(ht.remove(intkey(i * 2)));
    }
    for (int i = 0; i < 10; ++i) {
        const int* p = ht.get(intkey(i));
        if (i % 2) {
            assert(p);
            std::cout << i << " => " << *p << std::endl;
        } else {
            assert(!p);
            std::cout << i << " => " << "NULL" << std::endl;
        }
    }
}

#endif
