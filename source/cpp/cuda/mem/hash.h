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
#ifndef __hash__
#define __hash__

#pragma pack(4)

#include <string.h>
#include <malloc.h>
#include <cstdint>

class hashkey {
public:
    virtual uint64_t hash_code() = 0;
    virtual bool operator==(const hashkey& other) = 0;
};

template<class value>
struct node {
	hashkey key;
	value val;
	inline node(hashkey& key, value& val) : key(key), val(val){};

	node* next;
};

template<class value>
class hash {
	private:
		node<value>** table;
		int size;
		int cnt;

		int min_size;
        float rate;
	public:
		inline hash(int size=100, float rate=0.7f): min_size(size), size(size), rate(rate), cnt(0){
			table = (node<value>**) malloc(sizeof(node<value>*) * (size + 1));
			memset(table, 0, sizeof(node<value>*) * (size + 1));
		}

	private:
		inline int offset(hashkey& key) {
			uint64_t addrv = key.hash_code();
			addrv >>= 2;
			return (addrv % size) + 1;
		}

	public:
		inline value* get(hashkey& key) {
			int o = offset(key);
			node<value>* pp = table[o];
			while (pp) {
				if (pp->key == key) {
					return &pp->val;
				} else {
					pp = pp->next;
				}
			}
			return NULL;
		}
		inline bool insert(hashkey& key, value& val) {
			int o = offset(key);
			node<value>* pp = table[o];
			while (pp) {
				if (pp->key == key) {
					return false;
				} else {
					pp = pp->next;
				}
			}

			node<value>* pnode = new node<value>(key, val);
			pnode->next = table[o];
			table[o] = pnode;
			++cnt;
			if (cnt > size * rate) expand();
			return true;
		}
		inline bool remove(hashkey& key) {
			int o = offset(key);
			node<value>** pp = table + o;
			while (*pp) {
				if ((*pp)->key == key) {
					node<value>* todelete = *pp;
					*pp = (*pp)->next;
					delete todelete;
					--cnt;
					if (cnt < size * (1 - rate) && size > min_size) shrink();
					return true;
				} else {
					*pp = (*pp)->next;
				}
			}
			return false;
		}

	private:
		inline void expand() {
			int oldsize = size;
			node<value>** oldtable = table;

			size <<= 1;
			table = (node<value>**) malloc(sizeof(node<value>*) * (size + 1));
			memset(table, 0,  sizeof(node<value>*) * (size + 1));

			for (int i = 1; i <= oldsize; ++i) {
				node<value>* pnode = oldtable[i];
				while (pnode) {
					node<value>* pnext = pnode->next;

					int o = offset(pnode->key);
					pnode->next = table[o];
					table[o] = pnode;

					pnode = pnext;
				}
			}

			free(oldtable);
		}

		inline void shrink() {
			int oldsize = size;
			node<value>** oldtable = table;

			size >>= 1;
			table = (node<value>**) malloc(sizeof(node<value>*) * (size + 1));
			memset(table, 0,  sizeof(node<value>*) * (size + 1));

			for (int i = 1; i <= oldsize; ++i) {
				node<value>* pnode = oldtable[i];
				while (pnode) {
					node<value>* pnext = pnode->next;

					int o = offset(pnode->key);
					pnode->next = table[o];
					table[o] = pnode;

					pnode = pnext;
				}
			}

			free(oldtable);
		}
};

#endif
