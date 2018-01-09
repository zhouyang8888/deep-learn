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

class hash_key {
public:
    virtual uint64_t hash_code() const =0;
    virtual bool operator==(const hash_key& other) const =0;
};

template<class hashkey, class value>
struct node {
	hashkey key;
	value val;
	inline node(const hashkey& key, const value& val) : key(key), val(val){};

	node* next;
};

template<class hashkey, class value>
class hash {
	private:
		node<hashkey, value>** table;
		int size;
		int cnt;

		int min_size;
        float rate;
	public:
		inline hash(int size=100, float rate=0.7f): min_size(size), size(size), rate(rate), cnt(0){
			table = (node<hashkey, value>**) malloc(sizeof(node<hashkey, value>*) * (size + 1));
			memset(table, 0, sizeof(node<hashkey, value>*) * (size + 1));
		}

	private:
		inline int offset(const hashkey& key) const {
            const hash_key& hkey = dynamic_cast<const hash_key&>(key);
			uint64_t addrv = hkey.hash_code();
			return (addrv % size) + 1;
		}

	public:
		inline const value* get(const hashkey& key) const {
			int o = offset(key);
			node<hashkey, value>* pp = table[o];
			while (pp) {
				if (pp->key == key) {
					return &pp->val;
				} else {
					pp = pp->next;
				}
			}
			return NULL;
		}
		inline bool insert(const hashkey& key, const value& val) {
			int o = offset(key);
			node<hashkey, value>* pp = table[o];
			while (pp) {
				if (pp->key == key) {
					return false;
				} else {
					pp = pp->next;
				}
			}

			node<hashkey, value>* pnode = new node<hashkey, value>(key, val);
			pnode->next = table[o];
			table[o] = pnode;
			++cnt;
			if (cnt > size * rate) expand();
			return true;
		}
		inline bool remove(const hashkey& key) {
			int o = offset(key);
			node<hashkey, value>** pp = table + o;
			while (*pp) {
				if ((*pp)->key == key) {
					node<hashkey, value>* todelete = *pp;
					*pp = (*pp)->next;
					delete todelete;
					--cnt;
					if (cnt < size * (1 - rate) && size > min_size) shrink();
					return true;
				} else {
					pp = &(*pp)->next;
				}
			}
			return false;
		}

	private:
		inline void expand() {
			int oldsize = size;
			node<hashkey, value>** oldtable = table;

			size <<= 1;
			table = (node<hashkey, value>**) malloc(sizeof(node<hashkey, value>*) * (size + 1));
			memset(table, 0,  sizeof(node<hashkey, value>*) * (size + 1));

			for (int i = 1; i <= oldsize; ++i) {
				node<hashkey, value>* pnode = oldtable[i];
				while (pnode) {
					node<hashkey, value>* pnext = pnode->next;

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
			node<hashkey, value>** oldtable = table;

			size >>= 1;
			table = (node<hashkey, value>**) malloc(sizeof(node<hashkey, value>*) * (size + 1));
			memset(table, 0,  sizeof(node<hashkey, value>*) * (size + 1));

			for (int i = 1; i <= oldsize; ++i) {
				node<hashkey, value>* pnode = oldtable[i];
				while (pnode) {
					node<hashkey, value>* pnext = pnode->next;

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
