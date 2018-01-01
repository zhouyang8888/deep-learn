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
#pragma once
#pragma pack(4)

#include <string.h>
#include <malloc.h>

struct node {
	void* key;
	void* val;
	inline node(void* key, void* val) : key(key), val(val){};

	node* next;
};

class hash {
	private:
		node** table;
		int size;
		int cnt;

		int min_size;
	public:
		inline hash(int size=100, float rate=0.7f): min_size(size), size(size), rate(rate), cnt(0){
			table = (node**) malloc(sizeof(node*) * (size + 1));
			memset(table, 0, sizeof(node*) * (size + 1));
		}

	private:
		inline int offset(void* key) {
			uint64 addrv = (uint64) key;
			addrv >>= 2;
			return (addrv % size) + 1;
		}

	public:
		inline void* get(void* key) {
			int o = offset(key);
			node* pp = table[o];
			while (pp) {
				if (pp->key == key) {
					return pp->val;
				} else {
					pp = pp->next;
				}
			}
			return null;
		}
		inline bool insert(void* key, void* val) {
			int o = offset(key);
			node* pp = table[o];
			while (pp) {
				if (pp->key == key) {
					return false;
				} else {
					pp = pp->next;
				}
			}

			node* pnode = new node(key, val);
			pnode->next = table[o];
			table[o] = pnode;
			++cnt;
			if (cnt > size * rate) expand();
			return true;
		}
		inline bool remove(void* key) {
			int o = offset(key);
			node** pp = table + o;
			while (*pp) {
				if ((*pp)->key == key) {
					node* todelete = *pp;
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
			node** oldtable = table;

			size <<= 1;
			table = (node**) malloc(sizeof(node*) * (size + 1));
			memset(table, 0,  sizeof(node*) * (size + 1));

			for (int i = 1; i <= oldsize; ++i) {
				node* pnode = oldtable[i];
				while (pnode) {
					node* pnext = pnode->next;

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
			node** oldtable = table;

			size >>= 1;
			table = (node**) malloc(sizeof(node*) * (size + 1));
			memset(table, 0,  sizeof(node*) * (size + 1));

			for (int i = 1; i <= oldsize; ++i) {
				node* pnode = oldtable[i];
				while (pnode) {
					node* pnext = pnode->next;

					int o = offset(pnode->key);
					pnode->next = table[o];
					table[o] = pnode;

					pnode = pnext;
				}
			}

			free(oldtable);
		}
};
