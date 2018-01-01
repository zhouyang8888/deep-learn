/*
 * =====================================================================================
 *
 *       Filename:  mem.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/30 12时51分34秒
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

#include "hash.h"

struct m_info {
	void* p_h;
	void* p_d;
	int sz;

	struct m_info* prev;
	struct m_info* next;
};

struct d_global
class mem {
	public:
		m_info* head;
		inline m_info* alloc(int sz) {
			p_h = malloc(sz);
			assert(NULL != p_h);
			cudaMalloc(&p_d, sz);
		}
		void dealloc(m_info* info);
};
