/*
 * =====================================================================================
 *
 *       Filename:  momentum.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/08 13时40分54秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __momentum__
#define __momentum__
#include "matrix0.h"

class momentum {
	private:
		matrix* v;
		float alpha;
	public:
		momentum(int row, int col, float alpha=0.9f);
		~momentum();

		void update(matrix* theta, matrix* g, float lambda);
};

#endif

