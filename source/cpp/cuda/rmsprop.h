/*
 * =====================================================================================
 *
 *       Filename:  rmsprop.h
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

#ifndef __rmsprop__
#define __rmsprop__
#include "matrix0.h"

class rmsprop {
	private:
		matrix* r;
		float rho;
	public:
		rmsprop(int row, int col, float rho=0.999f);
		~rmsprop();

		void update(matrix* theta, matrix* g, float lambda);
};

#endif

