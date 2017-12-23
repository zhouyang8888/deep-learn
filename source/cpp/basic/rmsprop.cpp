/*
 * =====================================================================================
 *
 *       Filename:  rmsprop.cpp
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

#include "rmsprop.h"
#include "constant.h"
#include <cmath>

rmsprop::rmsprop(int row, int col, float rho) : rho(rho)
{
	r = new matrix(row, col);
}

rmsprop::~rmsprop()
{
	delete r;
}

void rmsprop::update(matrix* theta, matrix* g, float lambda)
{
	for (int i = 0; i < r->get_row() * r->get_col(); ++i) {
		r->val[i] = rho * r->val[i] + (1 - rho) * g->val[i] * g->val[i];
		theta->val[i] -= (lambda * g->val[i] / sqrt(constant::epsilon + r->val[i]));
	}
}

