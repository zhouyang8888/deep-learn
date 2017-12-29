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

#include "momentum.h"

momentum::momentum(int row, int col, float alpha) : alpha(alpha)
{
	v = new matrix(row, col);
}

momentum::~momentum()
{
	delete v;
}

void momentum::update(matrix* theta, matrix* g, float lambda)
{
	v->multiply_num(alpha);
	g->multiply_num(lambda);
	v->minus(*g);

	theta->plus(*v);
}


