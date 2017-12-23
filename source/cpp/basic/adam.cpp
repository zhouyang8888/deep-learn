/*
 * =====================================================================================
 *
 *       Filename:  adam.cpp
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

#include "adam.h"
#include "constant.h"
#include <cmath>

adam::adam(int row, int col, float rho1, float rho2) : rho1(rho1), rho2(rho2)
{
	s = new matrix(row, col);
	r = new matrix(row, col);
	rho1_power = rho1;
	rho2_power = rho2;
}

adam::~adam()
{
	delete s;
	delete r;
}

void adam::update(matrix* theta, matrix* g, float lambda)
{
	float alpha = 1 - rho1_power;
	float beta  = 1 - rho2_power;
	for (int i = 0; i < s->get_row() * s->get_col(); ++i) {
		s->val[i] = (rho1 * s->val[i] + (1 - rho1) * g->val[i]) / alpha;
		r->val[i] = (rho2 * r->val[i] + (1 - rho2) * g->val[i] * g->val[i]) / beta;

		theta->val[i] = theta->val[i] - lambda * s->val[i] / sqrt(r->val[i] + constant::epsilon);
	}
	rho1_power *= rho1;
	rho2_power *= rho2;
}
