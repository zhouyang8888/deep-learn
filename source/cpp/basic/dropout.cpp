/*
 * =====================================================================================
 *
 *       Filename:  dropout.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/18 17时00分13秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "dropout.h"
#include <cassert>
#include <iostream>

dropout::dropout(matrix* x, matrix* y, matrix* dx, matrix* dy, float prob)
	: x(x), y(y), dx(dx), dy(dy), prob(prob)

{
	assert(x->get_row() == y->get_row());
	assert(x->get_col() == y->get_col());
	mask = new matrix(x->get_row(), x->get_col());
}

dropout::~dropout()
{
	delete mask;
}

void dropout::eval()
{
	for (int i = 0; i < mask->get_row() * mask->get_col(); ++i) {
		if (((float) rand()) / RAND_MAX < prob) {
			mask->val[i] = 1;
			y->val[i] = x->val[i] / prob;
		} else {
			mask->val[i] = 0;
			y->val[i] = 0;
		}
	}
}

void dropout::bp()
{
	for (int i = 0; i < mask->get_row() * mask->get_col(); ++i) {
		if ( 1 == mask->val[i]) {
			dx->val[i] = dy->val[i] / prob;
		} else {
			dx->val[i] = 0;
		}
	}
}

void dropout::stat(const char* str)
{
	if (str) std::cout << str << std::endl;
	x->print("x:");
	y->print("y:");
	dy->print("dy:");
	dx->print("dx:");
}

#ifdef __TEST_DROPOUT__
#undef __TEST_DROPOUT__

int main(int argc, char** argv)
{
	srand(time(0));
	float* pf = new float[48];
	for (int i = 0; i < 48; ++i)
		pf[i] = i;

	float* pdf = new float[48];
	for (int i = 0; i < 48; ++i)
		pdf[i] = i;

	matrix* x = new matrix(pf, 2, 24);
	matrix* y = new matrix(2, 24);
	matrix* dx = new matrix(2, 24);
	matrix* dy = new matrix(pdf, 2, 24);

	dropout mp(x, y, dx, dy, 0.5f);
	mp.eval();
	mp.bp();
	mp.stat("dropout:");

	return 0;
}

#endif
